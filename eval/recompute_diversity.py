#!/usr/bin/env python3
"""Offline diversity + embedding computation for paper plots.

No re-generation, reads saved K=16 completion dumps that passk_eval.py wrote
alongside eval (runs/_logs/passk16_<tag>/<adapter>_completions.jsonl), and:

  1. Re-grades correctness with the same grader passk_eval uses
     (_Scorer + extract_final_answer + answer_matches).
  2. Re-embeds every completion with the same embedder as selfdistill/passk_eval
     (Qwen3-Embedding-0.6B, same instruct prompt, _strip_think, normalize)
  3. Recomputes `diversity_among_correct` and additionally `diversity_among_all` (over all K completions).
  4. Persists the full per-adapter embedding matrix (float16) + a parallel index so the
     UMAP/t-SNE/collapse plots can be built again downstream
"""
import argparse, glob, json, os, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "roam"))

from passk_eval import _Scorer
from train import extract_final_answer, _strip_think


def make_embedder(device, cap):
    """Mirror selfdistill._embedder / passk_eval's embed, but with a configurable char cap.

    Identical model, instruct prompt, normalize_embeddings, float64 output; the ONLY change
    is `t[:cap]` instead of the hardcoded `t[:4000]`.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    reason = ("Instruct: Represent the core solution strategy and reasoning approach of this "
              "text (the sequence of mathematical steps and technique used), not its tone, "
              "reasoning narration style, wording, notation, verbosity, or formatting. Two "
              "texts should be close only if they use the same strategy.\nQuery:")
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device); enc.eval()

    def embed(texts):
        with torch.no_grad():
            return enc.encode([t[:cap] for t in texts], convert_to_numpy=True,
                              normalize_embeddings=True, prompt=reason,
                              show_progress_bar=False, batch_size=8).astype(np.float64)
    return embed


def mean_pairwise_cos_dist(E):
    """1 - mean off-diagonal cosine similarity among the rows of (normalized) E.

    EXACTLY passk_eval.py:227-229. Requires len(E) >= 2.
    """
    S = E @ E.T
    m = len(E)
    return 1.0 - float((S.sum() - np.trace(S)) / (m * (m - 1)))


def discover_dumps(logs_dir, only=None):
    """Return [(tag, adapter_name, jsonl_path)] for every completions dump found."""
    out = []
    for d in sorted(glob.glob(str(logs_dir / "passk16_*"))):
        dp = pathlib.Path(d)
        if not dp.is_dir():
            continue
        tag = dp.name
        if only and tag not in only:
            continue
        for f in sorted(glob.glob(str(dp / "*_completions.jsonl"))):
            adapter = pathlib.Path(f).name[:-len("_completions.jsonl")]
            out.append((tag, adapter, f))
    return out


def load_dump(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def process_adapter(tag, adapter, path, scorer, caps, emb_dir, device):
    rows = load_dump(path)                      # [{id, answer, completions[K]}]
    # flatten, remembering problem boundaries
    stripped, meta = [], []                     # meta row -> (prob_idx, comp_idx)
    per_prob = []                               # list of (id, gold, [global_row_indices])
    for pi, r in enumerate(rows):
        gold = r["answer"]
        idxs = []
        for ci, c in enumerate(r["completions"]):
            idxs.append(len(stripped))
            stripped.append(_strip_think(c))    # strip BEFORE the cap, as the original does
            meta.append((pi, ci))
        per_prob.append((r.get("id", ""), gold, idxs))

    # correctness: identical to passk_eval (extract_final_answer then _Scorer.correct)
    is_correct = np.zeros(len(stripped), dtype=bool)
    for pi, r in enumerate(rows):
        gold = r["answer"]
        base = per_prob[pi][2][0]
        for ci, c in enumerate(r["completions"]):
            is_correct[base + ci] = scorer.correct(extract_final_answer(c), gold)

    n_total = len(stripped)
    n_correct = int(is_correct.sum())
    # pass@1 / pass@k(any) — cross-check vs original JSON
    pass1 = n_correct / n_total
    n_any = sum(1 for _, _, idxs in per_prob if is_correct[idxs].any())
    passk_any = n_any / len(per_prob)

    result = {"adapter": adapter, "n_probs": len(per_prob), "n_completions": n_total,
              "pass@1": pass1, "pass@k_any": passk_any, "by_cap": {}}
    emb_full = None

    for cap in caps:
        embed = make_embedder(device, cap)
        E = embed(stripped)                     # [N, dim] float64, normalized
        if cap == max(caps):
            emb_full = E                        # persist the widest-cap matrix
        div_correct, div_all = [], []
        pp_div_correct = []                     # per-problem, for distribution plots
        for _id, gold, idxs in per_prob:
            idxs = np.asarray(idxs)
            cidx = idxs[is_correct[idxs]]
            if len(cidx) >= 2:
                dc = mean_pairwise_cos_dist(E[cidx])
                div_correct.append(dc); pp_div_correct.append((_id, dc, int(len(cidx))))
            if len(idxs) >= 2:
                div_all.append(mean_pairwise_cos_dist(E[idxs]))
        result["by_cap"][str(cap)] = {
            "diversity_among_correct": float(np.mean(div_correct)) if div_correct else None,
            "n_probs_with_ge2_correct": len(div_correct),
            "diversity_among_all": float(np.mean(div_all)) if div_all else None,
            "n_probs_with_ge2_all": len(div_all),
        }
        del embed

    # persist embeddings (widest cap) + index
    emb_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}__{adapter}"
    np.save(emb_dir / f"{stem}.emb_f16.npy", emb_full.astype(np.float16))
    with open(emb_dir / f"{stem}.index.jsonl", "w") as ix:
        for row, (pi, ci) in enumerate(meta):
            _id, gold, _ = per_prob[pi]
            ix.write(json.dumps({"row": row, "problem_id": _id, "gold": gold,
                                 "completion_idx": ci, "is_correct": bool(is_correct[row]),
                                 "char_len": len(rows[pi]["completions"][ci])}) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default=str(HERE / "runs/_logs"))
    ap.add_argument("--caps", default="4000,12000", help="comma char caps; widest is persisted")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="", help="comma tags to restrict to (dir names)")
    ap.add_argument("--out", default=str(HERE / "runs/_logs/divrecompute/summary.json"))
    args = ap.parse_args()

    logs_dir = pathlib.Path(args.logs_dir)
    caps = [int(x) for x in args.caps.split(",")]
    only = set(t for t in args.only.split(",") if t) or None
    out_path = pathlib.Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    emb_dir = out_path.parent / "emb"

    dumps = discover_dumps(logs_dir, only)
    print(f"[divrecompute] {len(dumps)} adapters, caps={caps}, device={args.device}")
    for tag, adapter, path in dumps:
        print(f"  - {tag}/{adapter}")

    # merge into any existing summary so re-runs are incremental
    summary = {}
    if out_path.exists():
        summary = json.load(open(out_path))

    scorer = _Scorer()
    try:
        for tag, adapter, path in dumps:
            key = f"{tag}/{adapter}"
            print(f"\n[divrecompute] {key}", flush=True)
            res = process_adapter(tag, adapter, path, scorer, caps, emb_dir, args.device)
            # attach the original stored diversity for the sanity gate
            orig_json = logs_dir / f"{tag}.json"
            orig_div = None
            if orig_json.exists():
                oj = json.load(open(orig_json))
                if adapter in oj:
                    orig_div = oj[adapter].get("diversity_among_correct")
            res["orig_diversity_among_correct_cap4000"] = orig_div
            c4 = res["by_cap"].get("4000", {}).get("diversity_among_correct")
            if orig_div is not None and c4 is not None:
                res["sanity_delta_cap4000"] = c4 - orig_div
                flag = "OK" if abs(c4 - orig_div) < 1e-3 else "**MISMATCH**"
                print(f"    sanity cap4000: recompute={c4:.5f} orig={orig_div:.5f} "
                      f"delta={c4-orig_div:+.5f} {flag}")
            for cap in caps:
                b = res["by_cap"][str(cap)]
                print(f"    cap{cap}: div_correct={b['diversity_among_correct']} "
                      f"(ge2={b['n_probs_with_ge2_correct']})  "
                      f"div_all={b['diversity_among_all']}")
            summary[key] = res
            json.dump(summary, open(out_path, "w"), indent=2)
    finally:
        scorer.close()
    print(f"\n[divrecompute] DONE -> {out_path}\n  embeddings -> {emb_dir}")


if __name__ == "__main__":
    main()
