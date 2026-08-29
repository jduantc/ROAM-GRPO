#!/usr/bin/env python3
"""Per-problem exemplar generator: clean panel (no title/border), annotated panel
(each x labeled with its completion index G#/R#), and a completions export (json+md).

Requires persisted embeddings from recompute_diversity.py:
    <emb_dir>/<tag>__<adapter>.emb_f16.npy    [N, dim] float16
    <emb_dir>/<tag>__<adapter>.index.jsonl     aligned rows

Usage:
  python fig_exemplar.py --pid test/counting_and_probability/876.json \
     --van passk16_van/van_g4500 --mc8 passk16_mc8/mc8_g4500 \
     --problem_src ../data/math_eval.jsonl --tag math876 --label MATH-L5
"""
import argparse, json, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"]})
HERE = pathlib.Path(__file__).resolve().parent

GRPO_C, ROAM_C = "#2e7d32", "#c0392b"


def load_adapter(emb_dir, key, correct_only=False):
    """key = '<tag>/<adapter>'. Returns dict problem_id -> {'E':[m,dim], 'correct':[m] bool}."""
    stem = key.replace("/", "__")
    E = np.load(emb_dir / f"{stem}.emb_f16.npy").astype(np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    idx = [json.loads(l) for l in open(emb_dir / f"{stem}.index.jsonl")]
    by_prob = {}
    for r in idx:
        if correct_only and not r["is_correct"]:
            continue
        by_prob.setdefault(r["problem_id"], {"rows": [], "correct": []})
        by_prob[r["problem_id"]]["rows"].append(r["row"])
        by_prob[r["problem_id"]]["correct"].append(bool(r["is_correct"]))
    out = {}
    for pid, d in by_prob.items():
        out[pid] = {"E": E[np.asarray(d["rows"])], "correct": np.asarray(d["correct"])}
    return out


def pair_div(E):
    """Mean pairwise cosine distance among rows of normalized E (>=2 rows), else None."""
    m = len(E)
    if m < 2:
        return None
    S = E @ E.T
    return 1.0 - float((S.sum() - np.trace(S)) / (m * (m - 1)))


def mds2d(E):
    from sklearn.manifold import MDS
    D = 1.0 - (E @ E.T)
    np.fill_diagonal(D, 0.0); D = np.clip(D, 0, None)
    return MDS(n_components=2, dissimilarity="precomputed", normalized_stress="auto",
               random_state=0, n_init=4, max_iter=500).fit_transform(D)


def _hull(ax, pts, color, alpha):
    if len(pts) < 3:
        return
    try:
        from scipy.spatial import ConvexHull
        v = pts[ConvexHull(pts).vertices]
        ax.fill(v[:, 0], v[:, 1], facecolor=color, alpha=alpha, edgecolor=color,
                linewidth=1.0, zorder=1)
    except Exception:
        pass


def dump_path(logs_dir, key):
    tag, adapter = key.split("/")
    return logs_dir / tag / f"{adapter}_completions.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--van", required=True, help="GRPO adapter key: '<tag>/<adapter>'")
    ap.add_argument("--mc8", required=True, help="ROAM adapter key: '<tag>/<adapter>'")
    ap.add_argument("--problem_src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--emb_dir", default=str(HERE / "runs/_logs/divrecompute/emb"),
                    help="directory with persisted embeddings from recompute_diversity.py")
    ap.add_argument("--logs_dir", default=str(HERE / "runs/_logs"),
                    help="directory with completion dump files")
    ap.add_argument("--out_dir", default=str(HERE / "runs/_logs/divrecompute/plots"),
                    help="output directory for figures and exports")
    args = ap.parse_args()

    emb_dir = pathlib.Path(args.emb_dir)
    logs_dir = pathlib.Path(args.logs_dir)
    OUT = pathlib.Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    van = load_adapter(emb_dir, args.van)
    mc8 = load_adapter(emb_dir, args.mc8)
    ve = van[args.pid]["E"][van[args.pid]["correct"]]
    me = mc8[args.pid]["E"][mc8[args.pid]["correct"]]
    vci = np.where(van[args.pid]["correct"])[0]
    mci = np.where(mc8[args.pid]["correct"])[0]
    XY = mds2d(np.vstack([ve, me]))
    vx, mx = XY[:len(ve)], XY[len(ve):]
    dv, dm = pair_div(ve), pair_div(me)

    raw = {}
    for tag_m, key in (("GRPO", args.van), ("ROAM", args.mc8)):
        for r in (json.loads(l) for l in open(dump_path(logs_dir, key))):
            if r["id"] == args.pid:
                raw[tag_m] = r; break
    gold = raw["GRPO"]["answer"]
    ptext = None
    for l in open(HERE / args.problem_src):
        r = json.loads(l)
        if str(r.get("id", "")) == args.pid:
            ptext = r.get("problem"); break

    def base(ax):
        _hull(ax, mx, ROAM_C, 0.18); _hull(ax, vx, GRPO_C, 0.18)
        ax.scatter(vx[:, 0], vx[:, 1], s=90, c=GRPO_C, marker="x", linewidths=1.6, zorder=3)
        ax.scatter(mx[:, 0], mx[:, 1], s=90, c=ROAM_C, marker="x", linewidths=1.6, zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # clean panel
    fig, ax = plt.subplots(figsize=(6, 5.4)); base(ax)
    ax.legend(handles=[Line2D([0], [0], marker="x", color=GRPO_C, lw=0, markeredgewidth=2, label="GRPO"),
                       Line2D([0], [0], marker="x", color=ROAM_C, lw=0, markeredgewidth=2, label="ROAM-GRPO")],
              fontsize=12, loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(OUT / f"{args.tag}_clean.png", dpi=200, bbox_inches="tight")

    # annotated panel
    fig, ax = plt.subplots(figsize=(7.5, 6.5)); base(ax)
    for (x, y), ci in zip(vx, vci):
        ax.annotate(f"G{ci}", (x, y), fontsize=7, color=GRPO_C, weight="bold",
                    xytext=(3, 3), textcoords="offset points")
    for (x, y), ci in zip(mx, mci):
        ax.annotate(f"R{ci}", (x, y), fontsize=8, color=ROAM_C, weight="bold",
                    xytext=(4, 4), textcoords="offset points")
    short = args.pid.replace("test/", "").replace(".json", "")
    ax.set_title(f"{short}  (GRPO d={dv:.3f}, ROAM-GRPO d={dm:.3f})  —  Gx/Rx = completion index",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"{args.tag}_annotated.png", dpi=200, bbox_inches="tight")

    # completions export
    rows = []
    for method, idxs, coords in (("GRPO", vci, vx), ("ROAM", mci, mx)):
        comps = raw[method]["completions"]
        for ci, (x, y) in zip(idxs, coords):
            rows.append({"label": f"{'G' if method=='GRPO' else 'R'}{int(ci)}", "method": method,
                         "completion_idx": int(ci), "x": round(float(x), 4), "y": round(float(y), 4),
                         "is_correct": True, "text": comps[int(ci)]})
    export = {"problem_id": args.pid, "problem": ptext, "gold": gold,
              "dataset": args.label, "grpo": args.van, "roam": args.mc8,
              "diversity_among_correct": {"GRPO": round(dv, 4), "ROAM": round(dm, 4)},
              "note": "x/y = 2D MDS coords in the figure; label G#/R# = GRPO/ROAM completion index; "
                      "only correct completions.", "completions": rows}
    json.dump(export, open(OUT / f"{args.tag}_completions.json", "w"), indent=2)
    with open(OUT / f"{args.tag}_completions.md", "w") as f:
        f.write(f"# {short} — correct completions\n\n")
        f.write(f"**Problem:** {ptext}\n\n**Answer:** {gold}\n\n")
        f.write(f"GRPO d={dv:.3f} · ROAM-GRPO d={dm:.3f}. Labels G#/R# match the annotated figure.\n")
        for r in rows:
            f.write(f"\n---\n\n## {r['label']}  ({r['method']}, xy=({r['x']:+.3f},{r['y']:+.3f}))\n\n")
            f.write(r["text"].strip() + "\n")
    print(f"[{args.tag}] GRPO {len(vci)} correct (d={dv:.3f}), ROAM {len(mci)} correct (d={dm:.3f}) "
          f"-> {args.tag}_clean/annotated.png + completions.json/md")


if __name__ == "__main__":
    main()
