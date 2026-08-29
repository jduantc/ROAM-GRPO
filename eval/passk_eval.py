#!/usr/bin/env python3
"""pass@k + diversity eval on a held-out test set.

For each named checkpoint (base + one per adapter), generates K plain-prompt completions
per problem via the vLLM+LoRA server, computes:
  pass@1   = mean per-completion correctness
  pass@k   = fraction of problems with >=1 correct among the K samples (using Chen estimator formula)
  dispersion = mean pairwise cosine distance of embeddings among the correct completions per problem
Runs standalone against its own vLLM server (default port 8010) so it doesn't collide with
servers used by the self-distill loop (8000/8001).
"""
import argparse, json, pathlib, subprocess, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BASE = "Qwen/Qwen3.5-2B"


# symbolic correctness scoring 
# answer_matches() runs sympy, which can hang indefinitely (GIL-holding C loops) on
# pathological LaTeX. We use a separate worker process that we
# can hard-kill. Also applies a cheap length guard first (clean gold answers are short, so
# an 80+ char extracted answer is not valid, this prevents hanging also).
def _match_worker(pred, gold):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "roam"))
    from train import answer_matches
    return bool(answer_matches(pred, gold))


class _Scorer:
    def __init__(self):
        import multiprocessing as mp
        self._ctx = mp.get_context("spawn")
        self.ex = None

    def _pool(self):
        import concurrent.futures as cf
        if self.ex is None:
            self.ex = cf.ProcessPoolExecutor(max_workers=1, mp_context=self._ctx)
        return self.ex

    def correct(self, extracted_answer, gold, timeout=6):
        import concurrent.futures as cf
        a = extracted_answer
        if a is None or len(a) > 80:          # cheap guard kills the hang source
            return False
        fut = self._pool().submit(_match_worker, a, gold)
        try:
            return bool(fut.result(timeout=timeout))
        except cf.TimeoutError:               # hung sympy: hard-kill the worker, rebuild pool
            ex = self.ex; self.ex = None
            for p in list(getattr(ex, "_processes", {}).values()):
                try: p.kill()
                except Exception: pass
            ex.shutdown(wait=False, cancel_futures=True)
            return False
        except Exception:
            return False

    def close(self):
        if self.ex is not None:
            self.ex.shutdown(wait=False, cancel_futures=True)
            self.ex = None


def wait_health(port, proc, logf, timeout_s=1800):
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"http://localhost:{port}/health/", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        if proc.poll() is not None:
            print(open(logf).read()[-3000:])
            raise RuntimeError("server died during startup")
        time.sleep(5)
    raise RuntimeError("server never became healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True,
                    help="comma list of name=path (path='' or 'base' -> no adapter)")
    ap.add_argument("--n_prob", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k_list", default=None,
                    help="comma-sep k values for the unbiased pass@k estimator "
                         "(Chen et al. 1 - C(n-c,k)/C(n,k)), computed from the K generated "
                         "samples, e.g. 1,4,8,16. Each must be <= --k.")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--out", default=str(HERE / "data/passk_results.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", default="../data/math_eval.jsonl",
                    help="held-out problem file (relative to eval/). Default = 400 MATH L5 "
                         "test problems, disjoint from training set.")
    ap.add_argument("--base_model", default=BASE,
                    help="base model to serve. Default Qwen3.5-9B keeps every prior run\n"
                         "byte-identical. Set this to evaluate a SMALLER/OLDER model that ran\n"
                         "the same self-distillation procedure on its own completions.")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="open a <think> block in the eval prompt so the policy reasons then "
                         "answers. MUST match how the adapter was trained, or you evaluate a "
                         "different prompt distribution than you trained (see NEXT_RUN_HANDOFF §2).")
    args = ap.parse_args()

    import sys, os
    ROAM_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "roam")
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, ROAM_DIR)
    import train as _M
    _M.ENABLE_THINKING = args.enable_thinking
    from train import build_prompt_column, extract_final_answer, _strip_think
    def _embedder(device, cap=4000):
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
    from transformers import AutoTokenizer
    import requests

    tok = AutoTokenizer.from_pretrained(args.base_model)
    # Determines whether base model is multimodal (Qwen3.5 tree) (*ForConditionalGeneration) or a plain text-only
    # causal LM. Thus decides whether LoRA needs remapping to `language_model.model.layers.*`.
    # Note: remapping a text-only adapter would rename nothing that vLLM looks for
    from transformers import AutoConfig
    _arch = (AutoConfig.from_pretrained(args.base_model).architectures or [""])[0]
    NEEDS_REMAP = _arch.endswith("ForConditionalGeneration")
    print(f"[passk] base={args.base_model} arch={_arch} lora_remap={NEEDS_REMAP}")
    rows = [json.loads(l) for l in open(HERE / args.heldout)]
    rng = np.random.default_rng(args.seed)
    probs = [rows[i] for i in rng.choice(len(rows), args.n_prob, replace=False)]
    prompts = [build_prompt_column(tok, r["problem"]) for r in probs]
    print(f"[passk] {len(probs)} held-out level-5 problems (never seen by GRPO/self-distill), K={args.k}")

    env = dict(**__import__("os").environ)
    logf = f"/tmp/passk_server_{args.port}.log"
    proc = subprocess.Popen(
        ["python", str(HERE / "vllm_serve_lora.py"), "--model", args.base_model,
         "--port", str(args.port),
         "--tensor_parallel_size", "1", "--gpu_memory_utilization", "0.85",
         "--max_model_len", "8192", "--enforce_eager", "True"],
        env={**env, "CUDA_VISIBLE_DEVICES": args.gpu},
        stdout=open(logf, "w"), stderr=subprocess.STDOUT)
    try:
        wait_health(args.port, proc, logf)
        print("[passk] server healthy")

        adapters = []
        for i, spec in enumerate(args.adapters.split(",")):
            name, path = spec.split("=", 1)
            adapters.append((name, None if path in ("", "base") else path, i + 1))

        remapped = {}
        for name, path, lora_id in adapters:
            if path is None:
                continue
            if not NEEDS_REMAP:
                remapped[name] = path      # text-only base: vLLM wants the keys as-saved
                continue
            # Tag the cache by base model: reusing a remap produced for a DIFFERENT base would
            # serve the wrong key layout, and the adapter would no-op with no error.
            tag = args.base_model.replace("/", "_")
            rp = str(HERE / f"runs/_vllm_adapter_eval_{name}__{tag}")
            legacy = str(HERE / f"runs/_vllm_adapter_eval_{name}")
            if (pathlib.Path(legacy) / "adapter_model.safetensors").exists() \
                    and args.base_model == BASE:
                rp = legacy                # keep reusing pre-existing 9B remaps unchanged
            if not (pathlib.Path(rp) / "adapter_model.safetensors").exists():
                raise RuntimeError(
                    f"LoRA adapter at {path} needs key remapping for multimodal base "
                    f"{args.base_model}, but no remapped adapter found at {rp}. "
                    f"For text-only bases (e.g. Qwen3.5-2B), remapping is not needed.")
            remapped[name] = rp

        embed = _embedder(f"cuda:{args.gpu}")
        scorer = _Scorer()
        results = {}
        dump_dir = pathlib.Path(args.out).with_suffix("")
        dump_dir.mkdir(parents=True, exist_ok=True)
        for name, path, lora_id in adapters:
            body_base = {"n": args.k, "temperature": 1.0, "top_p": 1.0, "max_tokens": args.max_new_tokens}
            if path is not None:
                body_base["lora_path"] = remapped[name]; body_base["lora_id"] = lora_id
            print(f"[passk] generating: {name} (adapter={'none' if path is None else path})")
            comps_by_prob = []
            CH = 16
            for c0 in range(0, len(prompts), CH):
                chunk_prompts = prompts[c0:c0 + CH]
                r = requests.post(f"http://localhost:{args.port}/generate/",
                                   json={**body_base, "prompts": chunk_prompts}, timeout=3600)
                r.raise_for_status()
                ids = r.json()["completion_ids"]
                texts = [tok.decode(x, skip_special_tokens=True) for x in ids]
                for pi in range(len(chunk_prompts)):
                    comps_by_prob.append(texts[pi * args.k:(pi + 1) * args.k])
                print(f"  {min(c0 + CH, len(prompts))}/{len(prompts)} problems generated", flush=True)

            # persist raw completions before scoring, so a scoring problem never
            # wastes expensive generation again
            with open(dump_dir / f"{name}_completions.jsonl", "w") as df:
                for r, comps in zip(probs, comps_by_prob):
                    df.write(json.dumps({"id": r.get("id",""), "answer": r["answer"],
                                         "completions": comps}, ensure_ascii=False) + "\n")

            n_correct_total = n_total = n_prob_any_correct = 0
            divs = []
            cn = []   # (num_correct, num_samples) per problem, for the unbiased pass@k estimator
            for r, comps in zip(probs, comps_by_prob):
                correct = [c for c in comps if scorer.correct(extract_final_answer(c), r["answer"])]
                n_correct_total += len(correct); n_total += len(comps)
                n_prob_any_correct += int(len(correct) > 0)
                cn.append((len(correct), len(comps)))
                if len(correct) >= 2:
                    # embed the SOLUTION (post-</think>), so strategy diversity is measured on the
                    # method, not on reasoning-token noise (no-op when thinking is off).
                    E = embed([_strip_think(c) for c in correct])
                    S = E @ E.T
                    m = len(correct)
                    divs.append(1.0 - float((S.sum() - np.trace(S)) / (m * (m - 1))))
            pass1 = n_correct_total / n_total
            passk = n_prob_any_correct / len(probs)
            diversity = float(np.mean(divs)) if divs else None
            results[name] = {"pass@1": pass1, f"pass@{args.k}": passk,
                              "diversity_among_correct": diversity,
                              "n_probs_with_ge2_correct": len(divs)}
            if args.k_list:
                from math import comb
                def _passk_unbiased(k):
                    vals = [ (1.0 if (n - c) < k else 1.0 - comb(n - c, k) / comb(n, k))
                             for c, n in cn if n >= k ]
                    return float(np.mean(vals)) if vals else None
                est = {f"pass@{k}": _passk_unbiased(k)
                       for k in (int(x) for x in args.k_list.split(","))}
                results[name]["passk_unbiased"] = est
                print(f"[passk] {name} unbiased estimator (n={args.k}): " +
                      " ".join(f"{kk}={vv:.4f}" for kk, vv in est.items() if vv is not None))
            print(f"[passk] {name}: pass@1={pass1:.3f} pass@{args.k}={passk:.3f} "
                  f"diversity(among correct)={diversity}")
            json.dump(results, open(args.out, "w"), indent=2)
        scorer.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        # vllm_serve_lora spawns a separate EngineCore subprocess (multiprocessing 'spawn')
        # that can outlive proc.terminate(); sweep anything still holding this GPU as a
        # safety net so repeated eval runs don't leak orphaned GPU memory.
        time.sleep(2)
        try:
            pids = subprocess.run(
                ["nvidia-smi", "-i", args.gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.split()
            for pid in pids:
                subprocess.run(["kill", "-9", pid.strip()], check=False)
        except Exception:
            pass

    print(f"[passk] DONE -> {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
