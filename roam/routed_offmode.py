"""Routed Off-mode. Per GRPO group: if the group is all-correct, route it to the off-mode track. Generate student completions
  from teacher prompt, then distills into the target prompt (plain prompt)
"""
import json, os, pathlib, re, threading
import numpy as np
import requests
import torch

HERE = pathlib.Path(__file__).resolve().parent

# reward_fn appends one record per GRPO group here; the trainer override drains it. Stashing the
# tier1 flags reward_fn already computed avoids re-running sympy grading, which is slow.
MARKER_RE = re.compile(r"\b(wait|hmm|let me re-?(check|evaluate|examine|derive|calculate)|"
                       r"i made (a|an) (mistake|error)|sign error|that'?s (wrong|not right)|"
                       r"let'?s re-?(do|try|start)|actually,|no\.|scratch that|"
                       r"i might have|let me reconsider)\b", re.I)

STASH: list = []


def stash_group(problem, gold, completions, tier1, schema_id=None):
    STASH.append(dict(problem=problem, gold=gold, completions=list(completions),
                      tier1=list(tier1), schema_id=schema_id))


class RoutedBCMixin:
    """Mix in front of GRPOTrainer. Adds a routed masked-CE term; inert when route_frac == 0."""

    def init_routed_offmode(self, cfg, tok):
        r = cfg.get("routed_offmode", {})
        self.rbc_enabled = bool(r.get("enabled", False))
        self.rbc_criterion = r.get("criterion", "all_correct")
        self.rbc_min_correct = int(r.get("min_correct", 2))
        # PG scale applied to a routed group's advantages. 
        # This is 0 by default (i.e. each track has its own loss), but can be set to
        # nonzero values for experimentation
        self.rbc_pg_scale = float(r.get("pg_scale", 0.0))
        self.rbc_lambda = float(r.get("lam", 1.0))
        self.rbc_n_cand = int(r.get("n_cand", 8))
        self.rbc_max_tokens = int(r.get("max_tokens", 3072))
        self.rbc_server = r.get("server_url", "http://localhost:8000")
        self.rbc_modal_w = float(r.get("modal_weight", 1.0))
        self.rbc_distinct_w = float(r.get("distinct_weight", 1.0))
        self.rbc_min_chars = int(r.get("min_chars", 400))
        # markers/1k chars above which a target is rejected
        self.rbc_max_markers = float(r.get("max_markers", 0.45))
        self.rbc_judge_url = r.get("judge_url", "http://localhost:8265")
        self.rbc_judge_model = r.get("judge_model", "Qwen/Qwen3.5-9B")
        self.rbc_judge_thresh = float(r.get("judge_thresh", 95.0))
        self.rbc_judge_max_chars = int(r.get("judge_max_chars", 2600))
        self.rbc_judge_max_tokens = int(r.get("judge_max_tokens", 2000))
        self.rbc_judge_timeout = int(r.get("judge_timeout", 240))
        # Reasoning-soundness measurement (validity_prompt.py).
        self.rbc_require_validity = int(r.get("require_validity", 1))
        self.rbc_validity_max_tokens = int(r.get("validity_max_tokens", 700))
        self._rbc_jtok = None
        self.rbc_tok = tok
        d = HERE / "routedbc_logs"
        d.mkdir(parents=True, exist_ok=True)
        self.rbc_queue = d / f"queue_{r.get('tag','run')}.jsonl"
        self.rbc_stats = dict(steps=0, routed=0, cand=0, tier1_pass=0, gated=0,
                              bc_modal=0, bc_distinct=0, degen=0,
                              val_ok=0, val_bad=0, val_unparsed=0)
        self._rbc_batch = []
        self._rbc_done_this_step = False
        # per-problem history of target texts already trained on
        self.rbc_hist_path = self.rbc_queue.parent / f"history_{r.get('tag','run')}.json"
        # bug handling for cross-run interference. Paths are keyed by --route_tag, so a relaunch under the
        # same tag silently inherited the previous run's files. 
        # The trainer does not support resume (there is no --resume_from_checkpoint), so a fresh process is always a fresh run: archive and start
        # empty, do not append to a file this process did not create.
        import time as _t
        for _p in (self.rbc_queue, self.rbc_hist_path):
            if _p.exists():
                _dst = _p.with_name(f"_prev_{int(_t.time())}_{_p.name}")
                _p.rename(_dst)
                print(f"[rbc] archived pre-existing {_p.name} -> {_dst.name} (no cross-run state)",
                      flush=True)
        self._rbc_hist = {}

    def _rbc_phash(self, problem):
        import hashlib
        return hashlib.sha1(problem[:200].encode()).hexdigest()[:12]

    def _rbc_used(self, problem):
        return set(self._rbc_hist.get(self._rbc_phash(problem), []))

    def _rbc_mark_used(self, problem, text):
        import hashlib
        k = self._rbc_phash(problem)
        h = hashlib.sha1(text[:400].encode()).hexdigest()[:16]
        self._rbc_hist.setdefault(k, [])
        if h not in self._rbc_hist[k]:
            self._rbc_hist[k].append(h)
        try:
            json.dump(self._rbc_hist, open(self.rbc_hist_path, "w"))
        except Exception:
            pass

    @staticmethod
    def _thash(text):
        import hashlib
        return hashlib.sha1(text[:400].encode()).hexdigest()[:16]

    @staticmethod
    def _key(problem, cand):
        import hashlib
        return hashlib.sha1((problem[:200] + "||" + cand[:400]).encode()).hexdigest()[:16]

    # ---------- gates (cheap, lexical; the judge handles distinctness) ----------
    def _rbc_clean(self, text):
        """Scaffold-aware trim: keep the first block, drop a benign section LABEL, reject a genuine
        BACK-REFERENCE, and require \\boxed to SURVIVE the trim. Mirrors scaffold_clean.py, whose
        absence caused the SS0-A degeneration (6/42 trimmed bodies retained \\boxed)."""
        g = self.rbc_stats.setdefault("gate", {})
        def _rej(k):
            g[k] = g.get(k, 0) + 1
            return None
        # Thinking mode: gate on the solution (everything after </think>) -- the marker/boxed/
        # scaffold gates were calibrated on solution text, and self-doubt language or metacommentary are
        # more common in the thinking trace. But the distilled solution keeps the full trace (think + </think> + gated solution), reassembled at
        # the end, so the policy learns to reason to the diverse method, matching what eval samples.
        import train as _M
        _think = ""
        if "</think>" in text:
            _think = text.split("</think>", 1)[0] + "</think>\n\n"
        elif _M.ENABLE_THINKING:
            return _rej("no_close_think")
        t = _M._strip_think(text).strip()
        t = re.sub(r"^\s*(?:\*\*)?Method\s*\d+[^\n]*\n", "", t)      # benign label
        t = re.split(r"\n\s*(?:\*\*)?Method\s*\d+\b", t)[0]           # keep first block only
        if re.search(r"\b(as in Method|as instructed|the other (approach|solution)|"
                     r"as (before|above|earlier)|from earlier|previous context)\b", t, re.I):
            return _rej("backref")
        # Removes completions that contain unnatural language due to the prompt (e.g. metacommentary)
        # note that the student is explicitly prompted to not include such language and provide a 
        # standalone solution
        if re.search(r"\bMethod\s*\d+\b|genuinely different", t, re.I):
            return _rej("scaffold_echo")
        if not re.search(r"\\boxed\{\s*\S", t):
            return _rej("no_boxed")
        # Solution should not include the prompt
        if re.search(r"Put the final answer in \\boxed|Provide a solution\.|"
                     r"Solve the problem step by step, then put the final answer", t, re.I):
            return _rej("prompt_echo")
        # truncate before gating
        i = t.find("\\boxed")
        if i >= 0:
            j, depth = t.find("{", i), 0
            if j >= 0:
                for k in range(j, min(len(t), j + 400)):
                    if t[k] == "{": depth += 1
                    elif t[k] == "}":
                        depth -= 1
                        if depth == 0:
                            t = t[:k + 1] + ("$" if t[i-1:i] == "$" else "")
                            break
        if len(t) < self.rbc_min_chars:
            return _rej("too_short")
        if len(re.findall(r"[=+\-*/^]", t)) < 5:
            return _rej("too_little_math")
        if len(MARKER_RE.findall(t)) / max(len(t) / 1000.0, 1e-9) > self.rbc_max_markers:
            return _rej("markers")
        return _think + t          # _think is "" when thinking is off

    # in-loop distinctness judge 
    def _rbc_judge(self, problem, modal, cands, schema_id):
        """Judge cands against modal and return [(cand, score)] for the distinct ones.
        """
        if not cands:
            return []
        try:
            from distinctness_prompt import DISTINCTNESS
            if getattr(self, "_rbc_jtok", None) is None:
                from transformers import AutoTokenizer
                self._rbc_jtok = AutoTokenizer.from_pretrained(self.rbc_judge_model)
            from train import _strip_think
            _modal_sol = _strip_think(modal)
            prompts = [self._rbc_jtok.apply_chat_template(
                [{"role": "user", "content": DISTINCTNESS
                  .replace("{problem}", problem)
                  .replace("{a}", _modal_sol[:self.rbc_judge_max_chars])
                  .replace("{b}", _strip_think(c)[:self.rbc_judge_max_chars])}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False) for c in cands]
            r = requests.post(f"{self.rbc_judge_url}/generate/",
                              json={"prompts": prompts, "n": 1, "temperature": 0.0,
                                    "top_p": 1.0, "max_tokens": self.rbc_judge_max_tokens},
                              timeout=self.rbc_judge_timeout)
            r.raise_for_status()
            got = r.json()["completion_ids"]
        except Exception as e:
            # In case of exception, doesn't block training on the judge. (otherwise judge may hang and be costly)
            # The canary reports if this becomes chronic.
            self.rbc_stats["judge_fail"] = self.rbc_stats.get("judge_fail", 0) + 1
            print(f"[rbc] judge failed ({type(e).__name__}); no distinct target this group",
                  flush=True)
            return []
        # distinctness based on judge score
        stage1 = []
        for c, ids_ in zip(cands, got):
            txt = self._rbc_jtok.decode(ids_, skip_special_tokens=True)
            ms = list(re.finditer(r"SCORE:\s*(\d{1,3})", txt))
            if not ms:
                self.rbc_stats["judge_unparsed"] = self.rbc_stats.get("judge_unparsed", 0) + 1
                continue
            sc = float(ms[-1].group(1))
            self.rbc_stats["judged"] = self.rbc_stats.get("judged", 0) + 1
            if sc < self.rbc_judge_thresh:      # 95 = balanced optimum on the labelled pairs
                stage1.append((c, sc))
        if not stage1 or not self.rbc_require_validity:
            return stage1

        # logical soundness verifier
        try:
            from validity_prompt import VALIDITY
            from train import _strip_think
            vprompts = [self._rbc_jtok.apply_chat_template(
                [{"role": "user", "content": VALIDITY
                  .replace("{problem}", problem)
                  .replace("{b}", _strip_think(c)[:self.rbc_judge_max_chars])}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False) for c, _ in stage1]
            r2 = requests.post(f"{self.rbc_judge_url}/generate/",
                               json={"prompts": vprompts, "n": 1, "temperature": 0.0,
                                     "top_p": 1.0, "max_tokens": self.rbc_validity_max_tokens},
                               timeout=self.rbc_judge_timeout)
            r2.raise_for_status()
            got2 = r2.json()["completion_ids"]
        except Exception as e:
            self.rbc_stats["judge_fail"] = self.rbc_stats.get("judge_fail", 0) + 1
            print(f"[rbc] validity stage failed ({type(e).__name__}); keeping score-only verdicts",
                  flush=True)
            return stage1                        # does not block training on the judge
        out = []
        for (c, sc), ids2 in zip(stage1, got2):
            t2 = self._rbc_jtok.decode(ids2, skip_special_tokens=True)
            mv = re.findall(r"VALIDITY:\s*(sound|unsound)", t2, re.I)
            if not mv:
                self.rbc_stats["val_unparsed"] += 1
                out.append((c, sc)); continue
            if mv[-1].lower() == "sound":
                self.rbc_stats["val_ok"] += 1
                out.append((c, sc))
            else:
                self.rbc_stats["val_bad"] += 1
        return out

    # harvest an off-mode solution on a routed problem
    def _rbc_harvest(self, rec, modal):
        from train import (build_scaffold_prompt_column, extract_final_answer,
                                               answer_matches)
        seed = re.sub(r"\s+", " ", modal)[:300]
        prompt = build_scaffold_prompt_column(self.rbc_tok, rec["problem"], seed)
        try:
            r = requests.post(f"{self.rbc_server}/generate/",
                              json={"prompts": [prompt], "n": self.rbc_n_cand,
                                    "temperature": 1.0, "top_p": 1.0,
                                    "max_tokens": self.rbc_max_tokens}, timeout=600)
            r.raise_for_status()
            ids = r.json()["completion_ids"]
        except Exception as e:
            print(f"[rbc] harvest failed ({type(e).__name__}); skipping group", flush=True)
            return []
        texts = [self.rbc_tok.decode(x, skip_special_tokens=True) for x in ids]
        self.rbc_stats["cand"] += len(texts)
        out = []
        for t in texts:
            if len(t) < 200:
                self.rbc_stats["degen"] += 1
            # cleans solution before using verification. Verifying the raw completion 
            # can sometimes validate the wrong answer based on the edge case where a completion contains 
            # multiple boxed vals. 
            # answer_matches (not string equality): gold is sympy-ish, the model emits LaTeX, so
            # \frac{1}{2} vs 1/2 vs 0.5 must compare equal. Same grader reward_fn uses.
            c = self._rbc_clean(t)
            if c is None:
                continue
            ans = extract_final_answer(c)
            if ans is None or not answer_matches(ans, rec["gold"]):
                self.rbc_stats["clean_wrong"] = self.rbc_stats.get("clean_wrong", 0) + 1
                continue
            self.rbc_stats["tier1_pass"] += 1
            self.rbc_stats["gated"] += 1
            out.append(c)
        return out

    # TRL hooks
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if not self.rbc_enabled:
            STASH.clear()
            return out
        self.rbc_stats["steps"] += 1
        self._rbc_batch = []
        self._rbc_done_this_step = False
        recs, STASH[:] = list(STASH), []
        routed_idx = []
        for gi, rec in enumerate(recs):
            n, nc = len(rec["tier1"]), sum(rec["tier1"])
            if self.rbc_criterion == "all_correct":
                routed = (nc == n and n > 0)
            else:                                  # min_correct: >=k correct
                routed = (nc >= self.rbc_min_correct)
            if not routed:
                continue
            self.rbc_stats["routed"] += 1
            routed_idx.append(gi)
            corr = [c for c, k in zip(rec["completions"], rec["tier1"]) if k]
            # modal seed set to median length completion
            used = self._rbc_used(rec["problem"])
            ranked = sorted(corr, key=len)
            modal = ranked[len(ranked) // 2] 
            # The modal target is a fresh correct completion not already trained on
            fresh = [c for c in ranked if self._thash(c) not in used]
            mtarget = fresh[0] if fresh else modal      # shortest unused correct completion
            mclean = None
            # shortest completions in this case: median length outputs can contain more middling solutions, and the
            # long ones spiral 
            for cand_t in (fresh + [c for c in ranked if c not in fresh]):
                mclean = self._rbc_clean(cand_t)
                if mclean is not None:
                    break
            if mclean is None:
                self.rbc_stats["no_modal"] = self.rbc_stats.get("no_modal", 0) + 1
                continue                      # no gate-passing target in this group: skip it
            self._rbc_batch.append((rec["problem"], mclean, self.rbc_modal_w))
            self._rbc_mark_used(rec["problem"], mclean)
            self.rbc_stats["bc_modal"] += 1
            cands = self._rbc_harvest(rec, modal)
            # judge against the modal
            verdicts = self._rbc_judge(rec["problem"], mclean, cands, rec.get("schema_id"))
            with open(self.rbc_queue, "a") as f:                # append-only audit trail, not a source
                for c, sc in verdicts:
                    f.write(json.dumps(dict(key=self._key(rec["problem"], c), step=self.rbc_stats["steps"],
                                            problem=rec["problem"], modal=mclean, cand=c, score=sc,
                                            schema_id=rec.get("schema_id"))) + "\n")
            # Only JUDGED-DISTINCT candidates are trained on, and only ONE per routed group per step
            # (cap=2 overall: 1 modal + 1 distinct). This fixes the modal:distinct ratio at 1:1 rather
            # than letting it drift with harvest yield, bounds the extra forward/backward passes, and
            # keeps the dose inside Thread A's validated <=K=4 (1 per encounter x 3 epochs).
            # Prefer a candidate NOT already trained on, so the 3 encounters teach 3 different
            # methods; among those, take the most distinct (lowest same-method score).
            # Candidates are this step's own harvest, judged this step. `used` still rotates the
            # target across a problem's 3 encounters so they teach 3 methods, not one method 3x,
            # but it only ever filters within a freshly generated set.
            unused = [(c, sc) for c, sc in verdicts if self._thash(c) not in used]
            pool = unused or verdicts
            if pool:
                c, sc = min(pool, key=lambda cv: cv[1])          # most distinct = lowest same-method score
                self._rbc_batch.append((rec["problem"], c, self.rbc_distinct_w))
                self._rbc_mark_used(rec["problem"], c)
                self.rbc_stats["bc_distinct"] += 1
            # Per-route line: verify the arm is firing. 
            print(f"[rbc] ROUTED n={n} cand={len(cands)} distinct={len(verdicts)} "
                  f"bc={len(self._rbc_batch)}", flush=True)
        # Scale the PG advantages of routed groups. Groups are contiguous blocks of G in the batch
        # and stash is appended in the same prompt order, so group gi occupies [gi*G, (gi+1)*G).
        if routed_idx and self.rbc_pg_scale != 1.0 and "advantages" in out:
            adv = out["advantages"]
            G = self.args.num_generations
            for gi in routed_idx:
                lo, hi = gi * G, min((gi + 1) * G, adv.shape[0])
                if lo < adv.shape[0]:
                    adv[lo:hi] = adv[lo:hi] * self.rbc_pg_scale
        self.rbc_log()
        return out

    def _rbc_loss(self, model):
        """Masked CE on (plain prompt -> target). Prompt tokens are masked out with -100."""
        if not self._rbc_batch:
            return None
        dev = next(model.parameters()).device
        tot, wsum = None, 0.0
        from train import build_prompt_column
        for problem, target, w in self._rbc_batch:
            p = build_prompt_column(self.rbc_tok, problem)
            pid = self.rbc_tok(p, add_special_tokens=False).input_ids
            tid = self.rbc_tok(target, add_special_tokens=False).input_ids
            # Append stop token, otherwise distillation can teach "produce this
            # solution and keep going"
            _end = self.rbc_tok.convert_tokens_to_ids("<|im_end|>")
            if _end is None or (isinstance(_end, int) and _end < 0):
                _end = self.rbc_tok.eos_token_id
            if _end is not None and (not tid or tid[-1] != _end):
                tid = tid + [_end]
            if len(pid) + len(tid) > self.rbc_max_tokens + 1024:
                tid = tid[: self.rbc_max_tokens + 1024 - len(pid)]
            ids = torch.tensor([pid + tid], device=dev)
            lab = torch.tensor([[-100] * len(pid) + tid], device=dev)
            loss = model(input_ids=ids, labels=lab).loss
            tot = loss * w if tot is None else tot + loss * w
            wsum += w
        return tot / max(wsum, 1e-6)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        if not self.rbc_enabled or not self._rbc_batch or self._rbc_done_this_step:
            return loss
        bc = self._rbc_loss(model)
        if bc is None:
            return loss
        # Applied on one micro-batch only, scaled by grad_accum so the per-optimizer-step
        # contribution is exactly lambda * BC after HF's loss/accum division. Equivalent in
        # expectation to adding it every micro-batch, cheaper
        self._rbc_done_this_step = True
        self._rbc_last_bc = float(bc.detach())
        return loss + self.rbc_lambda * bc

    def rbc_log(self):
        s = self.rbc_stats
        # Canary: watches in case the judge hangs / errors out
        if s["routed"] >= 30 and s["bc_distinct"] == 0 and s["steps"] % 5 == 0:
            print(f"[rbc] CANARY: {s['routed']} -- the distinct path is inert"
                  f"judged={s.get('judged',0)} unparsed={s.get('judge_unparsed',0)} "
                  f"judge_fail={s.get('judge_fail',0)} "
                  f"val_ok={s.get('val_ok',0)} val_bad={s.get('val_bad',0)} "
                  f"val_unparsed={s.get('val_unparsed',0)} gates={s.get('gate',{})}", flush=True)
        if s["steps"] and s["steps"] % 25 == 0:
            print(f"[rbc] steps={s['steps']} routed={s['routed']} cand={s['cand']} "
                  f"tier1={s['tier1_pass']} gated={s['gated']} bc_modal={s['bc_modal']} "
                  f"bc_distinct={s['bc_distinct']} degen={s['degen']} "
                  f"judged={s.get('judged',0)} unparsed={s.get('judge_unparsed',0)} "
                  f"judge_fail={s.get('judge_fail',0)} gates={s.get('gate',{})}", flush=True)
