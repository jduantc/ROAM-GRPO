#!/usr/bin/env python3
"""train.py — GRPO + ROAM training script.

Subcommands:
    python train.py build-data      # build the L5 dataset jsonl
    python train.py train  [opts]   # run GRPO + ROAM

GPU layout (see run_grpo.sh):
    GPU0  policy training (bf16 + LoRA) + frozen strategy encoder
    GPU1  verifier server (TRL vllm-serve, judge model)
    GPU2  policy generation server (TRL vllm-serve, weight-synced)
"""
import argparse
import os
import pathlib
import json
import random
import collections
import re
import threading

# Reduce allocator fragmentation on the training GPU (must be set before torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent

# CONFIG
# Set by train() to the routed_offmode module when --route_criterion is passed. None keeps the
# vanilla path byte-identical (the reward_fn stash and the trainer subclass are both guarded on it).
_ROUTED_OFFMODE = None

CFG = dict(
    policy_model="Qwen/Qwen3.5-2B",
    encoder_model="Qwen/Qwen3-Embedding-0.6B",
    verifier_model="Qwen/Qwen3.5-9B",

    dataset_path=str(REPO / "data/math_train_l5_sub1000.jsonl"),

    # GRPO / generation
    num_generations=8,
    micro_batch=2,
    prompts_per_step=1,
    num_epochs=4,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    lora_r=8, lora_alpha=16,
    beta_kl=0.0,
    max_new_tokens=3072,
    enable_thinking=False,
    temperature=1.0, top_p=1.0,
    top_k=0, min_p=None,

    # reward
    validity_gate_mode="hard_gate",
    format_partial=0.1,
    scale_rewards="batch",

    # verifier
    verifier_backend="trl_server",
    verifier_server_url="http://localhost:8001",
    verifier_max_new_tokens=2560,

    # policy generation server
    vllm_generation_mode="server",
    vllm_server_url="http://localhost:8000",
    vllm_gpu_memory_utilization=0.45,
    vllm_max_model_length=3400,

    # devices
    encoder_device="cuda:0",

    # logging
    log_every_n_steps=5,
    wandb_project="structural-diversity-grpo",
    output_dir=str(HERE / "runs"),
    seed=0,
)

RTOL = 1e-6
CALL_TIMEOUT_SEC = 8

REASONING_INSTRUCTION = (
    "Instruct: Represent the core solution strategy and reasoning approach of this "
    "text (the sequence of mathematical steps and technique used), not its tone, "
    "reasoning narration style, wording, notation, verbosity, or formatting. Two "
    "texts should be close only if they use the same strategy.\nQuery:"
)

VERIFIER_SYSTEM = "You are a careful mathematical problem solver."
VERIFIER_TEMPLATE = """\
You are given a worked solution to one problem, then a NEW problem of the same \
type. Apply the SAME method/strategy shown in the worked solution to solve the \
new problem. Do not invent a different approach - reuse the demonstrated method, \
adapted to the new numbers. Show your work and put the final answer in \\boxed{{}}.

WORKED SOLUTION (method to reuse):
Problem: {q0}
Solution: {d0}

NEW PROBLEM (solve with the same method):
{qi}"""


# ============================================================ ANSWER CHECKER
def _convert_cmd(s, cmd, nargs):
    import sympy as sp  # noqa
    out = s
    while True:
        i = out.find('\\' + cmd)
        if i == -1:
            break
        k = i + len(cmd) + 1
        args = []
        for _ in range(nargs):
            while k < len(out) and out[k] != '{':
                k += 1
            if k >= len(out):
                break
            depth = 0
            start = k
            while k < len(out):
                if out[k] == '{':
                    depth += 1
                elif out[k] == '}':
                    depth -= 1
                    if depth == 0:
                        args.append(out[start + 1:k])
                        k += 1
                        break
                k += 1
        if cmd == 'frac' and len(args) == 2:
            repl = f'(({args[0]})/({args[1]}))'
        elif cmd == 'sqrt' and len(args) == 1:
            repl = f'sqrt({args[0]})'
        else:
            repl = out[i:k]
        out = out[:i] + repl + out[k:]
    return out


def latex_to_expr(s):
    import sympy as sp
    s = s.strip().strip('$').strip()
    for junk in ['\\left', '\\right', '\\!', '\\,', '\\ ']:
        s = s.replace(junk, '')
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    s = s.replace('\\cdot', '*').replace('\\times', '*')
    s = _convert_cmd(s, 'frac', 2)
    s = _convert_cmd(s, 'sqrt', 1)
    s = s.replace('\\pi', 'pi')
    s = re.sub(r'(\d)\s*(?=[a-zA-Z(])', r'\1*', s)
    s = re.sub(r'\)\s*(?=[a-zA-Z0-9(])', r')*', s)
    return sp.sympify(s, locals={'sqrt': sp.sqrt, 'pi': sp.pi})


def to_float(x):
    import sympy as sp
    try:
        return float(x)
    except Exception:
        return float(sp.N(sp.sympify(str(x), locals={'sqrt': sp.sqrt, 'pi': sp.pi})))


def with_timeout(fn, kwargs, seconds=CALL_TIMEOUT_SEC):
    result, error, finished = [None], [None], [False]

    def target():
        try:
            result[0] = fn(**kwargs)
        except Exception as e:
            error[0] = e
        finally:
            finished[0] = True

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if not finished[0]:
        raise TimeoutError(f"timed out after {seconds}s")
    if error[0] is not None:
        raise error[0]
    return result[0]


def _to_value_latex_aware(x):
    try:
        return to_float(x)
    except Exception:
        return to_float(latex_to_expr(str(x)))


def answer_matches(pred, gold, rtol=RTOL):
    """LaTeX-aware numeric match for model/verifier outputs (both may be LaTeX)."""
    if pred is None:
        return False

    def _check():
        p = _to_value_latex_aware(pred)
        g = _to_value_latex_aware(gold)
        return abs(p - g) <= rtol * max(1.0, abs(g))
    try:
        return with_timeout(_check, {})
    except Exception:
        return False


_NUMBER_RE = re.compile(r'-?\d+(?:\.\d+)?(?:/\d+)?')


def _extract_boxed(text):
    marker = r'\boxed'
    idx = text.rfind(marker)
    if idx == -1:
        return None
    k = idx + len(marker)
    while k < len(text) and text[k] != '{':
        k += 1
    if k >= len(text):
        return None
    depth = 0
    start = k
    while k < len(text):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:k]
        k += 1
    return None


def extract_final_answer(text):
    """Last \\boxed{...} if present, else last number-like token; None if nothing."""
    if not text:
        return None
    boxed = _extract_boxed(text)
    if boxed is not None and boxed.strip():
        return boxed.strip()
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else None


# ============================================================ DATASET
def build_dataset(cfg):
    """One row per hard-pass schema: {schema_id, problem, gold}. Requires >=1
    sibling in instances.jsonl (so Tier-2 can fire)."""
    sib_counts = collections.Counter()
    for line in open(cfg["instances_path"]):
        sib_counts[json.loads(line)["problem_id"]] += 1

    rows, n_total = [], 0
    skipped = collections.Counter()
    for line in open(cfg["check_table_path"]):
        r = json.loads(line)
        n_total += 1
        if r.get("status") != "hard_pass":
            skipped["not_hard_pass"] += 1
            continue
        pid = r["problem_id"]
        if sib_counts.get(pid, 0) < 1:
            skipped["no_siblings"] += 1
            continue
        try:
            d = json.loads((pathlib.Path(cfg["schema_dir"]) / r["_file"]).read_text())
        except Exception:
            skipped["unreadable"] += 1
            continue
        problem = d.get("problem")
        if not problem or not str(problem).strip():
            skipped["no_problem"] += 1
            continue
        rows.append({"schema_id": pid, "problem": problem.strip(), "gold": r["gold"]})

    out = pathlib.Path(cfg["dataset_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build-data] {n_total} schemas -> emitted {len(rows)} "
          f"(skipped {dict(skipped)}) -> {out}")
    return rows


class SiblingIndex:
    def __init__(self, instances_path, seed=0):
        self.by_schema = collections.defaultdict(list)
        for line in open(instances_path):
            d = json.loads(line)
            self.by_schema[d["problem_id"]].append(
                {"problem": d["instance_problem"], "answer": str(d["computed_answer"])})
        self.rng = random.Random(seed)

    def draw(self, schema_id):
        sibs = self.by_schema.get(schema_id)
        return self.rng.choice(sibs) if sibs else None


# ============================================================ ENCODER
class StrategyEncoder:
    """Frozen Qwen3-Embedding-0.6B + reasoning instruction. Never trained."""
    def __init__(self, model_id, device="cuda:0", batch_size=16):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.model = SentenceTransformer(model_id, device=device)
        self.model.eval()
        self.batch_size = batch_size

    def embed(self, texts):
        with self.torch.no_grad():
            emb = self.model.encode(
                texts, batch_size=self.batch_size, convert_to_numpy=True,
                normalize_embeddings=True, prompt=REASONING_INSTRUCTION,
                show_progress_bar=False)
        return emb.astype(np.float32)


# ============================================================ VERIFIER (tau)
class Verifier:
    """Frozen verifier: re-applies a demonstrated method to sibling problems."""
    def __init__(self, model_id, backend="trl_server", device="cuda:0",
                 max_new_tokens=512, server_url=None):
        self.model_id = model_id
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.server_url = server_url
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16).to(device)
            self.model.eval()
            self.device = device
        elif backend == "trl_server":
            assert server_url, "trl_server backend needs server_url"
            import requests  # noqa: F401
        else:
            raise ValueError(f"unknown verifier backend {backend}")

    @staticmethod
    def _worked_solution(text, max_chars=6000):
        """Extract the solution body (post-</think> if present) for the verifier."""
        if "</think>" in text:
            sol = text.split("</think>", 1)[1].strip()
            if len(sol) > 20:
                return sol[:max_chars]
        return text[-max_chars:]

    def _prompt(self, q0, d0, qi):
        return VERIFIER_TEMPLATE.format(q0=q0, d0=self._worked_solution(d0), qi=qi)

    def reapply_batch(self, triples):
        """triples: list of (q0, d0, qi) -> list of generated texts."""
        if not triples:
            return []
        prompts = [self._prompt(*t) for t in triples]
        if self.backend == "hf":
            return self._gen_hf(prompts)
        return self._gen_trl_server(prompts)

    def _gen_hf(self, prompts):
        import torch
        chats = [self.tok.apply_chat_template(
                    [{"role": "system", "content": VERIFIER_SYSTEM},
                     {"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = self.tok(chats, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)

    def _gen_trl_server(self, prompts):
        import requests
        messages = [[{"role": "system", "content": VERIFIER_SYSTEM},
                     {"role": "user", "content": p}] for p in prompts]
        resp = requests.post(
            f"{self.server_url}/chat/",
            json={"messages": messages, "n": 1, "temperature": 0.0,
                  "top_p": 1.0, "max_tokens": self.max_new_tokens},
            timeout=600)
        resp.raise_for_status()
        completion_ids = resp.json()["completion_ids"]   # list[list[int]], one per prompt
        return self.tok.batch_decode(completion_ids, skip_special_tokens=True)


# ============================================================ REWARD
class RewardState:
    """Running per-schema rho-hat (reward-hacking canary)."""
    def __init__(self):
        self.rho_pass = collections.Counter()
        self.rho_total = collections.Counter()

    def update(self, schema_id, passed):
        self.rho_total[schema_id] += 1
        if passed:
            self.rho_pass[schema_id] += 1

    def rho_hat(self, schema_id):
        t = self.rho_total[schema_id]
        return (self.rho_pass[schema_id] / t) if t else float("nan")

    def mean_rho_hat(self):
        vals = [self.rho_pass[s] / self.rho_total[s]
                for s in self.rho_total if self.rho_total[s] > 0]
        return float(np.mean(vals)) if vals else float("nan")


def make_reward_fn(encoder, verifier, siblings, state, cfg):
    G = cfg["num_generations"]
    gate_mode = cfg["validity_gate_mode"]
    partial = cfg["format_partial"]
    disable_verifier = cfg.get("disable_verifier", False)

    def reward_fn(prompts, completions, completion_ids=None, **kwargs):
        n = len(completions)
        gold = kwargs["gold"]
        schema_id = kwargs["schema_id"]
        problem = kwargs["problem"]
        log_metric = kwargs.get("log_metric")
        log_extra = kwargs.get("log_extra")

        # ---- Tier 1: parse + answer-match ----
        parse, tier1, base, extracted = [0]*n, [0]*n, [0.0]*n, [None]*n
        for i in range(n):
            ans = extract_final_answer(completions[i])
            extracted[i] = ans
            if ans is not None:
                parse[i] = 1
                if answer_matches(ans, gold[i]):
                    tier1[i] = 1
            base[i] = 1.0 if tier1[i] else (partial if parse[i] else 0.0)

        # ---- Tier 2: schematic re-application (tier1-passers only) ----
        valid, gate_pass = [0]*n, [None]*n
        tau_ans, sib_ans = [None]*n, [None]*n
        if not disable_verifier:
            triples, meta = [], []
            for i in [j for j in range(n) if tier1[j]]:
                sib = siblings.draw(schema_id[i])
                if sib is None:
                    continue
                triples.append((problem[i], completions[i], sib["problem"]))
                meta.append((i, sib))
            gens = verifier.reapply_batch(triples)
            for (i, sib), gen in zip(meta, gens):
                a = extract_final_answer(gen)
                tau_ans[i], sib_ans[i] = a, sib["answer"]
                passed = (a is not None) and answer_matches(a, sib["answer"])
                gate_pass[i] = bool(passed)
                state.update(schema_id[i], passed)
        for i in range(n):
            if not tier1[i]:
                valid[i] = 0
            elif disable_verifier or gate_mode == "canary":
                valid[i] = 1
            else:
                valid[i] = 1 if gate_pass[i] else 0

        # ---- stash per-group Tier-1 flags for the routed off-mode arm (routed_offmode.py) ----
        # Stash rather than re-grade: sympy answer-matching is the slow part of this function, and the
        # routed trainer needs exactly the flags already computed above. No-op unless routing is on.
        if _ROUTED_OFFMODE is not None:
            for g0 in range(0, n, G):
                grp = list(range(g0, min(g0 + G, n)))
                _ROUTED_OFFMODE.stash_group(problem[g0], gold[g0],
                                       [completions[i] for i in grp],
                                       [tier1[i] for i in grp],
                                       schema_id[g0] if schema_id else None)

        # ---- per-group embedding dispersion (monitoring only) ----
        group_pair_cos = []
        for g0 in range(0, n, G):
            grp = list(range(g0, min(g0 + G, n)))
            C = [i for i in grp if valid[i]]
            if len(C) < 2:
                continue
            embs = encoder.embed([completions[i] for i in C])
            S = embs @ embs.T
            mm = len(C)
            group_pair_cos.append((S.sum() - np.trace(S)) / (mm * (mm - 1)))

        R = [base[i] for i in range(n)]

        # ---- logging ----
        if log_metric is not None:
            log_metric("pass_rate", float(np.mean(tier1)))
            log_metric("parse_rate", float(np.mean(parse)))
            log_metric("n_correct_per_group",
                       float(np.mean([sum(tier1[g0:g0+G]) for g0 in range(0, n, G)])))
            drawn = [i for i in range(n) if gate_pass[i] is not None]
            if drawn:
                log_metric("tier2_pass_rate_instant",
                           float(np.mean([gate_pass[i] for i in drawn])))
                log_metric("tier2_gated_out_frac",
                           float(np.mean([0.0 if gate_pass[i] else 1.0 for i in drawn])))
            log_metric("rho_hat_mean_running", state.mean_rho_hat())
            if group_pair_cos:
                log_metric("pairwise_cos_valid_mean", float(np.mean(group_pair_cos)))
        if log_extra is not None:
            log_extra("parse", parse)
            log_extra("tier1_match", tier1)
            log_extra("gate_pass", ["-" if gate_pass[i] is None else str(int(gate_pass[i]))
                                    for i in range(n)])
            log_extra("valid", valid)
            log_extra("base", [round(base[i], 3) for i in range(n)])
            log_extra("R", [round(R[i], 4) for i in range(n)])
            log_extra("extracted_ans", [str(extracted[i]) for i in range(n)])
            log_extra("tau_ans", [str(tau_ans[i]) for i in range(n)])
            log_extra("sibling_ans", [str(sib_ans[i]) for i in range(n)])
            log_extra("rho_hat_schema",
                      [round(state.rho_hat(schema_id[i]), 3) for i in range(n)])
        return R

    return reward_fn


# TRAINER
ENABLE_THINKING = False # recommended off for Qwen3.5-2b, prevents non-ending thinking spirals


def _strip_think(text):
    """Return the solution part (post-</think>), or the full text if thinking is off."""
    if text and "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


def build_prompt_column(tokenizer, problem):
    msgs = [{"role": "user", "content":
             problem + "\n\nSolve the problem step by step, then put the final "
                       "answer in \\boxed{}."}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=ENABLE_THINKING)


def train(args, cfg):
    run_name = args.run_name or f"grpo_{cfg['validity_gate_mode']}"
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    import trl_timeout_patch; trl_timeout_patch.apply()
    from trl import GRPOTrainer, GRPOConfig

    tok = AutoTokenizer.from_pretrained(cfg["policy_model"])

    ds = load_dataset("json", data_files=cfg["dataset_path"], split="train")
    if args.smoke:
        ds = ds.select(range(min(16, len(ds))))
    ds = ds.map(lambda r: {"prompt": build_prompt_column(tok, r["problem"])})

    print(f"[setup] encoder on {cfg['encoder_device']} ...")
    encoder = StrategyEncoder(cfg["encoder_model"], device=cfg["encoder_device"])
    if cfg["disable_verifier"]:
        print("[setup] verifier DISABLED (Tier-2 skipped; valid==tier1)")
        verifier = None
    else:
        print(f"[setup] verifier ({cfg['verifier_backend']}) @ {cfg['verifier_server_url']}")
        verifier = Verifier(cfg["verifier_model"], backend=cfg["verifier_backend"],
                            device=cfg["encoder_device"],
                            max_new_tokens=cfg["verifier_max_new_tokens"],
                            server_url=cfg["verifier_server_url"])
    siblings = SiblingIndex(cfg["instances_path"], seed=cfg["seed"])
    state = RewardState()
    reward_fn = make_reward_fn(encoder, verifier, siblings, state, cfg)
    reward_fn.__name__ = "embed_diversity_reward"

    if cfg["vllm_generation_mode"] == "server":
        from urllib.parse import urlparse as _urlparse
        _sp = _urlparse(cfg["vllm_server_url"]).port or 8000
        vllm_kwargs = dict(vllm_mode="server", vllm_server_base_url=cfg["vllm_server_url"],
                           vllm_group_port=51216 + (_sp - 8000))
    else:
        vllm_kwargs = dict(vllm_mode="colocate",
                           vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
                           vllm_max_model_length=cfg["vllm_max_model_length"],
                           vllm_tensor_parallel_size=1)

    accum_steps = cfg["prompts_per_step"] * cfg["num_generations"] // cfg["micro_batch"]
    grpo_args = GRPOConfig(
        output_dir=str(pathlib.Path(cfg["output_dir"]) / run_name),
        run_name=run_name,
        per_device_train_batch_size=cfg["micro_batch"],
        num_generations=cfg["num_generations"],
        gradient_accumulation_steps=accum_steps,
        steps_per_generation=accum_steps,
        num_train_epochs=cfg["num_epochs"],
        max_steps=args.max_steps,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        beta=cfg["beta_kl"],
        max_completion_length=cfg["max_new_tokens"],
        temperature=cfg["temperature"], top_p=cfg["top_p"],
        top_k=cfg["top_k"], min_p=cfg["min_p"],
        scale_rewards=cfg["scale_rewards"],
        model_init_kwargs={"dtype": "bfloat16"},   # bf16 base (18GB) not fp32 (36GB)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True,
        bf16=True,
        log_completions=True,
        num_completions_to_print=cfg["num_generations"],
        logging_steps=1,
        save_steps=args.save_steps, save_strategy="steps",
        report_to=([] if args.no_wandb else ["wandb"]),
        seed=cfg["seed"],
        **vllm_kwargs,
    )
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                      target_modules="all-linear", task_type="CAUSAL_LM")
    TrainerCls = GRPOTrainer
    if getattr(args, "route_criterion", None):
        import sys as _sys, pathlib as _pl
        _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
        import routed_offmode as _rbc
        globals()["_ROUTED_OFFMODE"] = _rbc
        class TrainerCls(_rbc.RoutedBCMixin, GRPOTrainer):   # mixin FIRST so hooks win
            pass
    trainer = TrainerCls(model=cfg["policy_model"], reward_funcs=reward_fn,
                         args=grpo_args, train_dataset=ds, peft_config=lora)
    if getattr(args, "route_criterion", None):
        trainer.init_routed_offmode({"routed_offmode": {
            "enabled": True, "criterion": args.route_criterion, "lam": args.route_lambda,
            "n_cand": args.route_n_cand, "max_tokens": cfg["max_new_tokens"],
            "server_url": args.vllm_server_url, "tag": args.route_tag,
            "min_correct": args.route_min_correct,
            "judge_url": args.route_judge_url, "judge_thresh": args.route_judge_thresh,
            "judge_max_tokens": args.route_judge_max_tokens,
            "require_validity": args.route_require_validity,
            "validity_max_tokens": args.route_validity_max_tokens,
            "modal_weight": args.route_modal_w, "distinct_weight": args.route_distinct_w,
            "pg_scale": args.route_pg_scale}}, tok)

    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        ap = pathlib.Path(args.init_adapter) / "adapter_model.safetensors"
        if not ap.exists():
            raise SystemExit(f"--init_adapter: no adapter_model.safetensors under {args.init_adapter}")
        sd = load_file(str(ap))
        res = set_peft_model_state_dict(trainer.model, sd)
        n_unexpected = len(getattr(res, "unexpected_keys", []) or [])
        print(f"[run] continuing existing adapter: {args.init_adapter} "
              f"({len(sd)} tensors loaded, {n_unexpected} unexpected)")
        if n_unexpected:
            print(f"[warn] {n_unexpected} unexpected adapter keys -- LoRA config mismatch? "
                  f"first few: {list(getattr(res, 'unexpected_keys', []))[:3]}")
    print(f"[run] {run_name}: {len(ds)} prompts, G={cfg['num_generations']}, "
          f"gate={cfg['validity_gate_mode']}, beta={cfg['beta_kl']}, "
          f"verifier={'off' if cfg['disable_verifier'] else cfg['verifier_backend']}")
    trainer.train()
    trainer.save_model(str(pathlib.Path(cfg["output_dir"]) / run_name / "final"))
    print(f"[done] {run_name}; mean rho_hat = {state.mean_rho_hat()}")


def sanity():
    """Cheap, no-model regression checks (answer matching + novelty edge cases)."""
    ok = True
    cases = [("6", "6", True), ("5", "6", False), ("-1/8", "-\\frac{1}{8}", True),
             ("\\frac{3}{4}", "\\frac{3}{4}", True), ("\\frac{1}{2}", "0.5", True),
             ("2\\sqrt{2}", "2\\sqrt{2}", True), ("3.14", "\\pi", False)]
    for pred, gold, want in cases:
        got = answer_matches(pred, gold)
        flag = "ok" if got == want else "XX"
        if got != want:
            ok = False
        print(f"  [{flag}] answer_matches({pred!r},{gold!r})={got} want={want}")
    assert extract_final_answer("x \\boxed{-\\frac{1}{8}} y") == "-\\frac{1}{8}"
    assert extract_final_answer("the answer is 42.") == "42"
    print(f"  overall: {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-data")
    sub.add_parser("sanity")

    pt = sub.add_parser("train")
    pt.add_argument("--gate_mode", default=CFG["validity_gate_mode"],
                    choices=["hard_gate", "canary"])
    pt.add_argument("--scale_rewards", default=CFG["scale_rewards"],
                    choices=["group", "batch", "none"],
                    help="reward normalization. 'batch' normalizes across the full batch; "
                         "'group' (TRL default) divides by per-group reward std.")
    pt.add_argument("--disable_verifier", action="store_true",
                    help="skip Tier-2 (valid==tier1); training-loop debug only")
    pt.add_argument("--verifier_backend", default=CFG["verifier_backend"],
                    choices=["trl_server", "hf"])
    pt.add_argument("--verifier_server_url", default=CFG["verifier_server_url"])
    pt.add_argument("--verifier_max_new_tokens", type=int,
                    default=CFG["verifier_max_new_tokens"])
    pt.add_argument("--vllm_mode", default=CFG["vllm_generation_mode"],
                    choices=["server", "colocate"])
    pt.add_argument("--vllm_server_url", default=CFG["vllm_server_url"])
    pt.add_argument("--encoder_device", default=CFG["encoder_device"])
    pt.add_argument("--max_steps", type=int, default=-1)
    pt.add_argument("--max_new_tokens", type=int, default=CFG["max_new_tokens"])
    pt.add_argument("--micro_batch", type=int, default=CFG["micro_batch"],
                    help="completions per logp/backward micro-forward. Lower if the "
                         "fp32 logits (micro_batch x seq x vocab~248k) OOM at long "
                         "max_new_tokens. Must divide num_generations.")
    pt.add_argument("--num_generations", type=int, default=CFG["num_generations"],
                    help="G completions sampled per prompt. Must be divisible by micro_batch.")
    pt.add_argument("--prompts_per_step", type=int, default=CFG["prompts_per_step"],
                    help="prompts' worth of completions averaged into one optimizer step via "
                         "gradient accumulation. Raises effective batch size (more stable "
                         "gradients) without raising peak memory (micro_batch is unchanged); "
                         "costs staler vLLM weights between syncs and more wall-clock/step.")
    pt.add_argument("--temperature", type=float, default=CFG["temperature"],
                    help="sampling temperature for policy generation. >1.0 = more "
                         "diverse (pair with --min_p to avoid degeneration).")
    pt.add_argument("--top_p", type=float, default=CFG["top_p"],
                    help="nucleus top_p. Keep 1.0 when raising temperature; use --min_p instead.")
    pt.add_argument("--top_k", type=int, default=CFG["top_k"],
                    help="top_k cutoff (unset = disabled).")
    pt.add_argument("--min_p", type=float, default=CFG["min_p"],
                    help="min_p cutoff: drop tokens below min_p*max_prob. Small values "
                         "(0.02-0.1) let you raise temperature well past 1 without garbage.")
    pt.add_argument("--seed", type=int, default=CFG["seed"],
                    help="training seed (data shuffle, weight init, dropout). vLLM sampling "
                         "stays nondeterministic regardless.")
    pt.add_argument("--save_steps", type=int, default=100,
                    help="checkpoint cadence in optimizer steps (LoRA adapters are small).")
    pt.add_argument("--enable_thinking", action="store_true",
                    help="open a <think> block in every policy prompt. Threads to routed off-mode "
                         "harvest and eval. Pair with --max_new_tokens 8192.")
    pt.add_argument("--num_epochs", type=float, default=CFG["num_epochs"])
    pt.add_argument("--learning_rate", type=float, default=CFG["learning_rate"])
    pt.add_argument("--lr_scheduler", default=CFG["lr_scheduler_type"],
                    help="HF lr_scheduler_type: 'linear' (default, decay to 0) or 'constant' "
                         "(phase-2 continuations that test collapse under sustained-LR fine-tuning).")
    pt.add_argument("--dataset_path", default=CFG["dataset_path"],
                    help="training jsonl (needs problem, gold, schema_id columns)")
    pt.add_argument("--init_adapter", default=None,
                    help="path to an existing LoRA adapter to CONTINUE training instead of "
                         "creating a fresh one (interleaved GRPO+SFT loop).")
    pt.add_argument("--policy_model", default=CFG["policy_model"],
                    help="policy to train. Default Qwen3.5-9B keeps prior runs identical. Point "
                         "at a SMALLER/OLDER model to run the diversified-GRPO arm on it; TRL "
                         "resolves the architecture itself, but the tau verifier and the gen "
                         "server must be pointed at the same model too (run_grpo.sh POLICY_MODEL).")
    pt.add_argument("--run_name", default=None)
    pt.add_argument("--no_wandb", action="store_true")
    pt.add_argument("--smoke", action="store_true")
    # ---- routed off-mode arm (routed_offmode.py). Absent => vanilla.
    pt.add_argument("--route_criterion", default=None,
                    choices=[None, "all_correct", "min_correct"],
                    help="route a GRPO group to the BC track when this holds; None disables")
    pt.add_argument("--route_lambda", type=float, default=1.0)
    pt.add_argument("--route_n_cand", type=int, default=8)
    pt.add_argument("--route_tag", default="run")
    pt.add_argument("--route_min_correct", type=int, default=2,
                    help="min_correct criterion: route when >=k of G are correct")
    pt.add_argument("--route_judge_url", default="http://localhost:8265",
                    help="9B distinctness judge, called IN-LOOP on routed steps.")
    pt.add_argument("--route_judge_thresh", type=float, default=95.0,
                    help="SCORE < thresh => DIFFERENT method (95 = balanced optimum on labels).")
    pt.add_argument("--route_require_validity", type=int, default=1,
                    help="require distinct candidates to also pass a reasoning soundness audit "
                         "(VALIDITY: sound). Runs as a second judge call on candidates passing "
                         "the distinctness gate.")
    pt.add_argument("--route_validity_max_tokens", type=int, default=700,
                    help="Budget for the validity audit prompt (parse rate 30/30 at 700).")
    pt.add_argument("--route_judge_max_tokens", type=int, default=512,
                    help="judge generation budget. Do not go below 512 (SCORE gets truncated).")
    pt.add_argument("--route_modal_w", type=float, default=1.0,
                    help="Weight on the correctness-only (modal) BC target.")
    pt.add_argument("--route_distinct_w", type=float, default=1.0,
                    help="Weight on the judged-distinct BC target.")
    pt.add_argument("--route_pg_scale", type=float, default=0.0,
                    help="scale applied to a routed group's PG advantages (0 = BC only)")
    args = ap.parse_args()

    cfg = dict(CFG)
    if args.cmd == "build-data":
        build_dataset(cfg)
        return
    if args.cmd == "sanity":
        sanity()
        return
    cfg["policy_model"] = args.policy_model
    cfg["dataset_path"] = args.dataset_path
    cfg["validity_gate_mode"] = args.gate_mode
    cfg["scale_rewards"] = args.scale_rewards
    cfg["disable_verifier"] = args.disable_verifier
    cfg["verifier_backend"] = args.verifier_backend
    cfg["verifier_server_url"] = args.verifier_server_url
    cfg["vllm_generation_mode"] = args.vllm_mode
    cfg["vllm_server_url"] = args.vllm_server_url
    cfg["encoder_device"] = args.encoder_device
    cfg["num_epochs"] = args.num_epochs
    cfg["learning_rate"] = args.learning_rate
    cfg["lr_scheduler_type"] = args.lr_scheduler
    cfg["seed"] = args.seed
    cfg["enable_thinking"] = args.enable_thinking
    # Set the single source of truth BEFORE train() builds any prompt.
    global ENABLE_THINKING
    ENABLE_THINKING = args.enable_thinking
    cfg["max_new_tokens"] = args.max_new_tokens
    cfg["temperature"] = args.temperature
    cfg["top_p"] = args.top_p
    cfg["top_k"] = args.top_k
    cfg["min_p"] = args.min_p
    cfg["micro_batch"] = args.micro_batch
    cfg["num_generations"] = args.num_generations
    cfg["prompts_per_step"] = args.prompts_per_step
    cfg["verifier_max_new_tokens"] = args.verifier_max_new_tokens
    assert cfg["num_generations"] % cfg["micro_batch"] == 0, "micro_batch must divide num_generations"
    if args.smoke:
        cfg["max_new_tokens"] = 512
        cfg["verifier_max_new_tokens"] = 256
    train(args, cfg)


if __name__ == "__main__":
    main()
