<p align="center">
  <img src="images/embeddings.png" alt="embeddings" width="200">
</p>

# <div align="center">ROAM-GRPO</div>

Code for ROAM-GRPO (Routed Off-mode Approach Mining for GRPO).

## Repo Structure

```
ROAM-GRPO/
├── roam/                          # Core algorithm
│   ├── train.py                   # GRPO + ROAM training script
│   ├── routed_offmode.py           # ROAM loss, routing, off-mode mining, self-distillation
│   ├── distinctness_prompt.py      # LLM judge prompt (distinctness scoring)
│   ├── validity_prompt.py         # LLM judge prompt (reasoning-soundness)
│   ├── run_grpo.sh                # Training launch script
│   └── trl_timeout_patch.py       # Timeout patch for TRL training loop
├── eval/                          # Evaluation
│   ├── passk_eval.py              # pass@k evaluation with vLLM serving
│   ├── vllm_serve_lora.py         # vLLM server with LoRA hot-loading
│   └── recompute_diversity.py     # Embedding/diversity computation offline
├── figures/                       # Paper figure scripts
│   ├── curves_r1.py               # Training-curve figure (pass@1/16, dispersion)
│   └── fig_exemplar.py            # Per-problem exemplars
└── example_completions/           # Completions used for diagrams in paper
└── data/                          # Datasets
    ├── math_train_l5_sub1000.jsonl     # Training set: 1000 MATH Level-5 problems
    ├── math_eval.jsonl                 # Eval set: 400 MATH Level-5 test problems
    └── omnimath_eval.jsonl             # Eval set: 400 Omni-MATH L5/6 problems
```

## Setup

### Requirements
- Python 3.10+
- PyTorch 2.x with CUDA
- transformers >= 4.57
- trl >= 1.4.0
- vllm >= 0.19.1
- sentence-transformers (for Qwen3-Embedding-0.6B)
- For figure scripts: matplotlib, numpy, scikit-learn, scipy

### Models
- **Trained model:** Qwen3.5-2B with LoRA adapters (r=8, alpha=16, all linear projections)
- **LLM judge:** Qwen3.5-9B (for validity and diversity verification)
- **Embedder:** Qwen3-Embedding-0.6B (for solution dispersion metric)

## Training

Launch GRPO + ROAM training:

```bash
cd roam/
bash run_grpo.sh
```

## Evaluation

### pass@k + Diversity

```bash
cd eval/
python passk_eval.py \
    --adapters "grpo_g5000=path/to/adapter,roam_g5000=path/to/adapter" \
    --k 16 --k_list 1,4,8,16 \
    --heldout ../data/math_eval.jsonl \
    --base_model Qwen/Qwen3.5-2B \
    --gpu 0
```


