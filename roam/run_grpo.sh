#!/usr/bin/env bash
# Orchestrate a full server-mode GRPO + ROAM `comparison run.
#
#   GPU GEN_GPU    : policy generation server   (TRL vllm-serve, weight-synced)
#   GPU TAU_GPU    : verifier tau server         (TRL vllm-serve, judge model)
#   GPU TRAIN_GPU  : policy training (bf16 + LoRA) + frozen strategy encoder
#
# Starts both servers, waits for /health/, runs the trainer (which sees ONLY the
# training GPU -- required for the single-device weight-sync NCCL group), then
# tears both servers down.
#
# Usage:
#   ./run_grpo.sh --diversity_weight 0.3 --run_name w0.3_treatment [more train args]
#   ./run_grpo.sh --diversity_weight 0.0 --run_name w0.0_control
#   SMOKE: ./run_grpo.sh --smoke --max_steps 6 --diversity_weight 0.3 --run_name smoke --no_wandb
#
# NOTE: no `set -u` -- the conda cuda-nvcc activation hook references an unset var.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Activate your conda environment (adjust path as needed)
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate grpo_train

POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen3.5-2B}"   # env-overridable
TAU_MODEL="${TAU_MODEL:-Qwen/Qwen3.5-9B}"         # judge / verifier model
# Gen server defaults to single-GPU (TP=1). For multi-GPU generation, set e.g.
# GEN_GPU=2,3 GEN_TP=2 ./run_grpo.sh ...
GEN_GPU="${GEN_GPU:-2}"
GEN_TP="${GEN_TP:-1}"
# GEN_DP: data-parallel replicas of the gen server. For a small policy (2B) that fits on one card,
# DP scales generation THROUGHPUT ~linearly with no tensor-parallel comm overhead, and this run is
# generation-bound -- so DP is the preferred multi-GPU gen lever over TP here. The gen card set must
# expose GEN_TP*GEN_DP GPUs (e.g. GEN_GPU=1,2,3 GEN_TP=1 GEN_DP=3, or GEN_GPU=1,2,3,4 TP=2 DP=2).
GEN_DP="${GEN_DP:-1}"
TAU_GPU="${TAU_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"
# GEN_EAGER=False enables vLLM CUDA-graph capture on the generation server. Decode is thousands of
# tiny kernel launches per sequence and on a 2B that overhead dominates, so graphs typically buy
# 1.3-2x. SPEED ONLY -- not sampling, not the loss, not results. Default True keeps the defensive
# setting this repo has used throughout (SS9: vLLM is flaky in this env).
GEN_EAGER="${GEN_EAGER:-True}"
GEN_PORT="${GEN_PORT:-8000}"
TAU_PORT="${TAU_PORT:-8001}"

# A comma-separated CUDA_VISIBLE_DEVICES list must agree with TP. Without this
# guard, `GEN_GPU=2,3` + the default TP=1 starts a server that silently uses one
# card, while the cleanup path treats the list as one invalid GPU id.
GEN_GPU_LIST="${GEN_GPU//,/ }"
GEN_GPU_COUNT=$(wc -w <<< "$GEN_GPU_LIST")
GEN_WORLD=$(( GEN_TP * GEN_DP ))
if [ "$GEN_GPU_COUNT" -ne "$GEN_WORLD" ]; then
    echo "[run_grpo] ERROR: GEN_GPU=$GEN_GPU exposes $GEN_GPU_COUNT GPU(s), but GEN_TP*GEN_DP=$GEN_WORLD" >&2
    echo "[run_grpo] Set matching values, e.g. GEN_GPU=2,3 GEN_TP=2, or GEN_GPU=1,2,3 GEN_DP=3" >&2
    exit 2
fi

SMOKE=0
for a in "$@"; do [ "$a" = "--smoke" ] && SMOKE=1; done
# gen server context must hold prompt (~500) + max_new_tokens. Override GEN_MAXLEN
# via env to match --max_new_tokens (keep ~600 tok headroom over the completion cap).
if [ "$SMOKE" = "1" ]; then GEN_MAXLEN="${GEN_MAXLEN:-1024}"; else GEN_MAXLEN="${GEN_MAXLEN:-8192}"; fi

LOGDIR="$HERE/runs/_logs"; mkdir -p "$LOGDIR"
# always pass a distinct GEN_PORT per concurrent arm.
GEN_LOG="$LOGDIR/gen_server_${GEN_PORT}.log"; TAU_LOG="$LOGDIR/tau_server_${TAU_PORT}.log"

start_server() {   # name gpu model port maxlen gpu_util tp logfile [dp]  -> echoes ONLY the pid
    local name=$1 gpu=$2 model=$3 port=$4 maxlen=$5 util=$6 tp=$7 log=$8 dp=${9:-1}
    echo "[run_grpo] starting $name server on GPU $gpu (port $port, max_len $maxlen, tp=$tp dp=$dp)" >&2
    : > "$log"
    #  --host 127.0.0.1. vllm_serve DEFAULTS TO 0.0.0.0, i.e. the inference endpoint is exposed to
    # the public internet. Observed 2026-08-01 in gen_server_8020.log: unsolicited scans from
    # 91.231.89.118 and 91.196.152.215 ("GET /", "GET /favicon.ico") plus "Invalid HTTP request
    # received". That is (a) an open, unauthenticated LLM endpoint on this box and (b) connection
    # churn on the very server the trainer holds a keep-alive pool to. Everything that talks to
    # these servers uses http://localhost, so binding to loopback costs nothing.
    # A server is a small process tree (HTTP parent + EngineCore + TP workers).
    # Start a session so cleanup can terminate the whole tree, not just the
    # parent whose PID was returned by the shell.
    CUDA_VISIBLE_DEVICES="$gpu" setsid python -m trl.scripts.vllm_serve \
        --model "$model" --host 127.0.0.1 --port "$port" --tensor_parallel_size "$tp" \
        --data_parallel_size "$dp" \
        --gpu_memory_utilization "$util" --max_model_len "$maxlen" --enforce_eager "$GEN_EAGER" \
        > "$log" 2>&1 &
    echo $!          # stdout: pid only (captured by the caller)
}

wait_health() {    # port pid name
    local port=$1 pid=$2 name=$3
    echo "[run_grpo] waiting for $name /health/ on $port (up to ~40 min) ..."
    for i in $(seq 1 480); do
        if env -u LD_LIBRARY_PATH curl -sf -m 5 "http://localhost:$port/health/" -o /dev/null 2>/dev/null; then
            echo "[run_grpo] $name healthy (~$((i*5))s)"; return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[run_grpo] $name DIED during startup; tail:"; tail -30 "$4"; return 1
        fi
        sleep 5
    done
    echo "[run_grpo] $name never became healthy"; return 1
}

# The tau (Tier-2 verifier) server used to start unconditionally, so a --disable_verifier run held a
# whole GPU for 20 h doing nothing and paid its startup wait. Skip it when the trainer will not call
# it. Behaviour with the flag ABSENT is byte-identical to before.
NEED_TAU=1
for a in "$@"; do [ "$a" = "--disable_verifier" ] && NEED_TAU=0; done

GEN_PID=$(start_server gen "$GEN_GPU" "$POLICY_MODEL" "$GEN_PORT" "$GEN_MAXLEN" 0.85 "$GEN_TP" "$GEN_LOG" "$GEN_DP")
if [ "$NEED_TAU" = "1" ]; then
    TAU_PID=$(start_server tau "$TAU_GPU" "$TAU_MODEL" "$TAU_PORT" 8192 0.60 1 "$TAU_LOG")
else
    TAU_PID=""
    # cleanup() sweeps every compute PID on GEN_GPU/TAU_GPU/TRAIN_GPU. With no tau server
    # of our own, TAU_GPU must not name a card another job is using, or teardown kills that job.
    # Collapsing it onto GEN_GPU makes the sweep touch only cards this run actually owns.
    echo "[run_grpo] --disable_verifier: skipping tau server (TAU_GPU -> $GEN_GPU for cleanup scope)"
    TAU_GPU="$GEN_GPU"
fi

cleanup() {
    echo "[run_grpo] tearing down servers (gen $GEN_PID, tau $TAU_PID) ..."
    for pid in "$GEN_PID" "$TAU_PID"; do
        [ -n "$pid" ] || continue
        # setsid makes the returned PID the process-group leader. The fallback
        # keeps cleanup safe if a server was started by an older launcher.
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 3
    # Reap any lingering EngineCore children holding GPU memory -- but ONLY on the GPUs
    # this run was given. The unscoped `--query-compute-apps=pid` form swept the WHOLE box
    # and would kill unrelated jobs (probes, evals, another run's servers) on every exit.
    for g in $GEN_GPU_LIST ${TAU_GPU//,/ } ${TRAIN_GPU//,/ }; do
        for pid in $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
            kill -9 "$pid" 2>/dev/null || true
        done
    done
}
trap cleanup EXIT

wait_health "$GEN_PORT" "$GEN_PID" gen "$GEN_LOG"
if [ "$NEED_TAU" = "1" ]; then wait_health "$TAU_PORT" "$TAU_PID" tau "$TAU_LOG"; fi

# Unique torch.distributed rendezvous port per run -- concurrent single-GPU trainers otherwise
# collide on the default master port and one dies with DistNetworkError (300s connect timeout).
# Derived from the (first) train GPU + gen port so each concurrent run is distinct.
_FIRST_TRAIN_GPU="${TRAIN_GPU%%,*}"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="$(( 29400 + _FIRST_TRAIN_GPU * 10 + (GEN_PORT % 10) ))"
echo "[run_grpo] launching trainer on GPU $TRAIN_GPU (MASTER_PORT=$MASTER_PORT) ..."
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python "$HERE/train.py" train \
    --policy_model "$POLICY_MODEL" \
    --vllm_mode server --vllm_server_url "http://localhost:$GEN_PORT" \
    --verifier_backend trl_server --verifier_server_url "http://localhost:$TAU_PORT" \
    --encoder_device cuda:0 \
    "$@"
echo "[run_grpo] trainer finished."
