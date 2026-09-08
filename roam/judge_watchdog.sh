#!/usr/bin/env bash
# Judge health watchdog (single ROAM judge).
#
# The 9B off-mode judge can wedge (the engine hangs on "Rendering prompts" and
# /generate/ stops responding while the process stays alive) and, over long runs, can slowly
# degrade. This script starts the judge if it isn't up, then keeps it healthy:
#   - Restart after 2 consecutive probe failures, and
#   - Proactively recycle every RECYCLE_LOOPS probes (~3h by default) to avoid slow degradation.
#
# Point training at it with:  --route_judge_url http://localhost:$JUDGE_PORT
# Run it alongside training (e.g. in its own tmux pane / nohup).
# Stop: touch runs/_logs/JUDGE_WD_STOP
#
# All knobs are env-overridable (defaults in brackets):
#   JUDGE_MODEL [Qwen/Qwen3.5-9B]  JUDGE_PORT [8265]  JUDGE_GPU [1]
#   JUDGE_UTIL [0.85]  JUDGE_MAXLEN [12288]  PROBE_EVERY [150]s  RECYCLE_LOOPS [72]
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Activate your conda env if needed:
# source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate grpo_train

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-9B}"
JUDGE_PORT="${JUDGE_PORT:-8265}"
JUDGE_GPU="${JUDGE_GPU:-1}"
JUDGE_UTIL="${JUDGE_UTIL:-0.85}"
JUDGE_MAXLEN="${JUDGE_MAXLEN:-12288}"
PROBE_EVERY="${PROBE_EVERY:-150}"        # seconds between probes
RECYCLE_LOOPS="${RECYCLE_LOOPS:-72}"     # ~72 * 150s ≈ 3h between proactive recycles

LOGD="$HERE/runs/_logs"; mkdir -p "$LOGD"
JLOG="$LOGD/judge_server_${JUDGE_PORT}.log"; STOP="$LOGD/JUDGE_WD_STOP"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] judge_wd: $*" | tee -a "$LOGD/judge_wd.log" >/dev/null; }

probe(){ timeout 30 curl -s -X POST "http://localhost:$JUDGE_PORT/generate/" \
    -H 'Content-Type: application/json' \
    -d '{"prompts":["2+2="],"n":1,"max_tokens":4,"temperature":0.0}' \
    -o /dev/null -w "%{http_code}" 2>/dev/null; }

start_judge(){
  say "starting judge $JUDGE_MODEL on GPU $JUDGE_GPU (port $JUDGE_PORT)"
  # free the port + GPU first (kill a wedged/ghost server)
  for p in $(ss -ltnp 2>/dev/null | grep ":$JUDGE_PORT" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill -9 "$p" 2>/dev/null; done
  for p in $(nvidia-smi -i "$JUDGE_GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  CUDA_VISIBLE_DEVICES="$JUDGE_GPU" setsid nohup python -m trl.scripts.vllm_serve \
      --model "$JUDGE_MODEL" --host 127.0.0.1 --port "$JUDGE_PORT" \
      --tensor_parallel_size 1 --data_parallel_size 1 \
      --gpu_memory_utilization "$JUDGE_UTIL" --max_model_len "$JUDGE_MAXLEN" \
      --enforce_eager False > "$JLOG" 2>&1 & disown
  say "launched judge $JUDGE_PORT (log $JLOG)"
}

[ "$(probe)" = "200" ] && say "judge $JUDGE_PORT already healthy; supervising" || start_judge
fails=0; since=0
say "watching judge $JUDGE_PORT (probe every ${PROBE_EVERY}s; restart after 2 fails; recycle every ${RECYCLE_LOOPS} probes)"
while true; do
  [ -f "$STOP" ] && { say "stopped"; exit 0; }
  code=$(probe)
  if [ "$code" = "200" ]; then
    fails=0; since=$((since + 1))
    if [ "$since" -ge "$RECYCLE_LOOPS" ]; then
      say "PROACTIVE recycle judge $JUDGE_PORT (~${RECYCLE_LOOPS} probes uptime)"
      start_judge; fails=0; since=0; sleep 70
    fi
  else
    fails=$((fails + 1)); say "judge $JUDGE_PORT probe=$code consecutive_fails=$fails"
    if [ "$fails" -ge 2 ]; then start_judge; fails=0; since=0; sleep 70; fi
  fi
  sleep "$PROBE_EVERY"
done
