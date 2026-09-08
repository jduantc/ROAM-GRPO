#!/usr/bin/env bash
# Single-run training supervisor (step-freeze / crash -> auto-restart from checkpoint).
#
# The generation server is weight-synced to the trainer over a fixed NCCL group, so it can't be
# hot-swapped under a live trainer; the robust recovery for a frozen or dead run is to restart the
# run from its latest checkpoint.
#   - launches run_grpo.sh with the same args,
#   - watches the trainer's step counter in the run log,
#   - if the step freezes for STALL_MIN or the run dies before writing a final adapter, kills the
#     trainer+gen, reaps the GPUs, and relaunches run_grpo.sh with --init_adapter <latest ckpt>,
#   - stops after MAXR restarts (with COOLDOWN_MIN between) so a broken run can't crash-loop,
#   - writes a health snapshot to runs/_logs/HEALTH.txt every CHECK_S.
#
# Usage: pass the same env + args you'd give run_grpo.sh (a --run_name is required so the
# supervisor knows which log/checkpoints to track):
#   GEN_GPU=1 TRAIN_GPU=0 GEN_PORT=8000 \
#     bash supervise.sh --run_name roam --route_criterion all_correct \
#        --route_judge_url http://localhost:8265 --no_wandb
# Stop: touch runs/_logs/SUPERVISE_STOP   (kills the run and exits)
#
# Note: on restart the trainer resumes from the latest checkpoint but with a fresh --max_steps /
# --num_epochs budget. Set these values as desired
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$HERE/runs/_logs"; mkdir -p "$LOGD"
CHECK_S="${CHECK_S:-120}"; STALL_MIN="${STALL_MIN:-20}"; GRACE_MIN="${GRACE_MIN:-8}"
COOLDOWN_MIN="${COOLDOWN_MIN:-20}"; MAXR="${MAXR:-8}"
GEN_PORT="${GEN_PORT:-8000}"; TRAIN_GPU="${TRAIN_GPU:-0}"; GEN_GPU="${GEN_GPU:-2}"
STOP="$LOGD/SUPERVISE_STOP"; STATUS="$LOGD/HEALTH.txt"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] supervise: $*" | tee -a "$LOGD/supervise.log" >/dev/null; }

ARGS=("$@"); RUN_NAME=""
for i in "${!ARGS[@]}"; do [ "${ARGS[$i]}" = "--run_name" ] && RUN_NAME="${ARGS[$((i + 1))]}"; done
[ -z "$RUN_NAME" ] && { echo "supervise: --run_name <name> is required in the args" >&2; exit 2; }
RUNLOG="$LOGD/${RUN_NAME}.log"

run_pids(){  # PIDs of the actual run only: its run_grpo.sh wrapper + trainer. Positively matching
  # run_grpo.sh/train.py (never "supervise.sh") means this supervisor and its own subshells — which
  # share the run_name in their argv — are never miscounted as a live run.
  local p cl
  for p in $(pgrep -f "run_name ${RUN_NAME}" 2>/dev/null); do
    cl=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$cl" in *run_grpo.sh*|*train.py*) echo "$p" ;; esac
  done
}
latest_ckpt(){ ls -1dt "$HERE"/runs/"${RUN_NAME}"*/checkpoint-* 2>/dev/null | head -1; }
done_final(){ ls -d "$HERE"/runs/"${RUN_NAME}"*/final >/dev/null 2>&1; }
get_step(){ [ -f "$RUNLOG" ] || { echo ""; return; }
  local s; s=$(grep -aoE "\[rbc\] steps=[0-9]+" "$RUNLOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  [ -z "$s" ] && s=$(grep -aoE "[0-9]+/[0-9]+ \[" "$RUNLOG" 2>/dev/null | tail -1 | grep -oE "^[0-9]+"); echo "$s"; }

launch(){  # $1 = init adapter path (empty => honor whatever is already in ARGS / base model)
  local init="$1" extra=()
  [ -n "$init" ] && extra=(--init_adapter "$init")   # last --init_adapter wins in argparse
  say "launch run_grpo.sh (run_name=$RUN_NAME, init=${init:-as-given})"
  GEN_GPU="$GEN_GPU" TRAIN_GPU="$TRAIN_GPU" GEN_PORT="$GEN_PORT" \
    POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen3.5-2B}" \
    setsid nohup bash "$HERE/run_grpo.sh" "${ARGS[@]}" "${extra[@]}" > "$RUNLOG" 2>&1 & disown
}

kill_run(){
  say "killing run (trainer+gen), reaping GPUs $TRAIN_GPU,$GEN_GPU"
  for p in $(run_pids); do kill -9 "$p" 2>/dev/null; done
  for p in $(ss -ltnp 2>/dev/null | grep ":$GEN_PORT" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill -9 "$p" 2>/dev/null; done
  sleep 2
  for g in "$TRAIN_GPU" "$GEN_GPU"; do
    for p in $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  done
  sleep 4
}

restarts=0; last_restart=0; pstep=""; plast=$(date +%s)
launch ""
while true; do
  [ -f "$STOP" ] && { say "stop requested"; kill_run; exit 0; }
  sleep "$CHECK_S"
  now=$(date +%s)
  done_final && { say "run finished (final adapter present) — exiting"; break; }
  step=$(get_step); alive=$(run_pids | head -1); action=""
  if [ -z "$step" ]; then
    # no step logged yet: mid-startup, unless the process is gone AND the log has gone stale
    if [ -z "$alive" ] && [ -z "$(find "$LOGD" -maxdepth 1 -name "${RUN_NAME}.log" -newermt "-${GRACE_MIN} min" 2>/dev/null)" ]; then
      action="died during startup"
    fi
  elif [ "$step" != "$pstep" ]; then
    pstep="$step"; plast="$now"                        # advanced -> healthy
  else
    local_idle=$(( (now - plast) / 60 ))
    [ -z "$alive" ] && action="vanished at step $step"
    [ -n "$alive" ] && [ "$local_idle" -ge "$STALL_MIN" ] && action="frozen at step $step (${local_idle}min)"
  fi

  { echo "=== HEALTH $(date '+%F %T')  run=$RUN_NAME ==="
    echo "step=${step:-none}  last_advance=$(( (now - plast) / 60 ))min  alive=$([ -n "$alive" ] && echo yes || echo no)  restarts=$restarts/$MAXR"
    echo "latest_ckpt=$(latest_ckpt)"
    echo "status=${action:-ok}"
    echo "stop: touch $STOP"; } > "$STATUS"

  [ -z "$action" ] && continue
  [ $(( (now - last_restart) / 60 )) -lt "$COOLDOWN_MIN" ] && { say "… within cooldown, skip ($action)"; continue; }
  [ "$restarts" -ge "$MAXR" ] && { say "hit MAXR=$MAXR restarts — giving up ($action); manual attention"; break; }
  say "restart #$((restarts + 1)) — $action"
  kill_run
  launch "$(latest_ckpt)"
  restarts=$((restarts + 1)); last_restart="$now"; pstep=""; plast=$(date +%s)
done
