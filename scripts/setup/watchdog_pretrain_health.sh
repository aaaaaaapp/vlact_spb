#!/usr/bin/env bash
# Overnight health check for VLAct Qwen3 pretrain on node3.
# Restarts the job if ranks die; does not kill a healthy run just to pick up code.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP="${REPO_ROOT}/scripts/setup"
LOG_DIR="${REPO_ROOT}/logs/setup"
mkdir -p "${LOG_DIR}"
# shellcheck source=/dev/null
source "${SETUP}/nodes.env"

USER="${CLUSTER_USER:-liujn}"
NODE="${PRETRAIN_NODE:-${NODE003_IP}}"
POLL_SEC="${WATCHDOG_POLL_SEC:-180}"
MAX_RESTARTS="${WATCHDOG_MAX_RESTARTS:-12}"
COOLDOWN_SEC="${WATCHDOG_COOLDOWN_SEC:-180}"
HUNG_SEC="${WATCHDOG_HUNG_SEC:-900}"
MIN_VLACT_MEM_MB="${WATCHDOG_MIN_MEM_MB:-40000}"
STATE="${LOG_DIR}/watchdog_health.state"
WLOG="${LOG_DIR}/watchdog_health.log"
HB="${REPO_ROOT}/results/Checkpoints/vlact_qwen3_pretrain/train_heartbeat.json"
LOCK="${LOG_DIR}/watchdog_health.lock"

log() { echo "$(date -Is) [watchdog-health] $*"; }

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "$(date -Is) [watchdog-health] another instance holds ${LOCK}" >&2
  exit 0
fi

restarts=0
last_restart=0
if [[ -f "${STATE}" ]]; then
  # shellcheck source=/dev/null
  source "${STATE}" || true
fi
save_state() {
  cat >"${STATE}" <<EOF
restarts=${restarts}
last_restart=${last_restart}
last_status=${last_status:-unknown}
EOF
}

remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${USER}@${NODE}" "$1"
}

rank_count() {
  remote 'pgrep -c -f "starVLA/training/train_starvla_cotrain.py" || true' | tail -n1 | tr -d '[:space:]'
}

gpu_csv() {
  remote 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true'
}

latest_log() {
  if [[ -L "${LOG_DIR}/pretrain_qwen3_latest.log" ]]; then
    readlink -f "${LOG_DIR}/pretrain_qwen3_latest.log"
  else
    ls -1t "${LOG_DIR}"/pretrain_qwen3_*.log 2>/dev/null | head -1 || true
  fi
}

log_has_crash() {
  local f="$1"
  [[ -n "${f}" && -f "${f}" ]] || return 1
  # Only the last 80KB so old crashes do not retrigger forever.
  python3 - "$f" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
data = p.read_bytes()[-80000:].decode("utf-8", "replace")
needles = (
    "ChildFailedError",
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "AssertionError",
    "NCCL error",
    "Watchdog caught collective operation timeout",
)
# A living job can print historical Traceback in comments; require fatal wrappers.
fatal = (
    "ChildFailedError" in data
    or "CUDA out of memory" in data
    or "failed (exitcode:" in data
    or "SpeculationLog diverged" in data
    or "no_sync context manager is incompatible" in data
)
print("yes" if fatal else "no")
PY
}

kill_job() {
  log "killing VLAct train processes on ${NODE}"
  remote 'pkill -TERM -f "starVLA/training/train_starvla_cotrain.py" || true; pkill -TERM -f "scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh" || true' || true
  sleep 5
  remote 'pkill -KILL -f "starVLA/training/train_starvla_cotrain.py" || true; pkill -KILL -f "scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh" || true' || true
  sleep 3
}

restart_job() {
  local now
  now="$(date +%s)"
  if (( now - last_restart < COOLDOWN_SEC )); then
    log "skip restart (cooldown ${COOLDOWN_SEC}s)"
    return 0
  fi
  if (( restarts >= MAX_RESTARTS )); then
    log "FATAL: reached MAX_RESTARTS=${MAX_RESTARTS}; not restarting"
    last_status=max_restarts
    save_state
    return 1
  fi
  kill_job
  restarts=$((restarts + 1))
  last_restart="${now}"
  save_state
  log "restart #${restarts}: launch_pretrain_node3.sh"
  export PRETRAIN_NODE="${NODE}"
  export WANDB_ENTITY="${WANDB_ENTITY:-app123}"
  bash "${SETUP}/launch_pretrain_node3.sh"
  last_status=restarted
  save_state
}

job_healthy() {
  local ranks mem_max util_max hung_hb
  ranks="$(rank_count)"
  ranks="${ranks:-0}"
  if ! [[ "${ranks}" =~ ^[0-9]+$ ]]; then ranks=0; fi

  local csv min_mem=999999 max_mem=0 max_util=0 n=0
  csv="$(gpu_csv)"
  while IFS=',' read -r mem util; do
    mem="${mem// /}"
    util="${util// /}"
    [[ -z "${mem}" ]] && continue
    n=$((n + 1))
    (( mem > max_mem )) && max_mem="${mem}"
    (( mem < min_mem )) && min_mem="${mem}"
    (( util > max_util )) && max_util="${util}"
  done <<<"${csv}"

  log "ranks=${ranks} gpu_mem_max=${max_mem}MB gpu_util_max=${max_util}% log=$(basename "$(latest_log 2>/dev/null || echo none)")"

  if (( ranks < 8 )); then
    last_status=dead_ranks
    return 1
  fi

  local f
  f="$(latest_log)"
  if [[ -n "${f}" ]] && [[ "$(log_has_crash "${f}")" == "yes" ]]; then
    # ranks still up? elastic may be tearing down
    last_status=crash_in_log
    return 1
  fi

  if (( max_mem < MIN_VLACT_MEM_MB )); then
    last_status=low_gpu_mem
    return 1
  fi

  hung_hb=0
  if [[ -f "${HB}" ]]; then
    local hb_age
    hb_age=$(( $(date +%s) - $(stat -c %Y "${HB}") ))
    if (( hb_age > HUNG_SEC )); then
      log "heartbeat stale ${hb_age}s > ${HUNG_SEC}s"
      hung_hb=1
    fi
  fi
  if (( hung_hb == 1 && max_util < 5 )); then
    last_status=hung
    return 1
  fi

  last_status=ok
  return 0
}

log "started node=${NODE} poll=${POLL_SEC}s max_restarts=${MAX_RESTARTS}"
last_status=boot
save_state

while true; do
  if job_healthy; then
    save_state
  else
    log "unhealthy status=${last_status}; extracting tail"
    f="$(latest_log || true)"
    if [[ -n "${f}" && -f "${f}" ]]; then
      tail -n 40 "${f}" | tee -a "${WLOG}" >/dev/null || true
    fi
    restart_job || true
  fi
  sleep "${POLL_SEC}"
done
