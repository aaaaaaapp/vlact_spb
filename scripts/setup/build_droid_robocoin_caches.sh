#!/usr/bin/env bash
# Build caches for DROID / RoboCOIN / DROID_100 after modality.json is ready.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/playground/Datasets"
LOG_DIR="${REPO_ROOT}/logs/setup"
mkdir -p "${LOG_DIR}"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"
cd "${REPO_ROOT}"

log() { echo "$(date -Is) [droid-rc-cache] $*"; }

run() {
  local name="$1"
  shift
  local logfile="${LOG_DIR}/cache_job_${name}.log"
  log "start ${name} -> ${logfile}"
  if "$@" >>"${logfile}" 2>&1; then
    log "done ${name}"
  else
    log "FAILED ${name} (see ${logfile})"
    return 1
  fi
}

log "========== start =========="

run robocoin python examples/RoboCoin/compute_stats_gr00t.py \
  --data-root "${DATA_ROOT}/RoboCOIN" \
  --robot-type ROBOCOIN.AgileX_flip_wrap --skip-existing

run droid_100 python examples/DROID/compute_filtered_caches_droid_action_delta.py \
  --data-root "${DATA_ROOT}" --data-name DROID_100 \
  --robot-type oxe_droid_exterior1_wrist_manualvel_strict_50 \
  --filter-delta-eef-steps --skip-existing

run droid_ext1 python examples/DROID/compute_filtered_caches_droid_action_delta.py \
  --data-root "${DATA_ROOT}" --data-name DROID \
  --robot-type oxe_droid_exterior1_wrist_manualvel_strict_50 \
  --filter-delta-eef-steps --skip-existing

run droid_ext2 python examples/DROID/compute_filtered_caches_droid_action_delta.py \
  --data-root "${DATA_ROOT}" --data-name DROID \
  --robot-type oxe_droid_exterior2_wrist_manualvel_strict_50 \
  --filter-delta-eef-steps --skip-existing

log "========== done =========="
