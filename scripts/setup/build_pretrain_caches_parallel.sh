#!/usr/bin/env bash
# Build pretrain caches in parallel where datasets are independent (lossless, same outputs).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/playground/Datasets"
LOG_DIR="${REPO_ROOT}/logs/setup"
mkdir -p "${LOG_DIR}"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"
cd "$REPO_ROOT"

log() { echo "$(date -Is) [cache-parallel] $*"; }

run_job() {
  local name="$1"
  shift
  local logfile="${LOG_DIR}/cache_job_${name}.log"
  log "start ${name} -> ${logfile}"
  if "$@" >>"${logfile}" 2>&1; then
    log "done ${name}"
    return 0
  fi
  log "FAILED ${name} (see ${logfile})"
  return 1
}

PIDS=()
NAMES=()

launch() {
  local name="$1"
  shift
  run_job "${name}" "$@" &
  PIDS+=("$!")
  NAMES+=("${name}")
}

wait_all() {
  local fail=0 i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      log "job failed: ${NAMES[$i]}"
      fail=1
    fi
  done
  PIDS=()
  NAMES=()
  return "${fail}"
}

log "========== build_pretrain_caches_parallel start =========="

# Phase A: datasets with no cross-deps
if [[ -f "${DATA_ROOT}/InternData-A1/split_aloha_tasks.txt" ]] \
  && [[ -s "${DATA_ROOT}/InternData-A1/split_aloha_tasks.txt" ]]; then
  launch interna1 python examples/InternA1/preprocess/compute_caches_split_aloha.py \
    --data-root "${DATA_ROOT}/InternData-A1" \
    --task-list "${DATA_ROOT}/InternData-A1/split_aloha_tasks.txt" \
    --robot-type split_aloha_flip_wrap50 --skip-existing
else
  log "skip interna1 cache (split_aloha_tasks.txt missing/empty)"
fi

if [[ -d "${DATA_ROOT}/RoboCOIN" ]]; then
  launch robocoin python examples/RoboCoin/compute_stats_gr00t.py \
    --data-root "${DATA_ROOT}/RoboCOIN" \
    --robot-type ROBOCOIN.AgileX_flip_wrap --skip-existing
fi

if [[ -f "${DATA_ROOT}/DROID_100/meta/info.json" ]]; then
  launch droid_100 python examples/DROID/compute_filtered_caches_droid_action_delta.py \
    --data-root "${DATA_ROOT}" --data-name DROID_100 \
    --robot-type oxe_droid_exterior1_wrist_manualvel_strict_50 \
    --filter-delta-eef-steps --skip-existing
fi

wait_all || exit 1

# Phase B: full DROID (needs upload complete; skip during early parallel prep)
if [[ "${SKIP_DROID:-0}" == "1" ]]; then
  log "skip DROID cache (SKIP_DROID=1)"
elif [[ -f "${DATA_ROOT}/DROID/meta/info.json" ]]; then
  launch droid_ext1 python examples/DROID/compute_filtered_caches_droid_action_delta.py \
    --data-root "${DATA_ROOT}" --data-name DROID \
    --robot-type oxe_droid_exterior1_wrist_manualvel_strict_50 \
    --filter-delta-eef-steps --skip-existing
  launch droid_ext2 python examples/DROID/compute_filtered_caches_droid_action_delta.py \
    --data-root "${DATA_ROOT}" --data-name DROID \
    --robot-type oxe_droid_exterior2_wrist_manualvel_strict_50 \
    --filter-delta-eef-steps --skip-existing
  wait_all || exit 1
else
  log "skip DROID cache (meta/info.json missing)"
fi

# Phase C: MolmoAct (needs v21 conversion)
if [[ -d "${DATA_ROOT}/MolmoAct-Dataset/molmoact_household_v21/meta" ]]; then
  launch molmoact python examples/MolmoAct/compute_caches_molmoact_delta_eef.py \
    --data-root "${DATA_ROOT}/MolmoAct-Dataset" --mode filtered \
    --robot-types \
      molmoact_franka_exterior1_wrist_manualvel_strict_50 \
      molmoact_franka_exterior2_wrist_manualvel_strict_50 \
    --filter-delta-eef-steps --skip-existing
  wait_all || exit 1
else
  log "skip MolmoAct cache (v21 not ready)"
fi

log "========== build_pretrain_caches_parallel done =========="
