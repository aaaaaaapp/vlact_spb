#!/usr/bin/env bash
# Link ftp_upload datasets, postprocess, then build pretrain caches (staged).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/setup"
mkdir -p "${LOG_DIR}"

echo "========== $(date -Is) run_pretrain_data_setup start =========="

bash "${REPO_ROOT}/scripts/setup/link_datasets_from_ftp_upload.sh" \
  | tee "${LOG_DIR}/link_datasets.log"

bash "${REPO_ROOT}/scripts/setup/postprocess_datasets.sh" \
  | tee "${LOG_DIR}/postprocess_datasets.log"

# Phase-1 caches (skip MolmoAct until v21 conversion finishes).
source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"
cd "${REPO_ROOT}"
DATA_ROOT="${REPO_ROOT}/playground/Datasets"

{
  echo "========== $(date -Is) build_pretrain_caches phase1 (no MolmoAct) =========="

  if [[ -f "${DATA_ROOT}/InternData-A1/split_aloha_tasks.txt" ]]; then
    python examples/InternA1/preprocess/compute_caches_split_aloha.py \
      --data-root "${DATA_ROOT}/InternData-A1" \
      --task-list "${DATA_ROOT}/InternData-A1/split_aloha_tasks.txt" \
      --robot-type split_aloha_flip_wrap50 --skip-existing
  else
    echo "[WARN] missing split_aloha_tasks.txt"
  fi

  python examples/RoboCoin/compute_stats_gr00t.py \
    --data-root "${DATA_ROOT}/RoboCOIN" \
    --robot-type ROBOCOIN.AgileX_flip_wrap --skip-existing

  for robot_type in \
    oxe_droid_exterior1_wrist_manualvel_strict_50 \
    oxe_droid_exterior2_wrist_manualvel_strict_50; do
    python examples/DROID/compute_filtered_caches_droid_action_delta.py \
      --data-root "${DATA_ROOT}" --data-name DROID \
      --robot-type "${robot_type}" --filter-delta-eef-steps --skip-existing
  done

  if [[ -f "${DATA_ROOT}/DROID_100/meta/info.json" ]]; then
    python examples/DROID/compute_filtered_caches_droid_action_delta.py \
      --data-root "${DATA_ROOT}" --data-name DROID_100 \
      --robot-type oxe_droid_exterior1_wrist_manualvel_strict_50 \
      --filter-delta-eef-steps --skip-existing
  fi

  if [[ -d "${DATA_ROOT}/MolmoAct-Dataset/molmoact_household_v21/meta" ]]; then
    echo "========== $(date -Is) build_pretrain_caches phase2 (MolmoAct) =========="
    python examples/MolmoAct/compute_caches_molmoact_delta_eef.py \
      --data-root "${DATA_ROOT}/MolmoAct-Dataset" --mode filtered \
      --robot-types \
        molmoact_franka_exterior1_wrist_manualvel_strict_50 \
        molmoact_franka_exterior2_wrist_manualvel_strict_50 \
      --filter-delta-eef-steps --skip-existing
  else
    echo "[WARN] MolmoAct v21 not ready; rerun after convert completes:"
    echo "  bash scripts/setup/build_pretrain_caches.sh"
  fi

  echo "========== $(date -Is) run_pretrain_data_setup done =========="
} | tee "${LOG_DIR}/build_pretrain_caches.log"
