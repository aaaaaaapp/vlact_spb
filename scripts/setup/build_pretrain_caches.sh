#!/usr/bin/env bash
# Build dataset caches/statistics required before pretraining (README section 3).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/playground/Datasets"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"
cd "$REPO_ROOT"

echo "========== $(date -Is) build_pretrain_caches start =========="

if [[ ! -f "${DATA_ROOT}/InternData-A1/split_aloha_tasks.txt" ]]; then
  echo "ERROR: missing ${DATA_ROOT}/InternData-A1/split_aloha_tasks.txt"
  echo "Run scripts/setup/postprocess_datasets.sh after InternData-A1 download completes."
  exit 1
fi

python examples/InternA1/preprocess/compute_caches_split_aloha.py \
  --data-root "${DATA_ROOT}/InternData-A1" \
  --task-list "${DATA_ROOT}/InternData-A1/split_aloha_tasks.txt" \
  --robot-type split_aloha_flip_wrap50 --skip-existing

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

python examples/DROID/compute_filtered_caches_droid_action_delta.py \
  --data-root "${DATA_ROOT}" --data-name DROID_100 \
  --robot-type oxe_droid_exterior1_wrist_manualvel_strict_50 \
  --filter-delta-eef-steps --skip-existing

python examples/MolmoAct/compute_caches_molmoact_delta_eef.py \
  --data-root "${DATA_ROOT}/MolmoAct-Dataset" --mode filtered \
  --robot-types \
    molmoact_franka_exterior1_wrist_manualvel_strict_50 \
    molmoact_franka_exterior2_wrist_manualvel_strict_50 \
  --filter-delta-eef-steps --skip-existing

echo "========== $(date -Is) build_pretrain_caches done =========="
