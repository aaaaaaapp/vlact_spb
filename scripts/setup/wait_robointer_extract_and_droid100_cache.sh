#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTRACT_LOG="${REPO_ROOT}/logs/setup/extract_robointer_droid_data.log"
DATA_EXTRACTED="${REPO_ROOT}/playground/Datasets/RoboInter-lerobot_droid_anno-data"
DROID100="${REPO_ROOT}/playground/Datasets/DROID_100"
LOG="${REPO_ROOT}/logs/setup/droid100_cache_after_extract.log"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"

log() { echo "$(date -Is) [droid100-wait] $*" | tee -a "${LOG}"; }

log "waiting for RoboInter extract to finish..."
while pgrep -f "extract_robointer_droid_data.sh" >/dev/null; do
  sleep 60
done
while ! grep -q "========== done ==========" "${EXTRACT_LOG}" 2>/dev/null; do
  sleep 30
done

pq="$(find "${DATA_EXTRACTED}" -name '*.parquet' 2>/dev/null | wc -l)"
log "extracted parquet count=${pq}"
if [[ "${pq}" -lt 1000 ]]; then
  log "ERROR too few parquet files after extract"
  exit 1
fi

cd "${REPO_ROOT}"
if [[ -L "${DROID100}/data" ]] || [[ -d "${DROID100}/data" ]]; then
  rm -rf "${DROID100}/data"
fi
ln -sfn "${DATA_EXTRACTED}" "${DROID100}/data"
log "relinked DROID_100/data -> ${DATA_EXTRACTED}"

rm -f "${DROID100}/meta/steps_data_index_filtered_"*.pkl "${DROID100}/meta/stats_gr00t_filtered_"*.json

python examples/DROID/compute_filtered_caches_droid_action_delta.py \
  --data-root "${REPO_ROOT}/playground/Datasets" --data-name DROID_100 \
  --robot-type oxe_droid_exterior1_wrist_manualvel_strict_50 \
  --filter-delta-eef-steps --skip-existing >> "${LOG}" 2>&1

log "DROID_100 cache finished"
