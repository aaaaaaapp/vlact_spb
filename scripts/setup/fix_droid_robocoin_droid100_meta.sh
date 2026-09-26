#!/usr/bin/env bash
# Fix missing modality.json for DROID/RoboCOIN and re-prepare DROID_100 from RoboInter.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/playground/Datasets"
ROBOINTER_SRC="${DATA_ROOT}/RoboInter-Data/Annotation_with_action_lerobotv21/lerobot_droid_anno"
LOG_DIR="${REPO_ROOT}/logs/setup"
mkdir -p "${LOG_DIR}"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"
cd "${REPO_ROOT}"

log() { echo "$(date -Is) [fix-meta] $*"; }

log "========== DROID / RoboCOIN materialize + modality =========="
bash scripts/setup/materialize_writable_ftp_datasets.sh 2>&1 | tee -a "${LOG_DIR}/fix_meta_materialize.log"

log "========== DROID_100 re-prepare from RoboInter =========="
if [[ ! -d "${ROBOINTER_SRC}/meta" ]]; then
  echo "Missing RoboInter source: ${ROBOINTER_SRC}" >&2
  exit 1
fi

DROID100="${DATA_ROOT}/DROID_100"
for sub in data videos meta .cache; do
  if [[ -e "${DROID100}/${sub}" ]]; then
    log "remove stale DROID_100/${sub}"
    rm -rf "${DROID100}/${sub}"
  fi
done

python examples/DROID/prepare_droid_100.py \
  --source-dir "${ROBOINTER_SRC}" \
  --dataset-dir "${DROID100}"

log "DROID_100 layout:"
ls -la "${DROID100}/meta/"

log "========== done =========="
