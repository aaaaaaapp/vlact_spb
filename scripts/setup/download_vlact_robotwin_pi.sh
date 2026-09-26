#!/usr/bin/env bash
# Download VLAct Qwen3PI RoboTwin checkpoint with retries (proxy-friendly).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/setup/proxy.env"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-600}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-7200}"

REPO="StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune"
LOCAL_DIR="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune"
LOG="${ROOT}/logs/setup/download_vlact_robotwin_pi.log"
MAX_RETRIES="${MAX_RETRIES:-50}"
RETRY_SLEEP="${RETRY_SLEEP:-120}"

mkdir -p "$(dirname "${LOG}")" "$(dirname "${LOCAL_DIR}")"

if [[ -f "${LOCAL_DIR}/checkpoints/steps_100000_pytorch_model.pt" ]]; then
  echo "[$(date -Is)] Already downloaded: ${LOCAL_DIR}/checkpoints/steps_100000_pytorch_model.pt"
  ls -lh "${LOCAL_DIR}/checkpoints/steps_100000_pytorch_model.pt"
  exit 0
fi

source /mnt/jfzn/liujn/miniconda3/etc/profile.d/conda.sh
conda activate vlact

attempt=0
while (( attempt < MAX_RETRIES )); do
  attempt=$((attempt + 1))
  echo "[$(date -Is)] download attempt ${attempt}/${MAX_RETRIES} -> ${LOCAL_DIR}" | tee -a "${LOG}"

  if hf download "${REPO}" \
      --local-dir "${LOCAL_DIR}" \
      --max-workers 2 \
      >> "${LOG}" 2>&1; then
    if [[ -f "${LOCAL_DIR}/checkpoints/steps_100000_pytorch_model.pt" ]]; then
      echo "[$(date -Is)] SUCCESS" | tee -a "${LOG}"
      ls -lh "${LOCAL_DIR}/checkpoints/steps_100000_pytorch_model.pt" | tee -a "${LOG}"
      exit 0
    fi
    echo "[$(date -Is)] hf download exited 0 but checkpoint missing" | tee -a "${LOG}"
  else
    echo "[$(date -Is)] hf download failed (attempt ${attempt})" | tee -a "${LOG}"
    tail -5 "${LOG}" || true
  fi

  sleep "${RETRY_SLEEP}"
done

echo "[$(date -Is)] FAILED after ${MAX_RETRIES} attempts" | tee -a "${LOG}"
exit 1
