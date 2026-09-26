#!/usr/bin/env bash
# Priority download for VLAct RobotWin checkpoint (single task, no hf_transfer).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/setup/proxy.env"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=7200
export HF_HUB_ETAG_TIMEOUT=600

REPO="StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune"
LOCAL_DIR="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune"
CKPT="${LOCAL_DIR}/checkpoints/steps_100000_pytorch_model.pt"
LOG="${ROOT}/logs/setup/download_vlact_robotwin_pi_ckpt.log"
PAUSE_FLAG="${ROOT}/logs/setup/PAUSE_DATASET_DOWNLOADS_FOR_ROBOTWIN_CKPT"

mkdir -p "$(dirname "${LOG}")" "${LOCAL_DIR}/checkpoints"
touch "${PAUSE_FLAG}"

source /mnt/jfzn/liujn/miniconda3/etc/profile.d/conda.sh
conda activate vlact

if [[ -f "${CKPT}" ]] && (( $(stat -c%s "${CKPT}") > 10000000000 )); then
  echo "[$(date -Is)] already complete: ${CKPT}" | tee -a "${LOG}"
  exit 0
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date -Is)] priority download attempt ${attempt}" | tee -a "${LOG}"

  if hf download "${REPO}" \
      --include 'checkpoints/steps_100000_pytorch_model.pt' \
      --local-dir "${LOCAL_DIR}" \
      --max-workers 2 >> "${LOG}" 2>&1; then
    if [[ -f "${CKPT}" ]] && (( $(stat -c%s "${CKPT}") > 10000000000 )); then
      echo "[$(date -Is)] SUCCESS $(ls -lh "${CKPT}")" | tee -a "${LOG}"
      exit 0
    fi
  fi

  sz=$(stat -c%s "${CKPT}" 2>/dev/null || echo 0)
  echo "[$(date -Is)] retry in 60s (current ${sz} bytes)" | tee -a "${LOG}"
  sleep 60
done
