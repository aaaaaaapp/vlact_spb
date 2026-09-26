#!/usr/bin/env bash
# Wait for VLAct PI checkpoint download, then run RoboTwin smoke eval once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
LOG="${ROOT}/logs/setup/watchdog_robotwin_smoke.log"
POLL_SEC="${POLL_SEC:-300}"

mkdir -p "$(dirname "${LOG}")"

if [[ -f "${CKPT}" ]]; then
  echo "[$(date -Is)] checkpoint already present, running smoke" | tee -a "${LOG}"
  bash "${ROOT}/scripts/setup/run_robotwin_smoke_vlact_pi.sh" 2>&1 | tee -a "${LOG}"
  exit $?
fi

echo "[$(date -Is)] waiting for ${CKPT}" | tee -a "${LOG}"
while true; do
  if [[ -f "${CKPT}" ]]; then
    sz=$(stat -c%s "${CKPT}" 2>/dev/null || echo 0)
    # full checkpoint ~11.3 GB
    if (( sz > 10000000000 )); then
      echo "[$(date -Is)] checkpoint ready (${sz} bytes), running smoke" | tee -a "${LOG}"
      bash "${ROOT}/scripts/setup/run_robotwin_smoke_vlact_pi.sh" 2>&1 | tee -a "${LOG}"
      exit $?
    fi
    echo "[$(date -Is)] checkpoint incomplete (${sz} bytes), keep waiting" | tee -a "${LOG}"
  else
    du -sh /mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune 2>/dev/null | tee -a "${LOG}" || true
    tail -1 "${ROOT}/logs/setup/download_vlact_robotwin_pi.log" 2>/dev/null | tee -a "${LOG}" || true
  fi
  sleep "${POLL_SEC}"
done
