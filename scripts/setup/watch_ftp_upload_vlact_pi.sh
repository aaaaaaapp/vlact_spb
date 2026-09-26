#!/usr/bin/env bash
# 监控 /mnt/jfzn/ftp_upload，发现 steps_100000_pytorch_model.pt 后自动导入+smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FTP_DIR="${FTP_UPLOAD_DIR:-/mnt/jfzn/ftp_upload}"
LOG="${ROOT}/logs/setup/watch_ftp_upload_vlact_pi.log"
EXPECTED_SIZE=11276441870
INTERVAL="${WATCH_INTERVAL:-30}"

mkdir -p "$(dirname "${LOG}")"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

find_ckpt() {
  find "${FTP_DIR}" -maxdepth 3 -name 'steps_100000_pytorch_model.pt' -type f 2>/dev/null \
    | while read -r f; do
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if (( sz >= EXPECTED_SIZE - 1048576 && sz <= EXPECTED_SIZE + 1048576 )); then
          echo "$f"
          return 0
        fi
      done
}

log "watch start dir=${FTP_DIR} interval=${INTERVAL}s"

while true; do
  if ckpt=$(find_ckpt | head -1) && [[ -n "${ckpt}" ]]; then
    log "found ${ckpt} ($(ls -lh "${ckpt}"))"
    if RUN_SMOKE=1 bash "${ROOT}/scripts/setup/import_vlact_pi_from_ftp_upload.sh" "${ckpt}"; then
      log "import+smoke success"
      exit 0
    fi
    log "import failed, keep watching"
  fi
  sleep "${INTERVAL}"
done
