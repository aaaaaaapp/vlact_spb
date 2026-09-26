#!/usr/bin/env bash
# Fallback: direct curl/wget download of VLAct PI RoboTwin checkpoint (resume-friendly).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/setup/proxy.env"

BASE_URL="${HF_BASE_URL:-https://hf-mirror.com/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/resolve/main}"
LOCAL_DIR="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune"
LOG="${ROOT}/logs/setup/download_vlact_robotwin_pi_curl.log"
mkdir -p "${LOCAL_DIR}/checkpoints" "$(dirname "${LOG}")"

download_one() {
  local relpath="$1"
  local out="${LOCAL_DIR}/${relpath}"
  mkdir -p "$(dirname "${out}")"
  local url="${BASE_URL}/${relpath}"
  echo "[$(date -Is)] GET ${url}" | tee -a "${LOG}"
  curl -L --retry 30 --retry-delay 10 --retry-all-errors \
    -C - --proxy "${https_proxy}" \
    -o "${out}" "${url}" >> "${LOG}" 2>&1
  ls -lh "${out}" | tee -a "${LOG}"
}

validate_ckpt() {
  [[ -f "$1" ]] || return 1
  (( $(stat -c%s "$1") > 10000000000 )) || return 1
  /mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python - <<'PY' "$1"
import sys, zipfile
try:
    zipfile.ZipFile(sys.argv[1]).testzip()
except Exception as e:
    print(f"[validate] zip fail: {e}", file=sys.stderr)
    sys.exit(1)
print("[validate] zip ok")
PY
}

CKPT="${LOCAL_DIR}/checkpoints/steps_100000_pytorch_model.pt"
if validate_ckpt "${CKPT}" 2>/dev/null; then
  echo "[$(date -Is)] checkpoint already complete and valid" | tee -a "${LOG}"
  exit 0
fi
rm -f "${CKPT}"

for f in config.yaml dataset_statistics.json summary.jsonl README.md; do
  download_one "${f}"
done

download_one "checkpoints/steps_100000_pytorch_model.pt"

echo "[$(date -Is)] all files downloaded" | tee -a "${LOG}"
