#!/usr/bin/env bash
# 从 SFTP 落盘目录 /mnt/jfzn/ftp_upload 导入 VLAct PI 权重并校验
# 外部机: sftp -P 45432 -i key ftpuser@106.75.235.248  →  put 到 /upload/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FTP_DIR="${FTP_UPLOAD_DIR:-/mnt/jfzn/ftp_upload}"
SRC="${1:-}"
DEST="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
EXPECTED_SIZE=11276441870
LOG="${ROOT}/logs/setup/import_vlact_pi_from_ftp_upload.log"

mkdir -p "$(dirname "${DEST}")" "$(dirname "${LOG}")"

if [[ -z "${SRC}" ]]; then
  echo "用法: $0 <ftp_upload 里的文件或目录>" >&2
  echo "示例:" >&2
  echo "  $0 /mnt/jfzn/ftp_upload/steps_100000_pytorch_model.pt" >&2
  echo "  $0 /mnt/jfzn/ftp_upload/VLAct_Qwen3PI_Robotwin_all_Finetune/" >&2
  echo "" >&2
  echo "最近上传:" >&2
  ls -lht "${FTP_DIR}" 2>/dev/null | head -8 >&2 || true
  exit 1
fi

resolve_ckpt() {
  local p="$1"
  if [[ -f "${p}" && "${p}" == *steps_100000_pytorch_model.pt ]]; then
    echo "${p}"; return 0
  fi
  if [[ -d "${p}" ]]; then
    local f="${p%/}/checkpoints/steps_100000_pytorch_model.pt"
    [[ -f "${f}" ]] && { echo "${f}"; return 0; }
    f="$(find "${p}" -name 'steps_100000_pytorch_model.pt' -type f 2>/dev/null | head -1)"
    [[ -n "${f}" ]] && { echo "${f}"; return 0; }
  fi
  return 1
}

CKPT_SRC="$(resolve_ckpt "${SRC}")" || { echo "[ERROR] 找不到 steps_100000_pytorch_model.pt in ${SRC}" >&2; exit 2; }

{
  echo "========== $(date -Is) import start =========="
  echo "SRC=${CKPT_SRC}"
  echo "DEST=${DEST}"
  ls -lh "${CKPT_SRC}"

  sz=$(stat -c%s "${CKPT_SRC}")
  if (( sz != EXPECTED_SIZE )); then
    echo "[WARN] size=${sz} != expected=${EXPECTED_SIZE}"
  fi

  rm -f "${DEST}"
  cp -f "${CKPT_SRC}" "${DEST}"
  ls -lh "${DEST}"

  /mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python - <<'PY' "${DEST}"
import sys, zipfile
p = sys.argv[1]
z = zipfile.ZipFile(p)
bad = z.testzip()
members = len(z.namelist())
print(f"[validate] members={members} bad={bad}")
if bad:
    raise SystemExit(1)
PY

  echo "[validate] sha256=$(sha256sum "${DEST}" | awk '{print $1}')"
  echo "========== $(date -Is) import ok =========="
} | tee -a "${LOG}"

if [[ "${RUN_SMOKE:-1}" == "1" ]]; then
  unset CKPT_PATH STARVLA_PYTHON 2>/dev/null || true
  bash "${ROOT}/scripts/setup/run_robotwin_smoke_vlact_pi.sh" | tee -a "${LOG}"
fi
