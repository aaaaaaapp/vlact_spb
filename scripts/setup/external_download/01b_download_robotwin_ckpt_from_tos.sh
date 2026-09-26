#!/usr/bin/env bash
# 外部机：从 TOS 拉 VLAct PI 权重（必须不走 proxy），校验后供 rsync 到集群
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_env.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_tosutil_env.sh"

TOS_SRC="${TOS_ROBOTWIN_CKPT:-tos://c20250405/ljn/transfer/vlact/playground/Pretrained_models/VLAct-Qwen3VL4B-PI-Robotwin-All/VLAct-Qwen3VL4B-PI-Robotwin-All/checkpoints/steps_100000_pytorch_model.pt}"
TARGET="${STAGING_MODELS}/VLAct_Qwen3PI_Robotwin_all_Finetune"
CKPT="${TARGET}/checkpoints/steps_100000_pytorch_model.pt"
EXPECTED_SIZE=11276441870

export TOSUTIL_DISABLE_PROXY=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

validate_ckpt() {
  [[ -f "$1" ]] || return 1
  (( $(stat -c%s "$1") == EXPECTED_SIZE )) || return 1
  python3 - <<'PY' "$1"
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
bad = z.testzip()
if bad:
    raise SystemExit(f"zip corrupt at {bad}")
print(f"zip ok, members={len(z.namelist())}")
PY
}

mkdir -p "${TARGET}/checkpoints"

if validate_ckpt "${CKPT}" 2>/dev/null; then
  echo "[OK] 已存在且校验通过: $(ls -lh "${CKPT}")"
  sha256sum "${CKPT}"
  exit 0
fi

# 优先 TOS fuse 本地拷贝（若外部机已 mount）
FUSE_SRC="${TOS_FUSE_CKPT:-}"
if [[ -n "${FUSE_SRC}" && -f "${FUSE_SRC}" ]]; then
  echo "[fuse] cp ${FUSE_SRC} -> ${CKPT}"
  cp -f "${FUSE_SRC}" "${CKPT}"
else
  echo "[tosutil] cp noproxy ${TOS_SRC} -> ${CKPT}"
  rm -f "${CKPT}"
  "${TOSUTIL}" cp "${TOS_SRC}" "${CKPT}" -vchecksum -p=20 -tempFileDir=/tmp -conf="${TOSUTIL_RUNTIME_CONF}"
fi

validate_ckpt "${CKPT}"
echo "[OK] sha256=$(sha256sum "${CKPT}" | awk '{print $1}')"
echo
echo "同步到集群:"
echo "  bash ${SCRIPT_DIR}/04_sync_to_cluster.sh --execute --only robotwin"
