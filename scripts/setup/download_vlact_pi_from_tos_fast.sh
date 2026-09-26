#!/usr/bin/env bash
# 高并行 tosutil 拉 VLAct PI（整目录 -r，大文件 multipart 多线程）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP="${ROOT}/scripts/setup"
EXT="${SETUP}/external_download"

TOS_SRC="${TOS_ROBOTWIN_DIR:-tos://c20250405/ljn/transfer/vlact/playground/Pretrained_models/VLAct-Qwen3VL4B-PI-Robotwin-All/VLAct-Qwen3VL4B-PI-Robotwin-All/}"
LOCAL_DIR="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune"
CKPT="${LOCAL_DIR}/checkpoints/steps_100000_pytorch_model.pt"
LOG="${ROOT}/logs/setup/download_vlact_pi_from_tos_fast.log"

# 并行参数（可通过环境变量覆盖）
TOS_J="${TOS_J:-32}"          # 文件级并发
TOS_NFJ="${TOS_NFJ:-32}"      # 非分片任务并发
TOS_P="${TOS_P:-64}"          # 单大文件分片下载并发
TOS_THRESHOLD="${TOS_THRESHOLD:-33554432}"  # 32MB 以上走 multipart

# 集群直连 TOS 易超时；默认走 proxy（下载后 zip 校验，损坏则重试）
export TOSUTIL_DISABLE_PROXY="${TOSUTIL_DISABLE_PROXY:-0}"

mkdir -p "${LOCAL_DIR}/checkpoints" "$(dirname "${LOG}")"
exec >>"${LOG}" 2>&1

echo "========== $(date -Is) download_vlact_pi_from_tos_fast start =========="
echo "TOS_SRC=${TOS_SRC} LOCAL_DIR=${LOCAL_DIR}"
echo "parallel: j=${TOS_J} nfj=${TOS_NFJ} p=${TOS_P} threshold=${TOS_THRESHOLD} noproxy=${TOSUTIL_DISABLE_PROXY}"

# shellcheck source=/dev/null
source "${EXT}/_tosutil_env.sh"
if [[ "${TOSUTIL_DISABLE_PROXY}" == "0" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY 2>/dev/null || true
fi

# 提高 tosutil 配置里的默认并行
sed -i \
  -e "s|^defaultParallels=.*|defaultParallels=${TOS_P}|" \
  -e "s|^defaultDownloadParallels=.*|defaultDownloadParallels=${TOS_P}|" \
  "${TOSUTIL_RUNTIME_CONF}"

EXPECTED_SIZE=11276441870

validate_ckpt() {
  [[ -f "$1" ]] || return 1
  (( $(stat -c%s "$1") == EXPECTED_SIZE )) || return 1
  /mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python - <<'PY' "$1"
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
bad = z.testzip()
if bad:
    raise SystemExit(f"zip corrupt at {bad}")
print(f"zip ok members={len(z.namelist())}")
PY
}

if validate_ckpt "${CKPT}" 2>/dev/null; then
  echo "[OK] already valid: $(ls -lh "${CKPT}")"
else
  rm -f "${CKPT}"
  attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "[$(date -Is)] tosutil cp -r attempt=${attempt} j=${TOS_J} nfj=${TOS_NFJ} p=${TOS_P}"
    if "${TOSUTIL}" cp "${TOS_SRC}" "${LOCAL_DIR}/" \
      -r -f -u -vchecksum \
      -j="${TOS_J}" -nfj="${TOS_NFJ}" -p="${TOS_P}" -threshold="${TOS_THRESHOLD}" \
      -tempFileDir=/tmp -conf="${TOSUTIL_RUNTIME_CONF}"; then
      if validate_ckpt "${CKPT}" 2>/dev/null; then
        echo "[OK] downloaded: $(ls -lh "${CKPT}")"
        echo "[validate] sha256=$(sha256sum "${CKPT}" | awk '{print $1}')"
        break
      fi
      echo "[WARN] download finished but zip invalid, retry"
      rm -f "${CKPT}"
    else
      echo "[WARN] tosutil failed, retry in 15s"
    fi
    sleep 15
  done
fi

echo "[$(date -Is)] starting smoke eval"
unset CKPT_PATH STARVLA_PYTHON 2>/dev/null || true
bash "${SETUP}/run_robotwin_smoke_vlact_pi.sh"
echo "========== $(date -Is) download_vlact_pi_from_tos_fast done =========="
