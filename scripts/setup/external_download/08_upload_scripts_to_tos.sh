#!/usr/bin/env bash
# 将下载脚本包上传到 TOS（供外部服务器拉取）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_tosutil_env.sh"

bash "${SCRIPT_DIR}/pack_for_external_server.sh"

TOS_PATH="${TOS_BUCKET_PREFIX}/toolkit/vlact_external_download_scripts.tar.gz"
LOCAL="/tmp/vlact_external_download_scripts.tar.gz"

echo ">>> 上传 ${LOCAL} -> ${TOS_PATH}"
tos_retry upload_scripts \
  "${TOSUTIL}" cp "${LOCAL}" "${TOS_PATH}" -u -conf="${TOSUTIL_RUNTIME_CONF}"

echo
echo "[DONE] 外部服务器执行:"
echo "  mkdir -p ~/vlact_external_download && cd ~/vlact_external_download"
echo "  tosutil cp ${TOS_PATH} ./vlact_external_download_scripts.tar.gz -conf=~/.tosutilconfig"
echo "  tar xzf vlact_external_download_scripts.tar.gz"
echo "  cd external_download && source 00_env.sh && bash 02_download_one.sh --list"
