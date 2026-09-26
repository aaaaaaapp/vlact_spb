#!/usr/bin/env bash
# 打包「外部下载脚本」，传到另一台有公网的服务器上使用
# 用法: bash pack_for_external_server.sh
# 输出: /tmp/vlact_external_download_scripts.tar.gz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="/tmp/vlact_external_download_scripts.tar.gz"

tar czf "${OUT}" \
  -C "${SCRIPT_DIR}/.." \
  external_download

echo "[OK] 已打包: ${OUT} ($(ls -lh "${OUT}" | awk '{print $5}'))"
echo
echo "传到外部下载服务器（任选一种）:"
echo
echo "  # 1) scp（最直接）"
echo "  scp ${OUT} USER@EXTERNAL_HOST:~/"
echo "  ssh USER@EXTERNAL_HOST 'mkdir -p ~/vlact_external_download && tar xzf ~/vlact_external_download_scripts.tar.gz -C ~/vlact_external_download'"
echo "  # 脚本在 ~/vlact_external_download/external_download/"
echo
echo "  # 2) 先上传到 TOS，外部服务器再拉"
echo "  bash ${SCRIPT_DIR}/08_upload_scripts_to_tos.sh"
echo "  # 外部机: tosutil cp tos://c20250405/ljn/transfer/vlact/toolkit/vlact_external_download_scripts.tar.gz ./"
