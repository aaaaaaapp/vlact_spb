#!/usr/bin/env bash
# 【外部服务器】从 TOS 拉取 tosutil + 配置 + external_download 脚本（首次部署）
#
# 若本机尚无 tosutil，需先从集群 scp 一次，或安装火山引擎官方 tosutil 后配置 AK/SK。
# 已有 tosutil 时直接运行本脚本即可。
#
# 用法:
#   export TOSUTIL=~/bin/tosutil
#   export TOSUTIL_BASE_CONF=~/.tosutilconfig
#   bash 00_bootstrap_from_tos.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_tosutil_env.sh"

TOS_TOOLKIT="${TOS_BUCKET_PREFIX}/toolkit"
INSTALL_ROOT="${VLACT_TOOLKIT_ROOT:-${HOME}/vlact_toolkit}"
mkdir -p "${INSTALL_ROOT}"

echo "=============================================="
echo " 从 TOS 拉取 VLAct 外部下载工具包"
echo " TOS=${TOS_TOOLKIT}/"
echo " INSTALL_ROOT=${INSTALL_ROOT}"
echo "=============================================="

# 优先拉 tar 包（单文件，最快）
BUNDLE="${INSTALL_ROOT}/vlact_external_download_toolkit.tar.gz"
"${TOSUTIL}" cp "${TOS_TOOLKIT}/vlact_external_download_toolkit.tar.gz" "${BUNDLE}" \
  -u -conf="${TOSUTIL_RUNTIME_CONF}"

mkdir -p "${INSTALL_ROOT}/unpack"
tar xzf "${BUNDLE}" -C "${INSTALL_ROOT}/unpack"

# 布局: unpack/tosutil, unpack/.tosutilconfig, unpack/external_download/
install -m 755 "${INSTALL_ROOT}/unpack/tosutil" "${INSTALL_ROOT}/tosutil"
install -m 600 "${INSTALL_ROOT}/unpack/.tosutilconfig" "${INSTALL_ROOT}/.tosutilconfig"
rm -rf "${INSTALL_ROOT}/external_download"
mv "${INSTALL_ROOT}/unpack/external_download" "${INSTALL_ROOT}/external_download"
rm -rf "${INSTALL_ROOT}/unpack"

export TOSUTIL="${INSTALL_ROOT}/tosutil"
export TOSUTIL_BASE_CONF="${INSTALL_ROOT}/.tosutilconfig"

echo
echo "[DONE] 工具包已就绪:"
echo "  TOSUTIL=${TOSUTIL}"
echo "  TOSUTIL_BASE_CONF=${TOSUTIL_BASE_CONF}"
echo "  脚本目录=${INSTALL_ROOT}/external_download"
echo
echo "下一步:"
echo "  cd ${INSTALL_ROOT}/external_download"
echo "  source 00_env.sh"
echo "  bash 02_download_one.sh --list"
echo "  bash 01_download_robotwin_ckpt.sh"
echo "  bash 05_upload_to_tos.sh --only robotwin"
