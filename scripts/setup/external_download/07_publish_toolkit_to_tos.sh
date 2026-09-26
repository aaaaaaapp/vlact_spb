#!/usr/bin/env bash
# 【集群】将 tosutil + 配置 + external_download 脚本发布到 TOS，供外部服务器拉取
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_tosutil_env.sh"

TOS_TOOLKIT="${TOS_BUCKET_PREFIX}/toolkit"
BUNDLE="/tmp/vlact_external_download_toolkit.tar.gz"

echo "=============================================="
echo " 发布 VLAct 外部下载工具包 -> TOS"
echo " TOS=${TOS_TOOLKIT}/"
echo "=============================================="

# 打包（含 tosutil 二进制 + 配置 + 全部脚本）
tar czf "${BUNDLE}" \
  -C /mnt/jfzn/liujn/code/Starvla/Starvla tosutil \
  -C /mnt/jfzn/liujn .tosutilconfig \
  -C "${SCRIPT_DIR}/.." external_download

echo "[pack] $(ls -lh "${BUNDLE}")"

tos_retry publish_tosutil \
  "${TOSUTIL}" cp /mnt/jfzn/liujn/code/Starvla/Starvla/tosutil "${TOS_TOOLKIT}/tosutil" \
  -conf="${TOSUTIL_RUNTIME_CONF}"

tos_retry publish_config \
  "${TOSUTIL}" cp /mnt/jfzn/liujn/.tosutilconfig "${TOS_TOOLKIT}/.tosutilconfig" \
  -conf="${TOSUTIL_RUNTIME_CONF}"

tos_retry publish_scripts \
  tos_upload_dir "${SCRIPT_DIR}" "${TOS_TOOLKIT}/external_download"

tos_retry publish_bundle \
  "${TOSUTIL}" cp "${BUNDLE}" "${TOS_TOOLKIT}/vlact_external_download_toolkit.tar.gz" \
  -conf="${TOSUTIL_RUNTIME_CONF}"

echo
echo "[DONE] 外部服务器拉取："
echo "  bash 00_bootstrap_from_tos.sh"
echo "  或手动: tosutil cp ${TOS_TOOLKIT}/vlact_external_download_toolkit.tar.gz ./"
