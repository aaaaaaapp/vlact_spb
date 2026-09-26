#!/usr/bin/env bash
# 【外部服务器】将 STAGING_DIR 上传到 TOS（需 tosutil + .tosutilconfig）
#
# 用法:
#   source 00_env.sh
#   bash 01_download_robotwin_ckpt.sh          # 先在本地下好
#   bash 05_upload_to_tos.sh --only robotwin   # 上传到 TOS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_env.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_tosutil_env.sh"

ONLY="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="${2:-all}"; shift 2 ;;
    --only=*) ONLY="${1#*=}"; shift ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

echo "=============================================="
echo " 上传 STAGING_DIR -> TOS"
echo " STAGING_DIR=${STAGING_DIR}"
echo " TOS_PREFIX=${TOS_BUCKET_PREFIX}"
echo " ONLY=${ONLY}"
echo " 日志: ${TOS_STATE_DIR}/"
echo "=============================================="

upload_dir() {
  local name="$1"
  local local_path="$2"
  local tos_path="$3"
  [[ -d "${local_path}" ]] || { echo "[SKIP] 本地目录不存在: ${local_path}"; return 0; }
  tos_retry "${name}" tos_upload_dir "${local_path}/" "${tos_path}/"
}

case "${ONLY}" in
  robotwin)
    upload_dir robotwin_upload \
      "${STAGING_MODELS}/VLAct_Qwen3PI_Robotwin_all_Finetune" \
      "${TOS_BUCKET_PREFIX}/models/VLAct_Qwen3PI_Robotwin_all_Finetune"
    ;;
  pretrain)
    upload_dir pretrain_models_upload \
      "${STAGING_PRETRAINED}" \
      "${TOS_BUCKET_PREFIX}/playground/Pretrained_models"
    upload_dir pretrain_datasets_upload \
      "${STAGING_DATASETS}" \
      "${TOS_BUCKET_PREFIX}/playground/Datasets"
    ;;
  all)
    upload_dir robotwin_upload \
      "${STAGING_MODELS}/VLAct_Qwen3PI_Robotwin_all_Finetune" \
      "${TOS_BUCKET_PREFIX}/models/VLAct_Qwen3PI_Robotwin_all_Finetune"
    upload_dir pretrain_models_upload \
      "${STAGING_PRETRAINED}" \
      "${TOS_BUCKET_PREFIX}/playground/Pretrained_models"
    upload_dir pretrain_datasets_upload \
      "${STAGING_DATASETS}" \
      "${TOS_BUCKET_PREFIX}/playground/Datasets"
    ;;
  *)
    echo "未知 --only: ${ONLY}（robotwin / pretrain / all）" >&2
    exit 1
    ;;
esac

echo
echo "[DONE] 上传完成。在集群执行:"
echo "  bash ${SCRIPT_DIR}/06_download_from_tos.sh --only ${ONLY}"
