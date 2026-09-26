#!/usr/bin/env bash
# 【集群 node】从 TOS 下载到本地路径（断点续传 + 自动 retry）
#
# 用法:
#   bash 06_download_from_tos.sh --only robotwin
#   bash 06_download_from_tos.sh --only pretrain
#   bash 06_download_from_tos.sh --only all
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_env.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_tosutil_env.sh"

# 集群目标路径（覆盖 00_env.sh 的 staging，指向真实挂载）
export CLUSTER_VLACT_ROOT="${CLUSTER_VLACT_ROOT:-/mnt/jfzn/liujn/code/VLAct-main}"
export CLUSTER_MODEL_ROOT="${CLUSTER_MODEL_ROOT:-/mnt/jfzn/liujn/models/StarVLA}"

ONLY="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="${2:-all}"; shift 2 ;;
    --only=*) ONLY="${1#*=}"; shift ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

echo "=============================================="
echo " TOS -> 集群"
echo " TOS_PREFIX=${TOS_BUCKET_PREFIX}"
echo " ONLY=${ONLY}"
echo " 日志: ${TOS_STATE_DIR}/"
echo "=============================================="

download_dir() {
  local name="$1"
  local tos_path="$2"
  local local_path="$3"
  mkdir -p "${local_path}"
  tos_retry "${name}" tos_download_dir "${tos_path}/" "${local_path}/"
}

validate_robotwin_ckpt() {
  local ckpt="${CLUSTER_MODEL_ROOT}/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
  [[ -f "${ckpt}" ]] && (( $(stat -c%s "${ckpt}") > 10000000000 ))
}

case "${ONLY}" in
  robotwin)
    rm -f "${TOS_STATE_DIR}/robotwin_download.done"
    while ! validate_robotwin_ckpt; do
      download_dir robotwin_download \
        "${TOS_BUCKET_PREFIX}/models/VLAct_Qwen3PI_Robotwin_all_Finetune" \
        "${CLUSTER_MODEL_ROOT}/VLAct_Qwen3PI_Robotwin_all_Finetune"
      rm -f "${TOS_STATE_DIR}/robotwin_download.done"
      if validate_robotwin_ckpt; then
        break
      fi
      echo "[WARN] TOS 上权重未就绪或体积不足，30s 后重试..." >&2
      sleep 30
    done
    ;;
  pretrain)
    download_dir pretrain_models_download \
      "${TOS_BUCKET_PREFIX}/playground/Pretrained_models" \
      "${CLUSTER_VLACT_ROOT}/playground/Pretrained_models"
    download_dir pretrain_datasets_download \
      "${TOS_BUCKET_PREFIX}/playground/Datasets" \
      "${CLUSTER_VLACT_ROOT}/playground/Datasets"
    ;;
  all)
    rm -f "${TOS_STATE_DIR}/robotwin_download.done"
    download_dir robotwin_download \
      "${TOS_BUCKET_PREFIX}/models/VLAct_Qwen3PI_Robotwin_all_Finetune" \
      "${CLUSTER_MODEL_ROOT}/VLAct_Qwen3PI_Robotwin_all_Finetune"
    rm -f "${TOS_STATE_DIR}/robotwin_download.done"
    download_dir pretrain_models_download \
      "${TOS_BUCKET_PREFIX}/playground/Pretrained_models" \
      "${CLUSTER_VLACT_ROOT}/playground/Pretrained_models"
    download_dir pretrain_datasets_download \
      "${TOS_BUCKET_PREFIX}/playground/Datasets" \
      "${CLUSTER_VLACT_ROOT}/playground/Datasets"
    ;;
  *)
    echo "未知 --only: ${ONLY}（robotwin / pretrain / all）" >&2
    exit 1
    ;;
esac

echo
CKPT="${CLUSTER_MODEL_ROOT}/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
if [[ -f "${CKPT}" ]]; then
  echo "[OK] checkpoint: $(ls -lh "${CKPT}")"
  echo "运行 smoke:"
  echo "  bash ${CLUSTER_VLACT_ROOT}/scripts/setup/run_robotwin_smoke_vlact_pi.sh"
else
  echo "[INFO] checkpoint 尚未就绪: ${CKPT}"
fi
