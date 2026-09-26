#!/usr/bin/env bash
# 下载 VLAct Qwen3PI RoboTwin 测评权重（~10.5GB）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_env.sh"

TARGET="${STAGING_MODELS}/VLAct_Qwen3PI_Robotwin_all_Finetune"
CKPT="${TARGET}/checkpoints/steps_100000_pytorch_model.pt"

echo "=============================================="
echo " VLAct RobotWin PI Finetune"
echo " 目标: ${TARGET}"
echo " 集群路径: ${CLUSTER_MODEL_ROOT}/VLAct_Qwen3PI_Robotwin_all_Finetune/"
echo "=============================================="

if [[ -f "${CKPT}" ]] && (( $(stat -c%s "${CKPT}") > 10000000000 )); then
  echo "[OK] 已存在: $(ls -lh "${CKPT}")"
  exit 0
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
bash "${SCRIPT_DIR}/_hf_download.sh" \
  vlact_robotwin_pi \
  "StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune" \
  "${TARGET}" \
  model

echo
echo "[DONE] 下载完成。同步到集群:"
echo "  rsync -avP ${TARGET}/ ${CLUSTER_HOST}:${CLUSTER_MODEL_ROOT}/VLAct_Qwen3PI_Robotwin_all_Finetune/"
echo "或运行: bash ${SCRIPT_DIR}/04_sync_to_cluster.sh --only robotwin"
