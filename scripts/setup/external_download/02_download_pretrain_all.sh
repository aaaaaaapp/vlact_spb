#!/usr/bin/env bash
# 下载 VLAct 预训练所需资源（按 datasets.conf.sh 清单，含部分下载 include）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo " VLAct 预训练资源（部分数据集仅下所需子目录）"
echo " 清单: bash 02_download_one.sh --list"
echo "=============================================="

# 按组顺序下载，避免一次拉起过多任务
bash "${SCRIPT_DIR}/02_download_one.sh" --group models
bash "${SCRIPT_DIR}/02_download_one.sh" --group vlm
bash "${SCRIPT_DIR}/02_download_one.sh" --group vla_core
bash "${SCRIPT_DIR}/02_download_one.sh" --group robocoin

echo
echo "[NOTE] InternData-A1 需单独: bash 03_download_internedata.sh"
echo "[NOTE] DROID_100 若已下 RoboInter，集群 postprocess 可从标注派生，可不单独下 droid_100"
echo "[DONE] 同步/TOS: bash 05_upload_to_tos.sh 或 06_download_from_tos.sh"
