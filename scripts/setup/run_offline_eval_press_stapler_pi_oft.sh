#!/usr/bin/env bash
# Offline open-loop eval on press_stapler demo data for PI + OFT checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${VLACT_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
EVAL_PY="${ROOT}/examples/Robotwin/eval_files/offline_eval_robotwin.py"
REPORT="${ROOT}/logs/setup/PRESS_STAPLER_OFFLINE_EVAL_REPORT.md"
LOG_DIR="${ROOT}/logs/robotwin_offline"
mkdir -p "${LOG_DIR}"

PI_CKPT="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
OFT_CKPT="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3OFT_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
DATA_ROOT="${ROBOTWIN_DATA_ROOT:-/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin}"
STRIDE="${OFFLINE_STRIDE:-5}"
BATCH="${OFFLINE_BATCH_SIZE:-8}"
GPU="${GPU_ID:-7}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false

run_one() {
  local tag="$1" ckpt="$2" split="$3"
  local out="${LOG_DIR}/offline_${tag}_${split//\//_}.json"
  echo "[offline] ${tag} ${split}"
  "${PY}" "${EVAL_PY}" \
    --ckpt "${ckpt}" \
    --data_root "${DATA_ROOT}" \
    --data_name "${split}" \
    --robot_type robotwin_wrap32 \
    --stride "${STRIDE}" \
    --batch_size "${BATCH}" \
    --model_tag "${tag}" \
    --out "${out}"
}

{
  echo "# press_stapler 离线开环评测（LeRobot 演示轨迹）"
  echo ""
  echo "时间: $(date -Is)"
  echo ""
  echo "- 数据根目录: \`${DATA_ROOT}\`"
  echo "- stride=${STRIDE}, batch=${BATCH}, GPU=${GPU}"
  echo "- 指标: 归一化 MSE / 关节误差中位(°) / 窗口峰值中位(°) / ≥20°窗占比 / ≥50°窗占比 / 最大(°)"
  echo ""
} > "${REPORT}"

# 本机仅有 RoboTwin/press_stapler（50 ep 演示轨迹，开环测试集）
for split in "press_stapler"; do
  run_one "vlact_pi" "${PI_CKPT}" "${split}"
  run_one "vlact_oft" "${OFT_CKPT}" "${split}"
done

"${PY}" - <<'PY' "${LOG_DIR}" "${REPORT}"
import json, sys
from pathlib import Path

log_dir = Path(sys.argv[1])
report = Path(sys.argv[2])

rows = []
for path in sorted(log_dir.glob("offline_vlact_*.json")):
    data = json.loads(path.read_text())
    s = data["summary"]
    split = data["data_name"]
    rows.append((split, data["model_tag"], s))

append = ["", "## 结果汇总", "", "| 数据划分 | 模型 | 步数 | 归一化 MSE | 关节中位(°) | 窗口峰值中位(°) | ≥20°窗 | ≥50°窗 | 最大(°) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for split, tag, s in rows:
    append.append(
        f"| {split} | {tag} | {s['steps']} | {s['normalized_mse']:.6f} | "
        f"{s['joint_median_deg']:.2f} | {s['window_peak_median_deg']:.2f} | "
        f"{s['window_ge_20deg_frac']*100:.2f}% | {s['window_ge_50deg_frac']*100:.2f}% | "
        f"{s['max_joint_deg']:.2f} |"
    )
report.write_text(report.read_text() + "\n".join(append) + "\n")
print(report)
PY

echo "[offline] report=${REPORT}"
