#!/usr/bin/env bash
# Run press_stapler full eval: demo_clean (Easy) then demo_randomized (Hard), 100 ep each.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL="${ROOT}/scripts/setup/run_robotwin_eval_press_stapler_vlact_pi.sh"
REPORT="${ROOT}/logs/setup/PRESS_STAPLER_EVAL_REPORT.md"
MASTER_LOG="${ROOT}/logs/setup/press_stapler_full_eval.log"

mkdir -p "$(dirname "${REPORT}")"
{
  echo "# press_stapler 完整评测（VLAct Qwen3PI Robotwin All）"
  echo ""
  echo "启动: $(date -Is)"
  echo ""
  echo "### 复现对照标准"
  echo "- **Per-task 官方参考**（同 data-scaling 训练设定，Qwen3OFT 头）:"
  echo "  - \`demo_clean\` (Easy): press_stapler **99.0%**"
  echo "  - \`demo_randomized\` (Hard): press_stapler **96.0%**"
  echo "- **本 checkpoint 为 PI 头**；HF README 注明无单独 PI per-task 公布值。"
  echo "- **整榜平均参考**（50 任务 Data-Scaling Clean）: VLAct OFT **92.5%**"
  echo ""
} > "${REPORT}"

exec >>"${MASTER_LOG}" 2>&1
echo "========== $(date -Is) press_stapler full eval start =========="

GPU_CLEAN="${GPU_CLEAN:-0}"
GPU_HARD="${GPU_HARD:-1}"
PORT_CLEAN="${PORT_CLEAN:-6794}"
PORT_HARD="${PORT_HARD:-6795}"

echo "[parallel] demo_clean gpu=${GPU_CLEAN} port=${PORT_CLEAN}"
TASK_CONFIG=demo_clean GPU_ID="${GPU_CLEAN}" PORT="${PORT_CLEAN}" bash "${EVAL}" &
PID_CLEAN=$!

echo "[parallel] demo_randomized gpu=${GPU_HARD} port=${PORT_HARD}"
TASK_CONFIG=demo_randomized GPU_ID="${GPU_HARD}" PORT="${PORT_HARD}" bash "${EVAL}" &
PID_HARD=$!

wait "${PID_CLEAN}"
CLEAN_RC=$?
wait "${PID_HARD}"
HARD_RC=$?

echo "========== $(date -Is) press_stapler full eval done (clean=${CLEAN_RC}, hard=${HARD_RC}) =========="
echo "Report: ${REPORT}"
(( CLEAN_RC == 0 && HARD_RC == 0 ))
