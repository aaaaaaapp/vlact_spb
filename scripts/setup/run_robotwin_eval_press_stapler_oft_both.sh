#!/usr/bin/env bash
# press_stapler full eval: VLAct Qwen3OFT Robotwin All (Easy + Hard parallel).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL="${ROOT}/scripts/setup/run_robotwin_eval_press_stapler_vlact_oft.sh"
REPORT="${ROOT}/logs/setup/PRESS_STAPLER_OFT_VS_PI_REPORT.md"
MASTER_LOG="${ROOT}/logs/setup/press_stapler_oft_eval.log"

mkdir -p "$(dirname "${REPORT}")"
{
  echo "# press_stapler OFT vs PI 对比评测"
  echo ""
  echo "启动: $(date -Is)"
  echo ""
  echo "### 官方参考（Qwen3OFT · data-scaling）"
  echo "- Easy (\`demo_clean\`): **99.0%**"
  echo "- Hard (\`demo_randomized\`): **96.0%**"
  echo ""
  echo "### 已有 PI 结果（100 ep）"
  echo "- Easy: **89/100 = 89.0%**"
  echo "- Hard: **88/100 = 88.0%**"
  echo ""
} > "${REPORT}"

exec >>"${MASTER_LOG}" 2>&1
echo "========== $(date -Is) press_stapler OFT eval start =========="

GPU_CLEAN="${GPU_CLEAN:-5}"
GPU_HARD="${GPU_HARD:-6}"
PORT_CLEAN="${PORT_CLEAN:-6796}"
PORT_HARD="${PORT_HARD:-6797}"

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

echo "========== $(date -Is) press_stapler OFT eval done (clean=${CLEAN_RC}, hard=${HARD_RC}) =========="

# Append comparison summary.
CLEAN_LINE="$(grep 'Success rate' "${ROOT}/logs/robotwin_eval"/robotwin_full_press_stapler_demo_clean_oft_*.log 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -1 || true)"
HARD_LINE="$(grep 'Success rate' "${ROOT}/logs/robotwin_eval"/robotwin_full_press_stapler_demo_randomized_oft_*.log 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -1 || true)"

{
  echo ""
  echo "## 对比汇总"
  echo "| 模式 | 官方 OFT | VLAct PI (100ep) | VLAct OFT (100ep) |"
  echo "|------|----------|------------------|-------------------|"
  echo "| Easy | 99.0% | 89.0% | ${CLEAN_LINE:-pending} |"
  echo "| Hard | 96.0% | 88.0% | ${HARD_LINE:-pending} |"
} >> "${REPORT}"

echo "Report: ${REPORT}"
(( CLEAN_RC == 0 && HARD_RC == 0 ))
