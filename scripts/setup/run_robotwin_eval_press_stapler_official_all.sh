#!/usr/bin/env bash
# Official VLAct eval: PI + OFT, Easy + Hard (100 ep each), sequential by model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${ROOT}/scripts/setup/run_robotwin_eval_press_stapler_official.sh"
REPORT="${ROOT}/logs/setup/PRESS_STAPLER_OFFICIAL_EVAL_REPORT.md"
MASTER_LOG="${ROOT}/logs/setup/press_stapler_official_eval.log"

PI_CKPT="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
OFT_CKPT="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3OFT_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"

mkdir -p "$(dirname "${REPORT}")"
{
  echo "# press_stapler 官方链路评测（VLAct client + server_policy.py）"
  echo ""
  echo "启动: $(date -Is)"
  echo ""
  echo "| 模式 | 官方 OFT 参考 | 旧 compat PI | 旧 compat OFT |"
  echo "|------|--------------|--------------|---------------|"
  echo "| Easy | 99.0% | 89.0% | (compat 进行中/见日志) |"
  echo "| Hard | 96.0% | 88.0% | (compat 进行中/见日志) |"
  echo ""
} > "${REPORT}"

exec >>"${MASTER_LOG}" 2>&1
echo "========== $(date -Is) official eval all start =========="

run_pair() {
  local tag="$1" ckpt="$2" gpu_clean="$3" gpu_hard="$4" port_clean="$5" port_hard="$6"
  echo "[pair] ${tag} easy gpu=${gpu_clean} port=${port_clean}"
  MODEL_TAG="${tag}" CKPT_PATH="${ckpt}" TASK_CONFIG=demo_clean \
    GPU_ID="${gpu_clean}" PORT="${port_clean}" CKPT_SETTING="${tag}_official_clean" \
    bash "${RUN}" &
  local pid_clean=$!
  echo "[pair] ${tag} hard gpu=${gpu_hard} port=${port_hard}"
  MODEL_TAG="${tag}" CKPT_PATH="${ckpt}" TASK_CONFIG=demo_randomized \
    GPU_ID="${gpu_hard}" PORT="${port_hard}" CKPT_SETTING="${tag}_official_hard" \
    bash "${RUN}" &
  local pid_hard=$!
  wait "${pid_clean}"; local rc_clean=$?
  wait "${pid_hard}"; local rc_hard=$?
  echo "[pair] ${tag} done clean=${rc_clean} hard=${rc_hard}"
  (( rc_clean == 0 && rc_hard == 0 ))
}

GPU_CLEAN="${GPU_CLEAN:-5}"
GPU_HARD="${GPU_HARD:-6}"
PORT_PI_CLEAN="${PORT_PI_CLEAN:-6800}"
PORT_PI_HARD="${PORT_PI_HARD:-6801}"
PORT_OFT_CLEAN="${PORT_OFT_CLEAN:-6802}"
PORT_OFT_HARD="${PORT_OFT_HARD:-6803}"

run_pair "vlact_pi" "${PI_CKPT}" "${GPU_CLEAN}" "${GPU_HARD}" "${PORT_PI_CLEAN}" "${PORT_PI_HARD}"
run_pair "vlact_oft" "${OFT_CKPT}" "${GPU_CLEAN}" "${GPU_HARD}" "${PORT_OFT_CLEAN}" "${PORT_OFT_HARD}"

append_result() {
  local tag="$1" mode="$2"
  local pattern="robotwin_official_press_stapler_${mode}_${tag}_"
  local line
  line="$(grep -h 'Success rate' "${ROOT}/logs/robotwin_eval/${pattern}"*.log 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tail -1 || true)"
  echo "- ${tag} / ${mode}: ${line:-N/A}"
}

{
  echo ""
  echo "## 结果 $(date -Is)"
  append_result "vlact_pi" "demo_clean"
  append_result "vlact_pi" "demo_randomized"
  append_result "vlact_oft" "demo_clean"
  append_result "vlact_oft" "demo_randomized"
} >> "${REPORT}"

echo "========== $(date -Is) official eval all done =========="
echo "Report: ${REPORT}"
