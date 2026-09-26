#!/usr/bin/env bash
# Full RoboTwin eval: press_stapler × 100 episodes (demo_clean / demo_randomized).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-${ROOT}}"
VLACT_PYTHON="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python"
VLACT_PI_CKPT="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
STARVLA_PYTHON="${STARVLA_PYTHON:-${VLACT_PYTHON}}"
[[ "${STARVLA_PYTHON}" == *"/envs/vlact/"* ]] || STARVLA_PYTHON="${VLACT_PYTHON}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/jfzn/liujn/robotwin_env/bin/python}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval}"
CKPT_PATH="${CKPT_PATH:-${VLACT_PI_CKPT}}"

TASK="${TASK:-press_stapler}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6794}"
TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"
CKPT_SETTING="${CKPT_SETTING:-vlact_pi_full}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/robotwin_eval"
REPORT="${ROOT}/logs/setup/PRESS_STAPLER_EVAL_REPORT.md"
mkdir -p "${LOG_DIR}"

EVAL_LOG="${LOG_DIR}/robotwin_full_${TASK}_${TASK_CONFIG}_${TS}.log"
SERVER_LOG="${LOG_DIR}/policy_server_vlact_pi_${PORT}_${TS}.log"

export STARVLA_ENABLE_CUROBO="${STARVLA_ENABLE_CUROBO:-1}"
export ROBOTWIN_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-new_embodiment}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[eval] checkpoint missing: ${CKPT_PATH}" >&2
  exit 2
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[eval] task=${TASK} config=${TASK_CONFIG} episodes=${TEST_NUM} gpu=${GPU_ID} port=${PORT}"
echo "[eval] ckpt=${CKPT_PATH}"
echo "[eval] log=${EVAL_LOG}"

cd "${STARVLA_ROOT}"
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"

POLICY_SERVER="${POLICY_SERVER:-deployment/model_server/server_policy_starvla_compat.py}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
  "${STARVLA_PYTHON}" "${POLICY_SERVER}" \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --use_bf16 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
for ((t=0; t<600; t++)); do
  if grep -q "server running" "${SERVER_LOG}" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval] policy server exited early; see ${SERVER_LOG}" >&2
    tail -30 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
(( ready == 1 )) || { echo "[eval] server not ready"; exit 1; }

echo "[eval] policy server ready (pid=${SERVER_PID})"

STARVLA_PATH="${STARVLA_ROOT}" \
ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}" \
ROBOTWIN_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY}" \
ROBOTWIN_TEST_NUM="${TEST_NUM}" \
STARVLA_ENABLE_CUROBO="${STARVLA_ENABLE_CUROBO}" \
ROBOTWIN_POLICY_PORT="${PORT}" \
bash /mnt/jfzn/liujn/code/Starvla/Starvla/examples/Robotwin/eval_files/eval.sh \
  "${TASK}" "${TASK_CONFIG}" "${CKPT_SETTING}" "${SEED}" "${GPU_ID}" "${CKPT_PATH}" "${PORT}" \
  2>&1 | tee "${EVAL_LOG}"

# Parse final success rate from log (strip ANSI).
FINAL_LINE="$(sed 's/\x1b\[[0-9;]*m//g' "${EVAL_LOG}" | grep -E 'Success rate:' | tail -1 || true)"
echo "[eval] done. ${FINAL_LINE}"
echo "[eval] log=${EVAL_LOG} server_log=${SERVER_LOG}"

# Append to report file.
{
  echo ""
  echo "## $(date -Is) ${TASK} / ${TASK_CONFIG}"
  echo "- episodes: ${TEST_NUM}"
  echo "- log: \`${EVAL_LOG}\`"
  echo "- result: ${FINAL_LINE:-N/A}"
  echo ""
  echo "### 官方参考（VLAct README · data-scaling · Qwen3OFT 同数据设定）"
  echo "| 模式 | 任务 | 官方 Qwen3OFT | 本次 VLAct PI |"
  echo "|------|------|---------------|---------------|"
  if [[ "${TASK_CONFIG}" == "demo_clean" ]]; then
    echo "| Easy (demo_clean) | press_stapler | **99.0%** | ${FINAL_LINE:-pending} |"
  else
    echo "| Hard (demo_randomized) | press_stapler | **96.0%** | ${FINAL_LINE:-pending} |"
  fi
  echo ""
  echo "> PI checkpoint 无单独 per-task 官方数；对照 OFT 同任务参考值。全 50 任务 Data-Scaling 平均：VLAct OFT Clean **92.5%**。"
} >> "${REPORT}"
