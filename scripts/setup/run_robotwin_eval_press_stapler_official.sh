#!/usr/bin/env bash
# press_stapler eval via official VLAct client + server_policy.py (client-side unnorm).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_SH="${ROOT}/examples/Robotwin/eval_files/eval.sh"
VLACT_PYTHON="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python"
ROBOTWIN_PYTHON="/mnt/jfzn/liujn/robotwin_env/bin/python"
ROBOTWIN_PATH="/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval"

TASK="${TASK:-press_stapler}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6800}"
TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"
CKPT_SETTING="${CKPT_SETTING:-vlact_official}"
MODEL_TAG="${MODEL_TAG:-model}"
CKPT_PATH="${CKPT_PATH:?CKPT_PATH required}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/robotwin_eval"
mkdir -p "${LOG_DIR}"
EVAL_LOG="${LOG_DIR}/robotwin_official_${TASK}_${TASK_CONFIG}_${MODEL_TAG}_${TS}.log"
SERVER_LOG="${LOG_DIR}/policy_server_official_${MODEL_TAG}_${PORT}_${TS}.log"

export ROBOTWIN_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-new_embodiment}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cleanup() {
  [[ -n "${SERVER_PID:-}" ]] && kill "${SERVER_PID}" 2>/dev/null || true
  [[ -n "${SERVER_PID:-}" ]] && wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[official] task=${TASK} config=${TASK_CONFIG} model=${MODEL_TAG} episodes=${TEST_NUM} gpu=${GPU_ID} port=${PORT}"
echo "[official] ckpt=${CKPT_PATH}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 \
  "${VLACT_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --use_bf16 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
for ((t=0; t<600; t++)); do
  if grep -q "server running" "${SERVER_LOG}" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[official] server exited early; see ${SERVER_LOG}" >&2
    tail -40 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
(( ready == 1 )) || { echo "[official] server not ready"; exit 1; }
echo "[official] server ready pid=${SERVER_PID}"

ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}" \
ROBOTWIN_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY}" \
ROBOTWIN_TEST_NUM="${TEST_NUM}" \
bash "${EVAL_SH}" \
  "${TASK}" "${TASK_CONFIG}" "${CKPT_SETTING}" "${SEED}" "${GPU_ID}" \
  "${CKPT_PATH}" "${PORT}" \
  2>&1 | tee "${EVAL_LOG}"

FINAL_LINE="$(sed 's/\x1b\[[0-9;]*m//g' "${EVAL_LOG}" | grep -E 'Success rate:' | tail -1 || true)"
echo "[official] done ${FINAL_LINE}"
echo "[official] log=${EVAL_LOG}"
