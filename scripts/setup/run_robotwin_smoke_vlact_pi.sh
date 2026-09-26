#!/usr/bin/env bash
# Smoke test: VLAct Qwen3PI RoboTwin checkpoint on press_stapler (3 episodes).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-${ROOT}}"
# 避免 shell 里残留的 StarVLA eval 环境变量（CKPT_PATH / STARVLA_PYTHON）覆盖 VLAct PI
VLACT_PYTHON="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python"
VLACT_PI_CKPT="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
STARVLA_PYTHON="${STARVLA_PYTHON:-${VLACT_PYTHON}}"
[[ "${STARVLA_PYTHON}" == *"/envs/vlact/"* ]] || STARVLA_PYTHON="${VLACT_PYTHON}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/jfzn/liujn/robotwin_env/bin/python}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval}"
CKPT_PATH="${CKPT_PATH:-${VLACT_PI_CKPT}}"
[[ "${CKPT_PATH}" == *"VLAct_Qwen3PI_Robotwin_all_Finetune"* ]] || CKPT_PATH="${VLACT_PI_CKPT}"

TASK="${TASK:-press_stapler}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6794}"
TEST_NUM="${ROBOTWIN_TEST_NUM:-3}"
CKPT_SETTING="${CKPT_SETTING:-vlact_pi_smoke}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/robotwin_eval"
mkdir -p "${LOG_DIR}"

EVAL_LOG="${LOG_DIR}/robotwin_smoke_${TASK}_${TS}.log"
SERVER_LOG="${LOG_DIR}/policy_server_vlact_pi_${PORT}_${TS}.log"

export STARVLA_ENABLE_CUROBO="${STARVLA_ENABLE_CUROBO:-1}"
export ROBOTWIN_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-new_embodiment}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[smoke] checkpoint missing: ${CKPT_PATH}" >&2
  exit 2
fi

if [[ ! -x "${STARVLA_PYTHON}" ]]; then
  echo "[smoke] starVLA python missing: ${STARVLA_PYTHON}" >&2
  exit 2
fi

if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
  echo "[smoke] robotwin python missing: ${ROBOTWIN_PYTHON}" >&2
  exit 2
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[smoke] ckpt=${CKPT_PATH}"
echo "[smoke] task=${TASK} test_num=${TEST_NUM} port=${PORT} gpu=${GPU_ID}"
echo "[smoke] eval_log=${EVAL_LOG}"

cd "${STARVLA_ROOT}"
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
# base_vlm / Qwen3-0.6B expert 均已本地化，优先离线加载避免 hf-mirror 超时
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# 若仍需联网拉 HF，可: TRANSFORMERS_OFFLINE=0 bash ...（会 source proxy.env）
if [[ "${TRANSFORMERS_OFFLINE}" == "0" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/setup/proxy.env"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

# Starvla RobotWin eval client 需要 action_chunk_size 等 metadata；
# 用独立 server 入口，不改动默认 server_policy.py（LIBERO/SimplerEnv 等仍走原版）。
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
    echo "[smoke] policy server exited early; see ${SERVER_LOG}" >&2
    tail -30 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done

if (( ready == 0 )); then
  echo "[smoke] policy server not ready after 20min; see ${SERVER_LOG}" >&2
  tail -30 "${SERVER_LOG}" >&2 || true
  exit 1
fi

echo "[smoke] policy server ready (pid=${SERVER_PID})"

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

echo "[smoke] done. eval_log=${EVAL_LOG} server_log=${SERVER_LOG}"
