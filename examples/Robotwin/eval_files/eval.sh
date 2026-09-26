#!/usr/bin/env bash
# Official VLAct Robotwin eval (cluster paths). Usage:
#   bash eval.sh <task> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [port] [host]
set -euo pipefail

export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/jfzn/liujn/robotwin_env/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_FILES_PATH="${SCRIPT_DIR}"

if [[ $# -lt 6 ]]; then
  echo "Usage: bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [policy_port] [policy_host]" >&2
  exit 1
fi

policy_name="model2robotwin_interface"
task_name="$1"
task_config="$2"
ckpt_setting="${3:-starvla_demo}"
seed="${4:-0}"
gpu_id="${5:-0}"
policy_ckpt_path="$6"
policy_port="${7:-${ROBOTWIN_POLICY_PORT:-5694}}"
policy_host="${8:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${EVAL_FILES_PATH}/deploy_policy.yml}"

runtime_deploy_policy="$(mktemp "${TMPDIR:-/tmp}/robotwin_deploy_policy.XXXXXX.yml")"
cleanup() { rm -f "${runtime_deploy_policy}"; }
trap cleanup EXIT

sed \
  -e "s|^host:.*|host: \"${policy_host}\"|" \
  -e "s|^port:.*|port: ${policy_port}|" \
  -e "s|^unnorm_key:.*|unnorm_key: \"${ROBOTWIN_UNNORM_KEY:-new_embodiment}\"|" \
  "${deploy_policy_template}" > "${runtime_deploy_policy}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PATH="$(dirname "${ROBOTWIN_PYTHON}"):${PATH:-}"
export NINJA_PYTHON="${NINJA_PYTHON:-${ROBOTWIN_PYTHON}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export SAPIEN_NO_DISPLAY="${SAPIEN_NO_DISPLAY:-1}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

export PYTHONPATH="${ROBOTWIN_PATH}:${STARVLA_PATH}:${EVAL_FILES_PATH}:${PYTHONPATH:-}"

echo -e "\033[33m[eval.sh] sim gpu id: ${gpu_id}\033[0m"
echo "[eval.sh] STARVLA_PATH=${STARVLA_PATH}"
echo "[eval.sh] policy_ckpt_path=${policy_ckpt_path}"
echo "[eval.sh] policy_port=${policy_port}"
echo "[eval.sh] DEPLOY_POLICY=${runtime_deploy_policy}"
[[ -n "${ROBOTWIN_TEST_NUM:-}" ]] && echo "[eval.sh] ROBOTWIN_TEST_NUM=${ROBOTWIN_TEST_NUM}"

[[ -x "${ROBOTWIN_PYTHON}" ]] || { echo "[eval.sh] missing ${ROBOTWIN_PYTHON}" >&2; exit 1; }
[[ -d "${ROBOTWIN_PATH}" ]] || { echo "[eval.sh] missing ${ROBOTWIN_PATH}" >&2; exit 1; }
[[ -f "${policy_ckpt_path}" ]] || { echo "[eval.sh] missing ckpt ${policy_ckpt_path}" >&2; exit 1; }

cd "${ROBOTWIN_PATH}"

override_args=(
  --task_name "${task_name}"
  --task_config "${task_config}"
  --ckpt_setting "${ckpt_setting}"
  --seed "${seed}"
  --policy_name "${policy_name}"
)
[[ -n "${ROBOTWIN_TEST_NUM:-}" ]] && override_args+=(--test_num "${ROBOTWIN_TEST_NUM}")

PYTHONWARNINGS=ignore::UserWarning \
"${ROBOTWIN_PYTHON}" script/eval_policy.py --config "${runtime_deploy_policy}" \
  --policy_ckpt_path "${policy_ckpt_path}" \
  --overrides \
  "${override_args[@]}"
