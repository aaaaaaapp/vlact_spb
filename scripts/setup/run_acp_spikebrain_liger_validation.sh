#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-spikebrain
model=/mnt/jfzn/liujn/code/Starvla/Starvla/spikebrain/SpikeBrain-2.0-VL

export http_proxy="${http_proxy:-http://10.119.18.5:3128}"
export https_proxy="${https_proxy:-http://10.119.18.5:3128}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export PATH="/mnt/jfzn/miniconda3/envs/starvla_qwen/bin:${PATH}"
export PYTHONPATH="${repo}:${repo}/.acp_deps:/mnt/jfzn/miniconda3/envs/starvla_qwen/lib/python3.11/site-packages"
export NO_ALBUMENTATIONS_UPDATE=1

run_id=${RUN_ID:-acp_vlact_spikebrain_liger_numerics_1gpu_v1}
runtime_cache=/tmp/vlact_spikebrain_kernel_cache_${run_id}
persistent_cache=${repo}/.kernel_cache
hf_cache=${repo}/.hf_modules_cache/${run_id}
hf_package=${hf_cache}/transformers_modules/SpikeBrain_hyphen_2_dot_0_hyphen_VL

cd "${repo}"
mkdir -p logs/setup "${runtime_cache}" "${hf_package}"
cp -a "${persistent_cache}/." "${runtime_cache}/" 2>/dev/null || true
mkdir -p "${hf_cache}/transformers_modules"
: > "${hf_cache}/transformers_modules/__init__.py"
: > "${hf_package}/__init__.py"
for python_file in "${model}"/*.py; do
  [[ "$(basename "${python_file}")" == "__init__.py" ]] && continue
  cp -f "${python_file}" "${hf_package}/"
done

export HF_MODULES_CACHE="${hf_cache}"
export XDG_CACHE_HOME="${runtime_cache}/xdg"
export TORCH_HOME="${runtime_cache}/torch"
export TRITON_CACHE_DIR="${runtime_cache}/triton"
export TORCHINDUCTOR_CACHE_DIR="${runtime_cache}/torchinductor"
export TORCH_EXTENSIONS_DIR="${runtime_cache}/torch_extensions"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

validation_args=(
  --sequence-length "${SEQUENCE_LENGTH:-64}"
  --output "logs/setup/${run_id}.json"
)
for setting in ROPE RMS_NORM SWIGLU FUSED_LINEAR_CROSS_ENTROPY; do
  env_name="LIGER_${setting}"
  option="--$(printf '%s' "${setting}" | tr '[:upper:]_' '[:lower:]-')"
  default=false
  [[ "${setting}" == "ROPE" ]] && default=true
  value="${!env_name:-${default}}"
  if [[ "${value,,}" == "true" ]]; then
    validation_args+=("${option}")
  else
    validation_args+=("--no-${option#--}")
  fi
done

python scripts/setup/validate_spikebrain_liger.py "${validation_args[@]}" \
  2>&1 | tee "logs/setup/${run_id}.log"

cp -a "${runtime_cache}/." "${persistent_cache}/" 2>/dev/null || \
  echo "[vlact_spikebrain] warning: failed to persist validation kernel cache" >&2
