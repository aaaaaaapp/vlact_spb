#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${STARVLA_ROOT}"

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600  # timeout set to 1 hour (unit: seconds)
export DEEPSPEED_TIMEOUT=720  # collective wait timeout, unit: minutes (rank0 cold-cache filtering takes ~40min)

# ACP omits libGL, so prefer the repository-local headless OpenCV package.
local_dependency_dir="${STARVLA_ROOT}/.acp_deps"
if [[ -d "${local_dependency_dir}" ]]; then
  export PYTHONPATH="${local_dependency_dir}:${PYTHONPATH:-}"
fi

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenHybrid_xrobot_padding
freeze_module_list=qwen_vl_interface.model.visual
for layer_idx in $(seq 0 17); do
  freeze_module_list+=" ,qwen_vl_interface.model.language_model.layers.${layer_idx}"
done
freeze_module_list=${freeze_module_list// ,/,}
base_vlm=${BASE_VLM:-/mnt/jfzn/liujn/code/Starvla/Starvla/spikebrain/SpikeBrain-2.0-VL}
action_expert_model=${ACTION_EXPERT_MODEL:-/mnt/jfzn/liujn/code/VLAct-main/playground/Pretrained_models/Qwen3-0.6B}
config_yaml=./examples/InternA1/train_files/starvla_cotrain_agilex_franka_50_padding.yaml
data_root_dir=${DATA_ROOT_DIR:-/mnt/jfzn/liujn/code/VLAct-main/playground/Datasets}
export STARVLA_DATA_ROOT="${data_root_dir}"
run_root_dir=./results/Checkpoints
data_mix=agilex_franka_5data_manualvel_balance_50
vlm_coco_data=${VLM_COCO_DATA:-${data_root_dir}/LLaVA-OneVision-COCO/llava_jsons/sharegpt4v_coco.json::${data_root_dir}/LLaVA-OneVision-COCO/images}
vlm_cc3m_data=${VLM_CC3M_DATA:-${data_root_dir}/LLaVA-ReCap-CC3M/data}
vlm_coco_repeat=5
vlm_data=${vlm_coco_data}
for _ in $(seq 2 ${vlm_coco_repeat}); do
  vlm_data+=,${vlm_coco_data}
done
vlm_data+=,${vlm_cc3m_data}
heads=oft,gr00t,pi
head_loss_weights=oft:1,gr00t:1,pi:1
run_id=${RUN_ID:-vlact_spikebrain_pretrain_bf16}

# Stage persistent kernels onto node-local storage. Runtime cache lookups stay
# local while compiled artifacts remain reusable by later ACP jobs.
persistent_kernel_cache=${KERNEL_CACHE_PERSIST_ROOT:-${STARVLA_ROOT}/.kernel_cache}
runtime_kernel_cache=${KERNEL_CACHE_RUNTIME_ROOT:-/tmp/vlact_spikebrain_kernel_cache_${run_id}}
mkdir -p "${persistent_kernel_cache}" "${runtime_kernel_cache}"
if [[ "${KERNEL_CACHE_PRELOAD:-true}" == "true" ]]; then
  cp -a "${persistent_kernel_cache}/." "${runtime_kernel_cache}/" 2>/dev/null || true
fi
export XDG_CACHE_HOME="${runtime_kernel_cache}/xdg"
export TORCH_HOME="${runtime_kernel_cache}/torch"
export TRITON_CACHE_DIR="${runtime_kernel_cache}/triton"
export TORCHINDUCTOR_CACHE_DIR="${runtime_kernel_cache}/torchinductor"
export TORCH_EXTENSIONS_DIR="${runtime_kernel_cache}/torch_extensions"
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}"

# Transformers' dynamic-module loader does not discover SpikeBrain's nested
# relative import of ulysses.py. Mirror StarVLA's integration by pre-populating
# an isolated cache with the checkpoint's complete Python implementation.
setup_spikebrain_hf_modules_cache() {
  local cache_root="${HF_MODULES_CACHE:-${STARVLA_ROOT}/.hf_modules_cache/${run_id}}"
  local package="transformers_modules/SpikeBrain_hyphen_2_dot_0_hyphen_VL"
  local package_dir="${cache_root}/${package}"

  mkdir -p "${cache_root}/transformers_modules" "${package_dir}"
  : > "${cache_root}/transformers_modules/__init__.py"
  : > "${package_dir}/__init__.py"
  for python_file in "${base_vlm}"/*.py; do
    [[ "$(basename "${python_file}")" == "__init__.py" ]] && continue
    cp -f "${python_file}" "${package_dir}/"
  done

  export HF_MODULES_CACHE="${cache_root}"
  test -f "${package_dir}/ulysses.py"
  echo "[vlact_spikebrain] HF_MODULES_CACHE=${HF_MODULES_CACHE}"
}

setup_spikebrain_hf_modules_cache
# Native 8-GPU single-node defaults: VLA global batch 128 and VLM global batch 16.
# Override these only for a deliberate scaling study.
# Env overrides: DATA_ROOT_DIR / VLA_BS / VLM_BS / GRAD_ACCUM / NUM_PROCESSES / RUN_ID / MAIN_PROCESS_PORT / MAX_TRAIN_STEPS
vla_per_device_batch_size=${VLA_BS:-16}
vlm_per_device_batch_size=${VLM_BS:-2}
gradient_accumulation_steps=${GRAD_ACCUM:-1}
num_processes=${NUM_PROCESSES:-8}
max_train_steps=${MAX_TRAIN_STEPS:-100000}
base_lr=${BASE_LR:-1e-05}
action_lr=${ACTION_LR:-1e-04}
qwen_lr=${QWEN_LR:-1e-05}
weight_decay=${WEIGHT_DECAY:-0.01}
vlm_pack=false
camera_bucket_sampling=false
camera_bucket_window_size=${CAMERA_BUCKET_WINDOW_SIZE:-100}
logging_frequency=${LOG_FREQUENCY:-100}
eval_interval=${EVAL_INTERVAL:-100}
eval_num_batches=${EVAL_NUM_BATCHES:-1}
eval_seed=${EVAL_SEED:-42}
is_resume=${IS_RESUME:-false}
resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-latest}
enable_gradient_checkpointing=${ENABLE_GC:-true}
nonfinite_check_interval=${NONFINITE_CHECK_INTERVAL:-100}
use_liger_kernel=${USE_LIGER_KERNEL:-true}
liger_rope=${LIGER_ROPE:-true}
liger_rms_norm=${LIGER_RMS_NORM:-false}
liger_swiglu=${LIGER_SWIGLU:-false}
liger_fused_linear_cross_entropy=${LIGER_FUSED_LINEAR_CROSS_ENTROPY:-false}
accelerate_config_file=${ACCELERATE_CONFIG_FILE:-starVLA/config/deepseeds/deepspeed_zero2.yaml}
# === End of environment variable configuration ===
###########################################################################################


SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../setup" && pwd)"
# shellcheck source=/dev/null
requested_wandb_mode=${WANDB_MODE:-}
source "${SETUP_DIR}/wandb.env"
export WANDB_MODE="${requested_wandb_mode:-${WANDB_MODE:-online}}"
export WANDB_PROJECT="${WANDB_PROJECT:-VLAct_SpikeBrain_Pretrain}"
if [[ -z "${WANDB_ENTITY:-}" || "${WANDB_ENTITY}" == "liujn" || "${WANDB_ENTITY}" == "jianingliu" ]]; then
  export WANDB_ENTITY=app123
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file ${accelerate_config_file} \
  --main_process_port ${MAIN_PROCESS_PORT:-29502} \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla_cotrain.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.disjoint_action_layout True \
  --framework.mask_padded_action_dims True \
  --framework.single_arm_loss_weight 8.0 \
  --framework.dual_arm_loss_weight 1.0 \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.heads ${heads} \
  --framework.head_loss_weights ${head_loss_weights} \
  --framework.action_model.repeated_diffusion_steps 2 \
  --framework.action_model.expert_model_path ${action_expert_model} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.balance_dataset_weights True \
  --datasets.vla_data.per_device_batch_size ${vla_per_device_batch_size} \
  --datasets.vla_data.video_backend pyav \
  --datasets.vla_data.sequential_step_sampling False \
  --datasets.vla_data.camera_bucket_sampling ${camera_bucket_sampling} \
  --datasets.vla_data.camera_bucket_window_size ${camera_bucket_window_size} \
  --datasets.vla_data.prompt_prefix_embodiment True \
  --datasets.vlm_data.dataset_use "${vlm_data}" \
  --datasets.vlm_data.eval_dataset "" \
  --datasets.vlm_data.per_device_batch_size ${vlm_per_device_batch_size} \
  --datasets.vlm_data.data_flatten ${vlm_pack} \
  --trainer.gradient_accumulation_steps ${gradient_accumulation_steps} \
  --trainer.enable_gradient_checkpointing ${enable_gradient_checkpointing} \
  --trainer.shortest_angular_joint_loss True \
  --trainer.shortest_angular_joint_loss_diff True \
  --trainer.endpoint_wrap_loss_weight 0.5 \
  --trainer.loss_scale.vlm 0.2 \
  --trainer.freeze_modules ${freeze_module_list} \
  --framework.qwenvl.use_liger_kernel ${use_liger_kernel} \
  --framework.qwenvl.liger_rope ${liger_rope} \
  --framework.qwenvl.liger_rms_norm ${liger_rms_norm} \
  --framework.qwenvl.liger_swiglu ${liger_swiglu} \
  --framework.qwenvl.liger_fused_linear_cross_entropy ${liger_fused_linear_cross_entropy} \
  --framework.qwenvl.fail_on_nonfinite True \
  --framework.qwenvl.nonfinite_check_interval ${nonfinite_check_interval} \
  --trainer.learning_rate.base ${base_lr} \
  --trainer.learning_rate.qwen_vl_interface ${qwen_lr} \
  --trainer.learning_rate.action_model ${action_lr} \
  --trainer.weight_decay ${weight_decay} \
  --trainer.optimizer.weight_decay ${weight_decay} \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval ${eval_interval} \
  --trainer.eval_num_batches ${eval_num_batches} \
  --trainer.eval_seed ${eval_seed} \
  --trainer.is_resume ${is_resume} \
  --trainer.resume_from_checkpoint ${resume_from_checkpoint} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project "${WANDB_PROJECT:-VLAct_SpikeBrain_Pretrain}" \
  --wandb_entity "${WANDB_ENTITY:-app123}"
train_status=$?

if [[ "${KERNEL_CACHE_PERSIST:-true}" == "true" ]]; then
  cp -a "${runtime_kernel_cache}/." "${persistent_kernel_cache}/" 2>/dev/null || \
    echo "[vlact_spikebrain] warning: failed to persist kernel cache" >&2
fi

exit "${train_status}"
