#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${STARVLA_ROOT}"

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600  # timeout set to 1 hour (unit: seconds)
export DEEPSPEED_TIMEOUT=720  # collective wait timeout, unit: minutes (rank0 cold-cache filtering takes ~40min)



###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenHybrid_xrobot_padding
freeze_module_list=qwen_vl_interface.model.visual
for layer_idx in $(seq 0 17); do
  freeze_module_list+=" ,qwen_vl_interface.model.language_model.layers.${layer_idx}"
done
freeze_module_list=${freeze_module_list// ,/,}
base_vlm=${BASE_VLM:-./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}
config_yaml=./examples/InternA1/train_files/starvla_cotrain_agilex_franka_50_padding.yaml
data_root_dir=./playground/Datasets
run_root_dir=./results/Checkpoints
data_mix=agilex_franka_5data_manualvel_balance_50
vlm_coco_data=${VLM_COCO_DATA:-./playground/Datasets/LLaVA-OneVision-COCO/llava_jsons/sharegpt4v_coco.json::./playground/Datasets/LLaVA-OneVision-COCO/images}
vlm_cc3m_data=${VLM_CC3M_DATA:-./playground/Datasets/LLaVA-ReCap-CC3M/data}
vlm_coco_repeat=5
vlm_data=${vlm_coco_data}
for _ in $(seq 2 ${vlm_coco_repeat}); do
  vlm_data+=,${vlm_coco_data}
done
vlm_data+=,${vlm_cc3m_data}
image_size_buckets='[[320,180],[280,210]]'
heads=oft,gr00t,pi
head_loss_weights=oft:1,gr00t:1,pi:1
run_id=${RUN_ID:-vlact_qwen3_pretrain}
# Native 8-GPU single-node defaults: VLA global batch 128 and VLM global batch 16.
# Override these only for a deliberate scaling study.
# Env overrides: VLA_BS / VLM_BS / GRAD_ACCUM / NUM_PROCESSES / RUN_ID / MAIN_PROCESS_PORT / MAX_TRAIN_STEPS
vla_per_device_batch_size=${VLA_BS:-16}
vlm_per_device_batch_size=${VLM_BS:-2}
gradient_accumulation_steps=${GRAD_ACCUM:-1}
num_processes=${NUM_PROCESSES:-8}
max_train_steps=${MAX_TRAIN_STEPS:-100000}
base_lr=${BASE_LR:-1e-05}
qwen_lr=${QWEN_LR:-1e-05}
action_lr=${ACTION_LR:-1e-04}
vlm_pack=${VLM_PACK:-false}
camera_bucket_sampling=${CAMERA_BUCKET_SAMPLING:-false}
camera_bucket_window_size=${CAMERA_BUCKET_WINDOW_SIZE:-100}
logging_frequency=${LOG_FREQUENCY:-100}
eval_interval=${EVAL_INTERVAL:-100}
eval_num_batches=${EVAL_NUM_BATCHES:-1}
eval_seed=${EVAL_SEED:-42}
is_resume=${IS_RESUME:-false}
resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-latest}
# === End of environment variable configuration ===
###########################################################################################


SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../setup" && pwd)"
# shellcheck source=/dev/null
requested_wandb_mode=${WANDB_MODE:-}
source "${SETUP_DIR}/wandb.env"
export WANDB_MODE="${requested_wandb_mode:-${WANDB_MODE:-online}}"
export WANDB_PROJECT="${WANDB_PROJECT:-VLAct_Qwen3_Pretrain}"
if [[ -z "${WANDB_ENTITY:-}" || "${WANDB_ENTITY}" == "liujn" || "${WANDB_ENTITY}" == "jianingliu" ]]; then
  export WANDB_ENTITY=app123
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
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
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.balance_dataset_weights True \
  --datasets.vla_data.per_device_batch_size ${vla_per_device_batch_size} \
  --datasets.vla_data.video_backend pyav \
  --datasets.vla_data.sequential_step_sampling False \
  --datasets.vla_data.camera_bucket_sampling ${camera_bucket_sampling} \
  --datasets.vla_data.camera_bucket_window_size ${camera_bucket_window_size} \
  --datasets.vla_data.image_size_buckets "${image_size_buckets}" \
  --datasets.vla_data.prompt_prefix_embodiment True \
  --datasets.vlm_data.dataset_use "${vlm_data}" \
  --datasets.vlm_data.eval_dataset "" \
  --datasets.vlm_data.per_device_batch_size ${vlm_per_device_batch_size} \
  --datasets.vlm_data.data_flatten ${vlm_pack} \
  --trainer.gradient_accumulation_steps ${gradient_accumulation_steps} \
  --trainer.shortest_angular_joint_loss True \
  --trainer.shortest_angular_joint_loss_diff True \
  --trainer.endpoint_wrap_loss_weight 0.5 \
  --trainer.loss_scale.vlm 0.2 \
  --trainer.freeze_modules ${freeze_module_list} \
  --framework.qwenvl.use_liger_kernel True \
  --framework.qwenvl.liger_rope True \
  --framework.qwenvl.liger_fused_linear_cross_entropy True \
  --framework.qwenvl.liger_rms_norm True \
  --framework.qwenvl.liger_swiglu True \
  --trainer.learning_rate.base ${base_lr} \
  --trainer.learning_rate.qwen_vl_interface ${qwen_lr} \
  --trainer.learning_rate.action_model ${action_lr} \
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
  --wandb_project "${WANDB_PROJECT:-VLAct_Qwen3_Pretrain}" \
  --wandb_entity "${WANDB_ENTITY:-app123}"
