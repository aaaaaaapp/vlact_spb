#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export VLACT_ENV_DIR="${VLACT_ENV_DIR:-/mnt/jfzn/liujn/miniconda3/envs/vlact}"
export PATH="${VLACT_ENV_DIR}/bin:${PATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_TIMEOUT=3600 DEEPSPEED_TIMEOUT=720
exec "${VLACT_ENV_DIR}/bin/accelerate" launch \
  --config_file "${repo_root}/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
  --main_process_port "${MAIN_PROCESS_PORT:-29522}" --num_processes 8 \
  "${repo_root}/starVLA/training/train_starvla.py" \
  --config_yaml "${repo_root}/examples/Aloha/train_files/starvla_cotrain_aloha_real_ljn.yaml"
