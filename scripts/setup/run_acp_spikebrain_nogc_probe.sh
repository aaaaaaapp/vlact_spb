#!/usr/bin/env bash
set -euo pipefail

export http_proxy="${http_proxy:-http://10.119.18.5:3128}"
export https_proxy="${https_proxy:-http://10.119.18.5:3128}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export PATH="/mnt/jfzn/miniconda3/envs/starvla_qwen/bin:${PATH}"
export PYTHONPATH="/mnt/jfzn/liujn/code/VLAct-spikebrain/.acp_deps:/mnt/jfzn/miniconda3/envs/starvla_qwen/lib/python3.11/site-packages"
export NO_ALBUMENTATIONS_UPDATE=1

cd /mnt/jfzn/liujn/code/VLAct-spikebrain

python -c "import pandas, requests, torch, transformers, deepspeed, accelerate; print('ACP_IMPORT_CHECK_OK', flush=True)"

export NUM_PROCESSES=2
export VLA_BS=8
export VLM_BS=1
export GRAD_ACCUM=2
export MAX_TRAIN_STEPS=50
export RUN_ID=acp_vlact_spikebrain_nogc_cache_2gpu_50step_v4
export QWEN_LR=1e-5
export ACTION_LR=1e-4
export WEIGHT_DECAY=0.01
export LOG_FREQUENCY=10
export EVAL_INTERVAL=100000
export WANDB_MODE=offline
export ENABLE_GC=false
export NONFINITE_CHECK_INTERVAL=10
export MAIN_PROCESS_PORT=29624
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_CONFIG_FILE=starVLA/config/deepseeds/deepspeed_zero2_small_bucket.yaml

bash scripts/run_scripts/Pretrain/pretrain_spikebrain_single_node.sh \
  2>&1 | tee logs/setup/acp_vlact_spikebrain_nogc_cache_2gpu_50step_v4.log
