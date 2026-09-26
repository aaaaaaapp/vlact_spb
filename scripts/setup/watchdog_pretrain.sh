#!/usr/bin/env bash
# Watch downloads + conda env; run postprocess/caches/pretrain when ready.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP_DIR="${REPO_ROOT}/scripts/setup"
LOG_DIR="${REPO_ROOT}/logs/setup"
mkdir -p "$LOG_DIR"

export CONDA_BASE="${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}"
export ENV_NAME="${ENV_NAME:-vlact}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# shellcheck source=/dev/null
source "${SETUP_DIR}/proxy.env"

log() { echo "$(date -Is) [watchdog] $*"; }

wait_conda_env() {
  while true; do
    if source "${CONDA_BASE}/etc/profile.d/conda.sh" && conda activate "$ENV_NAME" 2>/dev/null; then
      if python -c "import torch, transformers, deepspeed" >/dev/null 2>&1; then
        log "conda env $ENV_NAME ready (torch + transformers + deepspeed)"
        return 0
      fi
    fi
    log "waiting for conda env $ENV_NAME (torch/transformers/deepspeed) ..."
    sleep 120
  done
}

wait_path() {
  local path="${1:?wait_path: missing path argument}"
  local desc="${2:-$(basename "$path")}"
  while [[ ! -e "$path" ]]; do
    log "waiting for ${desc} (${path})"
    sleep 300
  done
  log "ready: ${desc}"
}

log "watchdog started"

wait_conda_env

# Minimum for smoke validation
wait_path "${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/config.json" "Qwen3-VL-4B-Instruct-Action"
wait_path "${REPO_ROOT}/playground/Pretrained_models/Qwen3-0.6B/config.json" "Qwen3-0.6B"
wait_path "${REPO_ROOT}/playground/Datasets/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" "LLaVA-OneVision-COCO zip"

bash "${SETUP_DIR}/postprocess_datasets.sh" | tee "${LOG_DIR}/postprocess_datasets.watchdog.log"

# Wait for core VLA datasets (may take hours/days)
wait_path "${REPO_ROOT}/playground/Datasets/DROID/meta/info.json" "DROID"
wait_path "${REPO_ROOT}/playground/Datasets/DROID_100/meta/info.json" "DROID_100"
wait_path "${REPO_ROOT}/playground/Datasets/MolmoAct-Dataset/molmoact_household_v21/meta/info.json" "MolmoAct-Dataset"
wait_path "${REPO_ROOT}/playground/Datasets/InternData-A1/split_aloha_tasks.txt" "InternData-A1 split_aloha_tasks"
wait_path "${REPO_ROOT}/playground/Datasets/RoboCOIN/Cobot_Magic_clean_blackboard/meta/info.json" "RoboCOIN sample task"

bash "${SETUP_DIR}/postprocess_datasets.sh" | tee -a "${LOG_DIR}/postprocess_datasets.watchdog.log"
bash "${SETUP_DIR}/build_pretrain_caches.sh" | tee "${LOG_DIR}/build_pretrain_caches.watchdog.log"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"
export WANDB_MODE=disabled

log "launching pretrain (8 GPUs)"
bash scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh \
  | tee "${LOG_DIR}/pretrain_qwen3_single_node.watchdog.log"

log "pretrain finished"
