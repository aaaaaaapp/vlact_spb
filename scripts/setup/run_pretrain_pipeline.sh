#!/usr/bin/env bash
# Master pipeline: env -> download -> postprocess -> caches -> pretrain
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP_DIR="${REPO_ROOT}/scripts/setup"
LOG_DIR="${REPO_ROOT}/logs/setup"
mkdir -p "$LOG_DIR"

export CONDA_BASE="${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}"
export ENV_NAME="${ENV_NAME:-vlact}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "========== $(date -Is) VLAct pretrain pipeline =========="

# Step 1: conda env
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  bash "${SETUP_DIR}/create_conda_env.sh" | tee "${LOG_DIR}/create_conda_env.master.log"
fi
conda activate "$ENV_NAME"

# Step 2: start/resume downloads (background supervisor)
if ! pgrep -f "download_pretrain_assets.sh" >/dev/null 2>&1; then
  nohup bash "${SETUP_DIR}/download_pretrain_assets.sh" \
    > "${LOG_DIR}/download_supervisor.nohup.log" 2>&1 &
  echo "Started download supervisor PID=$!"
fi

# Step 3: wait for minimum assets (base models + COCO)
wait_for_dir() {
  local dir="$1"
  local marker="${2:-config.json}"
  while [[ ! -f "${dir}/${marker}" && ! -d "${dir}/${marker}" ]]; do
    echo "Waiting for ${dir} (${marker}) ..."
    sleep 120
  done
}

wait_for_dir "${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action" "config.json"
wait_for_dir "${REPO_ROOT}/playground/Pretrained_models/Qwen3-0.6B" "config.json"
wait_for_dir "${REPO_ROOT}/playground/Datasets/LLaVA-OneVision-COCO" "sharegpt4v_coco.zip"

bash "${SETUP_DIR}/postprocess_datasets.sh" | tee "${LOG_DIR}/postprocess_datasets.log"

# Step 4: wait for all VLA datasets (best effort)
for d in DROID DROID_100 RoboCOIN InternData-A1 MolmoAct-Dataset; do
  while [[ ! -d "${REPO_ROOT}/playground/Datasets/${d}" ]]; do
    echo "Waiting for dataset ${d} ..."
    sleep 300
  done
done

bash "${SETUP_DIR}/postprocess_datasets.sh" | tee -a "${LOG_DIR}/postprocess_datasets.log"
bash "${SETUP_DIR}/build_pretrain_caches.sh" | tee "${LOG_DIR}/build_pretrain_caches.log"

# Step 5: launch pretraining (8x A100)
cd "$REPO_ROOT"
export WANDB_MODE=disabled
bash scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh \
  | tee "${LOG_DIR}/pretrain_qwen3_single_node.log"

echo "========== $(date -Is) pipeline finished =========="
