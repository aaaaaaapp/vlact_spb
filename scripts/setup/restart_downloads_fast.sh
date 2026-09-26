#!/usr/bin/env bash
# Restart downloads with optimized settings (limited concurrency + hf_transfer).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/setup/proxy.env"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export MAX_CONCURRENT_REPOS=5
export MAX_WORKERS=8
export DOWNLOAD_TIMEOUT_SECONDS=7200
export INTERNDATA_PARALLEL=3

LOG="${REPO_ROOT}/logs/setup/restart_downloads_fast.log"
mkdir -p "$(dirname "$LOG")"

exec >>"$LOG" 2>&1
echo "========== $(date -Is) restart_downloads_fast =========="

# Stop old supervisor + duplicate hf download workers (keep watchdog/postprocess)
echo "Stopping old download workers..."
pkill -f "download_pretrain_assets.sh" 2>/dev/null || true
pkill -f "hf download.*VLAct-main/playground" 2>/dev/null || true
pkill -f "download_interndata_a1.sh" 2>/dev/null || true
sleep 5
rm -rf /tmp/vlact_hf_download_supervisor/slots/* 2>/dev/null || true
flock -u 9 2>/dev/null || rm -f /tmp/vlact_hf_download_supervisor/supervisor.lock 2>/dev/null || true

# Ensure hf_transfer in vlact env
source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}" 2>/dev/null || true
pip install -q "hf-transfer>=0.1.9" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true

echo "Starting optimized supervisor (max_concurrent=$MAX_CONCURRENT_REPOS workers=$MAX_WORKERS)..."
nohup env MAX_CONCURRENT_REPOS="$MAX_CONCURRENT_REPOS" MAX_WORKERS="$MAX_WORKERS" \
  DOWNLOAD_TIMEOUT_SECONDS="$DOWNLOAD_TIMEOUT_SECONDS" \
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  bash "${REPO_ROOT}/scripts/setup/download_pretrain_assets.sh" \
  >> "${REPO_ROOT}/logs/setup/download_supervisor.nohup.log" 2>&1 &
echo "supervisor PID=$!"

echo "Starting parallel InternData download (parallel=$INTERNDATA_PARALLEL, delayed 60s)..."
nohup bash -c 'sleep 60; exec env INTERNDATA_PARALLEL="'"$INTERNDATA_PARALLEL"'" HF_HUB_ENABLE_HF_TRANSFER=1 bash "'"${REPO_ROOT}"'/scripts/setup/download_interndata_a1.sh"' \
  >> "${REPO_ROOT}/logs/hf_downloads/interndata_a1_manual.log" 2>&1 &
echo "interndata PID=$!"

echo "========== $(date -Is) restart done =========="
