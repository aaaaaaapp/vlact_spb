#!/usr/bin/env bash
# Download cadene/droid then MolmoAct one at a time (avoid proxy 503).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP="${REPO_ROOT}/scripts/setup"
DATA="${REPO_ROOT}/playground/Datasets"
LOG="${REPO_ROOT}/logs/hf_downloads/droid_molmo_sequential.log"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "========== $(date -Is) sequential droid+molmo start host=$(hostname) =========="

export MAX_WORKERS="${MAX_WORKERS:-8}"
export DOWNLOAD_TIMEOUT_SECONDS="${DOWNLOAD_TIMEOUT_SECONDS:-7200}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Stop any parallel workers for these two datasets
pkill -f "download_worker.sh droid " 2>/dev/null || true
pkill -f "download_worker.sh molmoact " 2>/dev/null || true
pkill -f "hf download cadene/droid" 2>/dev/null || true
pkill -f "hf download allenai/MolmoAct-Dataset" 2>/dev/null || true
sleep 5

echo "[1/2] cadene/droid (~400GB) ..."
bash "${SETUP}/download_worker.sh" \
  droid cadene/droid "${DATA}/DROID" dataset

echo "[2/2] allenai/MolmoAct-Dataset ..."
bash "${SETUP}/download_worker.sh" \
  molmoact allenai/MolmoAct-Dataset "${DATA}/MolmoAct-Dataset" dataset \
  --include "molmoact_dataset_household/train/**" \
  --include "molmoact_dataset_tabletop/train/**"

echo "========== $(date -Is) sequential droid+molmo ALL DONE =========="
