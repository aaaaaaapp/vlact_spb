#!/usr/bin/env bash
# 通用 hf download 封装（支持断点续传 + 重试）
set -euo pipefail

NAME="$1"
REPO="$2"
TARGET="$3"
REPO_TYPE="${4:-model}"
shift 4 || true
EXTRA=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_env.sh"

LOG_DIR="${STAGING_DIR}/logs"
LOG="${LOG_DIR}/${NAME}.log"
mkdir -p "${LOG_DIR}" "$(dirname "${TARGET}")"

MIN_BYTES="${MIN_BYTES:-0}"
if [[ -f "${TARGET}/.download_complete" ]]; then
  echo "[$(date -Is)] SKIP ${NAME} (marked complete)" | tee -a "${LOG}"
  exit 0
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date -Is)] START ${NAME} attempt=${attempt} repo=${REPO} -> ${TARGET}" | tee -a "${LOG}"

  if "${HF_CMD}" download "${REPO}" \
      --repo-type "${REPO_TYPE}" \
      --local-dir "${TARGET}" \
      --max-workers "${MAX_WORKERS}" \
      "${EXTRA[@]}" >> "${LOG}" 2>&1; then
    if ! find "${TARGET}/.cache/huggingface/download" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
      echo "[$(date -Is)] COMPLETE ${NAME}" | tee -a "${LOG}"
      touch "${TARGET}/.download_complete"
      exit 0
    fi
    echo "[$(date -Is)] incomplete files remain, retry" | tee -a "${LOG}"
  else
    echo "[$(date -Is)] failed, retry in 30s" | tee -a "${LOG}"
  fi
  sleep 30
done
