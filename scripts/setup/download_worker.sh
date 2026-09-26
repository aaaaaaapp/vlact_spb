#!/usr/bin/env bash
# Download a single HF repo with auto-resume. Used by distributed download launcher.
set -u

NAME="${1:?name}"
REPO="${2:?repo}"
TARGET="${3:?target}"
REPO_TYPE="${4:-dataset}"
shift 4 || true
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_ROOT="${REPO_ROOT}/logs/hf_downloads"
LOG_FILE="${LOG_ROOT}/${NAME}.log"
MAX_WORKERS="${MAX_WORKERS:-8}"
DOWNLOAD_TIMEOUT_SECONDS="${DOWNLOAD_TIMEOUT_SECONDS:-7200}"

mkdir -p "$TARGET" "$LOG_ROOT"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/setup/proxy.env"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-600}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/mnt/jfzn/liujn/.cache/huggingface}"

if [[ -f "${HOME}/.huggingface/token" ]]; then
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.huggingface/token")"
fi

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}" 2>/dev/null || true

HF_CMD="${HF_CMD:-hf}"
command -v "$HF_CMD" >/dev/null 2>&1 || HF_CMD="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/hf"

if [[ -f "$LOG_FILE" ]] && grep -q "COMPLETE name=${NAME} " "$LOG_FILE"; then
  echo "$(date -Is) SKIP $NAME already complete on $(hostname)" >>"$LOG_FILE"
  exit 0
fi

echo "$(date -Is) WORKER_START name=$NAME host=$(hostname) repo=$REPO workers=$MAX_WORKERS" >>"$LOG_FILE"

while true; do
  echo "$(date -Is) downloading repo=$REPO target=$TARGET host=$(hostname)" >>"$LOG_FILE"
  if timeout --foreground "${DOWNLOAD_TIMEOUT_SECONDS}s" \
    env HF_ENDPOINT="$HF_ENDPOINT" \
         HF_HUB_ETAG_TIMEOUT="$HF_HUB_ETAG_TIMEOUT" \
         HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" \
         HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
         HF_TOKEN="${HF_TOKEN:-}" \
         http_proxy="$http_proxy" https_proxy="$https_proxy" \
         HTTP_PROXY="$HTTP_PROXY" HTTPS_PROXY="$HTTPS_PROXY" \
    "$HF_CMD" download "$REPO" \
      --repo-type "$REPO_TYPE" \
      --local-dir "$TARGET" \
      --max-workers "$MAX_WORKERS" \
      "${EXTRA_ARGS[@]}" >>"$LOG_FILE" 2>&1; then
    if ! find "$TARGET/.cache/huggingface/download" -type f -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
      echo "$(date -Is) COMPLETE name=$NAME repo=$REPO host=$(hostname)" >>"$LOG_FILE"
      exit 0
    fi
    echo "$(date -Is) partial incomplete, retrying" >>"$LOG_FILE"
  else
    echo "$(date -Is) failed or timed out, retry in 20s" >>"$LOG_FILE"
  fi
  sleep 20
done
