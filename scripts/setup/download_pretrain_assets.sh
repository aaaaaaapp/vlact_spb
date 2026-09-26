#!/usr/bin/env bash
# Auto-resume HuggingFace downloads for VLAct pretraining.
# Optimized: limited concurrent repos, hf_transfer, skip completed, priority queue.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/playground/Datasets"
MODEL_ROOT="${REPO_ROOT}/playground/Pretrained_models"
LOG_ROOT="${REPO_ROOT}/logs/hf_downloads"
STATE_DIR="/tmp/vlact_hf_download_supervisor"
SLOT_DIR="${STATE_DIR}/slots"

# Fewer concurrent repos = less proxy contention; more workers per repo = faster large files
MAX_CONCURRENT_REPOS="${MAX_CONCURRENT_REPOS:-5}"
MAX_WORKERS="${MAX_WORKERS:-8}"
DOWNLOAD_TIMEOUT_SECONDS="${DOWNLOAD_TIMEOUT_SECONDS:-7200}"

mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$LOG_ROOT" "$STATE_DIR" "$SLOT_DIR"
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

HF_CMD="${HF_CMD:-/usr/local/bin/hf}"
command -v "$HF_CMD" >/dev/null 2>&1 || HF_CMD="hf"
command -v "$HF_CMD" >/dev/null 2>&1 || HF_CMD="huggingface-cli"

exec 9>"${STATE_DIR}/supervisor.lock"
if ! flock -n 9; then
  echo "download supervisor already running"
  exit 0
fi

is_complete() {
  local name="$1"
  [[ -f "${LOG_ROOT}/${name}.log" ]] && grep -q "COMPLETE name=${name} " "${LOG_ROOT}/${name}.log"
}

acquire_slot() {
  local slot i
  while true; do
    for i in $(seq 1 "$MAX_CONCURRENT_REPOS"); do
      slot="${SLOT_DIR}/slot_${i}"
      if mkdir "$slot" 2>/dev/null; then
        echo "$i"
        return 0
      fi
    done
    sleep 15
  done
}

release_slot() {
  rmdir "${SLOT_DIR}/slot_${1}" 2>/dev/null || true
}

download_one() {
  local name="$1"
  local repo="$2"
  local target="$3"
  local repo_type="${4:-model}"
  shift 4 || true
  local extra_args=("$@")
  local log_file="${LOG_ROOT}/${name}.log"
  local slot

  if is_complete "$name"; then
    echo "$(date -Is) SKIP $name (already complete)" >>"${LOG_ROOT}/supervisor.log"
    return 0
  fi

  slot="$(acquire_slot)"
  mkdir -p "$target"
  echo "$(date -Is) START name=$name repo=$repo slot=$slot workers=$MAX_WORKERS" >>"$log_file"

  while true; do
    if is_complete "$name"; then
      release_slot "$slot"
      return 0
    fi
    echo "$(date -Is) downloading repo=$repo target=$target slot=$slot proxy=$http_proxy hf_transfer=$HF_HUB_ENABLE_HF_TRANSFER" >>"$log_file"
    if timeout --foreground "${DOWNLOAD_TIMEOUT_SECONDS}s" \
      env HF_ENDPOINT="$HF_ENDPOINT" \
           HF_HUB_ETAG_TIMEOUT="$HF_HUB_ETAG_TIMEOUT" \
           HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" \
           HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
           HF_TOKEN="${HF_TOKEN:-}" \
           http_proxy="$http_proxy" https_proxy="$https_proxy" \
           HTTP_PROXY="$HTTP_PROXY" HTTPS_PROXY="$HTTPS_PROXY" \
      "$HF_CMD" download "$repo" \
        --repo-type "$repo_type" \
        --local-dir "$target" \
        --max-workers "$MAX_WORKERS" \
        "${extra_args[@]}" >>"$log_file" 2>&1; then
      if ! find "$target/.cache/huggingface/download" -type f -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
        echo "$(date -Is) COMPLETE name=$name repo=$repo" >>"$log_file"
        release_slot "$slot"
        return 0
      fi
      echo "$(date -Is) partial incomplete files remain, retrying" >>"$log_file"
    else
      echo "$(date -Is) download failed or timed out via proxy $http_proxy" >>"$log_file"
    fi
    sleep 20
  done
}

start_child() {
  local name="$1"
  shift
  if is_complete "$name"; then
    echo "$(date -Is) skip start $name (complete)" >>"${LOG_ROOT}/supervisor.log"
    return 0
  fi
  local pid_file="${STATE_DIR}/${name}.pid"
  download_one "$name" "$@" &
  echo $! >"$pid_file"
  echo "$(date -Is) started $name pid=$(cat "$pid_file")" >>"${LOG_ROOT}/supervisor.log"
}

# --- Priority: large / blocking datasets first, then smaller ---
# VLM (largest)
start_child llava_recap_cc3m \
  lmms-lab/LLaVA-ReCap-CC3M \
  "${DATA_ROOT}/LLaVA-ReCap-CC3M" \
  dataset \
  --include "data/**"

# VLA large
start_child droid \
  cadene/droid \
  "${DATA_ROOT}/DROID" \
  dataset

start_child molmoact \
  allenai/MolmoAct-Dataset \
  "${DATA_ROOT}/MolmoAct-Dataset" \
  dataset \
  --include "molmoact_dataset_household/train/**" \
  --include "molmoact_dataset_tabletop/train/**"

start_child robointer_droid \
  InternRobotics/RoboInter-Data \
  "${DATA_ROOT}/RoboInter-Data" \
  dataset \
  --include "Annotation_with_action_lerobotv21/lerobot_droid_anno/**"

# Base models (skip if complete — cheap to check)
start_child qwen3_vl_action \
  StarVLA/Qwen3-VL-4B-Instruct-Action \
  "${MODEL_ROOT}/Qwen3-VL-4B-Instruct-Action" \
  model

start_child qwen3_0_6b \
  Qwen/Qwen3-0.6B \
  "${MODEL_ROOT}/Qwen3-0.6B" \
  model

start_child llava_coco \
  StarVLA/LLaVA-OneVision-COCO \
  "${DATA_ROOT}/LLaVA-OneVision-COCO" \
  dataset \
  --include "sharegpt4v_coco.zip" \
  --include "llava_jsons/**"

start_child droid_100 \
  lerobot/droid_100 \
  "${DATA_ROOT}/DROID_100" \
  dataset

# InternData-A1: use scripts/setup/download_interndata_a1.sh (parallel tar.gz)

ROBOCOIN_TASKS=(
  Cobot_Magic_clean_blackboard
  Cobot_Magic_fold_the_towel
  Cobot_Magic_move_the_bread
  Cobot_Magic_place_the_cup
  Cobot_Magic_stack_the_blocks
  Split_aloha_basket_storage_peach
  Split_aloha_fold_the_towel
  Split_aloha_move_the_bread
  Split_aloha_place_the_cup
  Split_aloha_stack_the_blocks
)

for task in "${ROBOCOIN_TASKS[@]}"; do
  safe_name="robocoin_${task,,}"
  start_child "$safe_name" \
    "RoboCOIN/${task}" \
    "${DATA_ROOT}/RoboCOIN/${task}" \
    dataset
done

echo "$(date -Is) supervisor started max_concurrent=$MAX_CONCURRENT_REPOS max_workers=$MAX_WORKERS hf_transfer=$HF_HUB_ENABLE_HF_TRANSFER" >>"${LOG_ROOT}/supervisor.log"

while true; do
  for pid_file in "${STATE_DIR}"/*.pid; do
    [[ -f "$pid_file" ]] || continue
    name="$(basename "$pid_file" .pid)"
    if ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      if is_complete "$name"; then
        echo "$(date -Is) worker $name finished OK" >>"${LOG_ROOT}/supervisor.log"
      else
        echo "$(date -Is) worker $name exited; check ${LOG_ROOT}/${name}.log" >>"${LOG_ROOT}/supervisor.log"
        # Restart failed/incomplete workers
        case "$name" in
          llava_recap_cc3m) start_child llava_recap_cc3m lmms-lab/LLaVA-ReCap-CC3M "${DATA_ROOT}/LLaVA-ReCap-CC3M" dataset --include "data/**" ;;
          droid) start_child droid cadene/droid "${DATA_ROOT}/DROID" dataset ;;
          molmoact) start_child molmoact allenai/MolmoAct-Dataset "${DATA_ROOT}/MolmoAct-Dataset" dataset --include "molmoact_dataset_household/train/**" --include "molmoact_dataset_tabletop/train/**" ;;
          robointer_droid) start_child robointer_droid InternRobotics/RoboInter-Data "${DATA_ROOT}/RoboInter-Data" dataset --include "Annotation_with_action_lerobotv21/lerobot_droid_anno/**" ;;
          qwen3_vl_action) start_child qwen3_vl_action StarVLA/Qwen3-VL-4B-Instruct-Action "${MODEL_ROOT}/Qwen3-VL-4B-Instruct-Action" model ;;
          qwen3_0_6b) start_child qwen3_0_6b Qwen/Qwen3-0.6B "${MODEL_ROOT}/Qwen3-0.6B" model ;;
          llava_coco) start_child llava_coco StarVLA/LLaVA-OneVision-COCO "${DATA_ROOT}/LLaVA-OneVision-COCO" dataset --include "sharegpt4v_coco.zip" --include "llava_jsons/**" ;;
          droid_100) start_child droid_100 lerobot/droid_100 "${DATA_ROOT}/DROID_100" dataset ;;
          robocoin_*) 
            task="${name#robocoin_}"
            # shellcheck disable=SC2201
            for t in "${ROBOCOIN_TASKS[@]}"; do
              [[ "${t,,}" == "$task" ]] && start_child "$name" "RoboCOIN/${t}" "${DATA_ROOT}/RoboCOIN/${t}" dataset && break
            done
            ;;
        esac
      fi
      rm -f "$pid_file"
    fi
  done
  sleep 60
done
