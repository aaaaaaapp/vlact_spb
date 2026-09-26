#!/usr/bin/env bash
# Wait for DROID upload -> postprocess -> caches -> launch 8-GPU pretrain with WandB.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP="${REPO_ROOT}/scripts/setup"
LOG_DIR="${REPO_ROOT}/logs/setup"
MASTER_LOG="${LOG_DIR}/overnight_pretrain.log"

export CONDA_BASE="${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}"
export ENV_NAME="${ENV_NAME:-vlact}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${LOG_DIR}"
exec >>"${MASTER_LOG}" 2>&1

log() { echo "$(date -Is) [overnight] $*" >&2; }

# shellcheck source=/dev/null
[[ -f "${SETUP}/proxy.env" ]] && source "${SETUP}/proxy.env"
# shellcheck source=/dev/null
[[ -f "${SETUP}/wandb.env" ]] && source "${SETUP}/wandb.env"

DROID_PATH="${REPO_ROOT}/playground/Datasets/DROID"
DROID_FTP="/mnt/jfzn/ftp_upload/vlact/playground/Datasets/DROID"
STABLE_CHECKS="${DROID_STABLE_CHECKS:-3}"
POLL_SEC="${DROID_POLL_SEC:-300}"

wait_droid_upload() {
  log "waiting for DROID upload to stabilize (poll=${POLL_SEC}s, need ${STABLE_CHECKS} stable)"
  local prev=-1 stable=0
  while true; do
    if [[ ! -f "${DROID_FTP}/meta/info.json" ]]; then
      log "DROID ftp meta missing, sleep ${POLL_SEC}s"
      sleep "${POLL_SEC}"
      continue
    fi
    local sz
    sz="$(du -sb "${DROID_FTP}" 2>/dev/null | awk '{print $1}')"
    local gb=$(( sz / 1024 / 1024 / 1024 ))
    log "DROID ftp size=${gb}GB (bytes=${sz})"
    if [[ "${prev}" -ge 0 && "${sz}" -eq "${prev}" ]]; then
      stable=$(( stable + 1 ))
      log "DROID stable count=${stable}/${STABLE_CHECKS}"
      if [[ "${stable}" -ge "${STABLE_CHECKS}" ]]; then
        log "DROID upload considered complete"
        return 0
      fi
    else
      stable=0
    fi
    prev="${sz}"
    sleep "${POLL_SEC}"
  done
}

wait_path() {
  local p="$1" desc="${2:-$1}"
  while [[ ! -e "${p}" ]]; do
    log "waiting: ${desc}"
    sleep 120
  done
  log "ready: ${desc}"
}

# shellcheck source=/dev/null
source "${SETUP}/nodes.env"

count_free_gpus() {
  local ip="$1" threshold="${2:-10000}"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${CLUSTER_USER}@${ip}" \
    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk -v t=${threshold} '{if(\$1<t) c++} END{print c+0}'" \
    2>/dev/null || echo 0
}

STAGE2_RUN_ID="${STAGE2_RUN_ID:-aloha_real_ljn_drop_glitch_stage2_full_bs16}"
STAGE2_STOPPED=0

pick_gpu_node() {
  local ip free threshold="${1:-10000}" best_ip="" best_free=0
  for ip in "${NODE001_IP}" "${NODE002_IP}" "${NODE003_IP}" "${NODE004_IP}"; do
    free="$(count_free_gpus "${ip}" "${threshold}")"
    log "GPU poll ${ip}: free_gpus(<${threshold}MB)=${free}/8"
    if [[ "${free}" -ge 8 ]]; then
      echo "${ip}"
      return 0
    fi
    if [[ "${free}" -gt "${best_free}" ]]; then
      best_free="${free}"
      best_ip="${ip}"
    fi
  done
  log "no 8-GPU idle node (best=${best_ip} free=${best_free}/8, threshold=${threshold}MB)"
  return 1
}

stop_stage2_on_node3() {
  log "freeing node3: stopping ${STAGE2_RUN_ID}"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${CLUSTER_USER}@${NODE003_IP}" bash -s <<EOF >&2
set +e
pkill -f "${STAGE2_RUN_ID}" 2>/dev/null
sleep 45
if pgrep -f "${STAGE2_RUN_ID}" >/dev/null 2>&1; then
  echo "stage2 still running"
  exit 1
fi
echo "stage2 stopped"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
EOF
}

wait_for_gpu_node() {
  local poll="${GPU_POLL_SEC:-300}"
  log "waiting for 8 GPUs on node1-node4 (poll=${poll}s); will stop stage2 on node3 if needed"
  while true; do
    if pick_ip="$(pick_gpu_node 10000)"; then
      log "selected node ${pick_ip} for pretrain (fully idle)"
      echo "${pick_ip}"
      return 0
    fi
    if [[ "${STAGE2_STOPPED}" -eq 0 ]]; then
      if stop_stage2_on_node3; then
        STAGE2_STOPPED=1
        # After stage2 stop, node3 may still host ~18GB/GPU background jobs; allow up to 25GB.
        if pick_ip="$(pick_gpu_node 25000)"; then
          log "selected node3 after stage2 stop (shared with maze_wm ~18GB/GPU, user accepted)"
          echo "${pick_ip}"
          return 0
        fi
        log "node3 stage2 stopped but 8 GPUs not yet below 25GB; keep polling"
      else
        log "stage2 stop failed or still running; will retry next poll"
      fi
    fi
    sleep "${poll}"
  done
}

log "========== overnight pipeline start =========="
log "WANDB_MODE=${WANDB_MODE:-unset} WANDB_PROJECT=${WANDB_PROJECT:-unset} WANDB_ENTITY=${WANDB_ENTITY:-unset}"

# Do not block on DROID for steps that only need other ftp datasets.
bash "${SETUP}/link_datasets_from_ftp_upload.sh"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${REPO_ROOT}"

log "postprocess_datasets (InternData parallel extract + MolmoAct convert)"
bash "${SETUP}/postprocess_datasets.sh" | tee "${LOG_DIR}/postprocess_overnight.log"

chmod +x "${SETUP}/build_pretrain_caches_parallel.sh"

DROID_STABLE_FLAG="${LOG_DIR}/.droid_upload_stable"
rm -f "${DROID_STABLE_FLAG}"

log "background phase-A caches (InternData/RoboCOIN/DROID_100) while DROID uploads"
nohup bash -c '
  while [[ ! -f "'"${DROID_STABLE_FLAG}"'" ]]; do
    SKIP_DROID=1 bash "'"${SETUP}"'/build_pretrain_caches_parallel.sh" || true
    sleep 180
  done
' > "${LOG_DIR}/build_pretrain_caches_phaseA.nohup.log" 2>&1 &
PHASEA_PID=$!
log "phase-A cache pid=${PHASEA_PID}"

wait_droid_upload
touch "${DROID_STABLE_FLAG}"

log "DROID stable; waiting for phase-A caches"
wait "${PHASEA_PID}" || log "WARN phase-A cache exited non-zero (see cache_job_*.log)"

log "phase-B DROID caches (parallel ext1/ext2)"
for rt in oxe_droid_exterior1_wrist_manualvel_strict_50 oxe_droid_exterior2_wrist_manualvel_strict_50; do
  python examples/DROID/compute_filtered_caches_droid_action_delta.py \
    --data-root "${REPO_ROOT}/playground/Datasets" --data-name DROID \
    --robot-type "${rt}" --filter-delta-eef-steps --skip-existing \
    >> "${LOG_DIR}/cache_job_droid_${rt}.log" 2>&1 &
done
wait

wait_path "${REPO_ROOT}/playground/Datasets/InternData-A1/split_aloha_tasks.txt" "InternData split_aloha_tasks"

# MolmoAct convert runs in background during postprocess; wait before MolmoAct cache.
log "waiting for MolmoAct v21 conversion"
while [[ ! -f "${REPO_ROOT}/playground/Datasets/MolmoAct-Dataset/molmoact_household_v21/meta/info.json" ]]; do
  if pgrep -f "convert_molmoact_to_lerobot_v21.py" >/dev/null; then
    tail -1 "${LOG_DIR}/molmoact_convert.log" 2>/dev/null || true
  else
    log "MolmoAct convert not running; re-trigger postprocess MolmoAct step"
    bash "${SETUP}/postprocess_datasets.sh" | tee -a "${LOG_DIR}/postprocess_overnight.log"
  fi
  sleep 600
done
log "MolmoAct v21 ready"

log "build_pretrain_caches (MolmoAct only; other sets done in parallel phases)"
python examples/MolmoAct/compute_caches_molmoact_delta_eef.py \
  --data-root "${REPO_ROOT}/playground/Datasets/MolmoAct-Dataset" --mode filtered \
  --robot-types \
    molmoact_franka_exterior1_wrist_manualvel_strict_50 \
    molmoact_franka_exterior2_wrist_manualvel_strict_50 \
  --filter-delta-eef-steps --skip-existing \
  | tee "${LOG_DIR}/build_pretrain_caches_molmoact.log"

log "pretrain prerequisites check"
for p in \
  playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/config.json \
  playground/Pretrained_models/Qwen3-0.6B/config.json \
  playground/Datasets/DROID/meta/info.json \
  logs/setup/.cc3m_ready; do
  [[ -e "${REPO_ROOT}/${p}" ]] || { log "ERROR missing ${p}"; exit 1; }
done

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-VLAct_Qwen3_Pretrain}"
source "${SETUP}/wandb.env"
export WANDB_ENTITY="${WANDB_ENTITY:-jianingliu}"

PRETRAIN_NODE="$(wait_for_gpu_node)"
log "launch pretrain on ${PRETRAIN_NODE} (8 GPUs, VLA global BS=16x8=128, VLM global BS=2x8=16, GAS=1)"
ssh -o BatchMode=yes "${CLUSTER_USER}@${PRETRAIN_NODE}" bash -s <<REMOTE_EOF | "${SETUP}/prefix_log.sh" | tee "${LOG_DIR}/pretrain_qwen3_$(date '+%Y%m%d_%H%M%S').log"
set -euo pipefail
export CONDA_BASE="${CONDA_BASE}"
export ENV_NAME="${ENV_NAME}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_ENTITY="${WANDB_ENTITY}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
source "\${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "\${ENV_NAME}"
cd "${REPO_ROOT}"
bash scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh
REMOTE_EOF

log "========== overnight pipeline done =========="
