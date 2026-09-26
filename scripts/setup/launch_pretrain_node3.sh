#!/usr/bin/env bash
# Launch 8-GPU VLAct pretrain on a cluster node (default: poll node1-4 for 8 idle GPUs).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP="${REPO_ROOT}/scripts/setup"
LOG_DIR="${REPO_ROOT}/logs/setup"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/pretrain_qwen3_$(date '+%Y%m%d_%H%M%S').log"
ln -sfn "$(basename "${LOG}")" "${LOG_DIR}/pretrain_qwen3_latest.log"
USER="${CLUSTER_USER:-liujn}"
POLL_SEC="${GPU_POLL_SEC:-300}"
MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-10000}"

# shellcheck source=/dev/null
source "${SETUP}/nodes.env"
# shellcheck source=/dev/null
source "${SETUP}/wandb.env"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-VLAct_Qwen3_Pretrain}"
if [[ -z "${WANDB_ENTITY:-}" || "${WANDB_ENTITY}" == "liujn" || "${WANDB_ENTITY}" == "jianingliu" ]]; then
  export WANDB_ENTITY=app123
fi
export http_proxy="${http_proxy:-http://10.119.18.5:3128}"
export https_proxy="${https_proxy:-http://10.119.18.5:3128}"

log() {
  local msg="$(date -Is) [launch] $*"
  echo "${msg}" >> "${LOG}"
  echo "${msg}" >&2
}

count_free_gpus() {
  local ip="$1" threshold="${2:-${MEM_THRESHOLD_MB}}"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${USER}@${ip}" \
    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk -v t=${threshold} '{if(\$1<t) c++} END{print c+0}'" \
    2>/dev/null || echo 0
}

pick_gpu_node() {
  local ip free threshold="${1:-${MEM_THRESHOLD_MB}}"
  for ip in "${NODE001_IP}" "${NODE002_IP}" "${NODE003_IP}" "${NODE004_IP}"; do
    free="$(count_free_gpus "${ip}" "${threshold}")"
    log "GPU poll ${ip}: free_gpus(<${threshold}MB)=${free}/8"
    if [[ "${free}" -ge 8 ]]; then
      echo "${ip}"
      return 0
    fi
  done
  return 1
}

resolve_node() {
  if [[ -n "${PRETRAIN_NODE:-}" ]]; then
    log "using PRETRAIN_NODE=${PRETRAIN_NODE} (colocate with existing jobs if GPUs are partially used)"
    echo "${PRETRAIN_NODE}"
    return 0
  fi
  if [[ "${WAIT_FOR_GPU:-0}" == "1" ]]; then
    log "waiting for 8 idle GPUs (poll=${POLL_SEC}s, mem<${MEM_THRESHOLD_MB}MB)"
    while true; do
      if pick_ip="$(pick_gpu_node "${MEM_THRESHOLD_MB}")"; then
        echo "${pick_ip}"
        return 0
      fi
      sleep "${POLL_SEC}"
    done
  fi
  echo "${NODE003_IP}"
}

NODE="$(resolve_node)"
log "log file: ${LOG}"
log "starting pretrain on ${USER}@${NODE} (WANDB_MODE=${WANDB_MODE})"

ssh -o BatchMode=yes "${USER}@${NODE}" bash -s <<REMOTE_EOF | "${SETUP}/prefix_log.sh" | tee -a "${LOG}"
set -euo pipefail
export CONDA_BASE="${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}"
export ENV_NAME="${ENV_NAME:-vlact}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_ENTITY="${WANDB_ENTITY}"
export http_proxy="${http_proxy}"
export https_proxy="${https_proxy}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
source "\${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "\${ENV_NAME}"
cd "${REPO_ROOT}"
nohup bash -c "stdbuf -oL -eL bash scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh 2>&1 | stdbuf -oL -eL bash scripts/setup/prefix_log.sh >> '${LOG}'" </dev/null >/dev/null 2>&1 &
echo "pretrain pid=\$!"
REMOTE_EOF
