#!/usr/bin/env bash
# Distribute VLAct dataset downloads across node001-004 (shared /mnt/jfzn).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/nodes.env"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/proxy.env"

DATA="${REPO_ROOT}/playground/Datasets"
MODEL="${REPO_ROOT}/playground/Pretrained_models"
LOG="${REPO_ROOT}/logs/setup/distributed_downloads.log"
mkdir -p "$(dirname "$LOG")"

exec >>"$LOG" 2>&1
echo "========== $(date -Is) launch_distributed_downloads from $(hostname) =========="

stop_local_downloads() {
  pkill -f "download_pretrain_assets.sh" 2>/dev/null || true
  pkill -f "download_interndata_a1.sh" 2>/dev/null || true
  pkill -f "download_interndata_shard.sh" 2>/dev/null || true
  pkill -f "download_worker.sh" 2>/dev/null || true
  pkill -f "hf download.*VLAct-main/playground" 2>/dev/null || true
  rm -rf /tmp/vlact_hf_download_supervisor/slots/* 2>/dev/null || true
  rm -f /tmp/vlact_hf_download_supervisor/supervisor.lock 2>/dev/null || true
}

remote_run() {
  local ip="$1"
  local label="$2"
  shift 2
  local cmd="$*"
  echo "--- launch $label ($ip) ---"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${CLUSTER_USER}@${ip}" bash -lc "
    set -u
    export http_proxy=${http_proxy} https_proxy=${https_proxy}
    export HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY}
    export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1
    export CONDA_BASE=/mnt/jfzn/liujn/miniconda3 ENV_NAME=vlact
    export REPO_ROOT=${REPO_ROOT}
    ${cmd}
  "
}

stop_local_downloads
for ip in "$NODE002_IP" "$NODE003_IP" "$NODE004_IP"; do
  ssh -o BatchMode=yes "${CLUSTER_USER}@${ip}" "pkill -f 'download_worker.sh|download_interndata|hf download.*VLAct-main/playground' 2>/dev/null; true" || true
done
sleep 3

chmod +x "${SCRIPT_DIR}/download_worker.sh" "${SCRIPT_DIR}/download_interndata_shard.sh"

# --- node001: LLaVA-ReCap-CC3M (~139GB) ---
nohup env MAX_WORKERS=8 bash "${SCRIPT_DIR}/download_worker.sh" \
  llava_recap_cc3m lmms-lab/LLaVA-ReCap-CC3M "${DATA}/LLaVA-ReCap-CC3M" dataset \
  --include "data/**" \
  >> "${REPO_ROOT}/logs/hf_downloads/node001_cc3m.nohup.log" 2>&1 &
echo "node001 CC3M PID=$!"

# --- node002: DROID then MolmoAct sequential (see download_droid_molmo_sequential.sh) ---
remote_run "$NODE002_IP" node002-droid-molmo-seq "
  nohup env MAX_WORKERS=8 HF_HUB_ENABLE_HF_TRANSFER=1 \
    bash ${REPO_ROOT}/scripts/setup/download_droid_molmo_sequential.sh \
    >> ${REPO_ROOT}/logs/hf_downloads/droid_molmo_sequential.nohup.log 2>&1 &
  echo droid_molmo_seq PID=\$!
"

# --- node003: InternData shard 0/3 (4 files) ---
remote_run "$NODE003_IP" node003-interndata "
  nohup env INTERNDATA_SHARD=0 INTERNDATA_SHARDS=3 INTERNDATA_PARALLEL=2 \
    bash ${REPO_ROOT}/scripts/setup/download_interndata_shard.sh \
    >> ${REPO_ROOT}/logs/hf_downloads/node003_interndata.nohup.log 2>&1 &
  echo interndata PID=\$!
"

# --- node004: InternData shard 1/3 + shard 2/3 via sequential launch, RoboCOIN, RoboInter ---
# shard 1 and 2 on node004 (6 files total); RoboCOIN 10 tasks 2-at-a-time
remote_run "$NODE004_IP" node004-interndata-robocoin "
  nohup env INTERNDATA_SHARD=1 INTERNDATA_SHARDS=3 INTERNDATA_PARALLEL=2 \
    bash ${REPO_ROOT}/scripts/setup/download_interndata_shard.sh \
    >> ${REPO_ROOT}/logs/hf_downloads/node004_interndata_s1.nohup.log 2>&1 &
  nohup env INTERNDATA_SHARD=2 INTERNDATA_SHARDS=3 INTERNDATA_PARALLEL=2 \
    bash ${REPO_ROOT}/scripts/setup/download_interndata_shard.sh \
    >> ${REPO_ROOT}/logs/hf_downloads/node004_interndata_s2.nohup.log 2>&1 &
  nohup bash -c '
    source ${REPO_ROOT}/scripts/setup/proxy.env
    export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1
    TASKS=(Cobot_Magic_clean_blackboard Cobot_Magic_fold_the_towel Cobot_Magic_move_the_bread \
      Cobot_Magic_place_the_cup Cobot_Magic_stack_the_blocks Split_aloha_basket_storage_peach \
      Split_aloha_fold_the_towel Split_aloha_move_the_bread Split_aloha_place_the_cup Split_aloha_stack_the_blocks)
    for t in \"\${TASKS[@]}\"; do
      safe=\"robocoin_\${t,,}\"
      bash ${REPO_ROOT}/scripts/setup/download_worker.sh \"\$safe\" \"RoboCOIN/\$t\" \"${DATA}/RoboCOIN/\$t\" dataset
    done
    bash ${REPO_ROOT}/scripts/setup/download_worker.sh robointer_droid InternRobotics/RoboInter-Data \
      ${DATA}/RoboInter-Data dataset --include \"Annotation_with_action_lerobotv21/lerobot_droid_anno/**\"
  ' >> ${REPO_ROOT}/logs/hf_downloads/node004_robocoin.nohup.log 2>&1 &
  echo robocoin PID=\$!
"

echo "========== $(date -Is) all nodes launched =========="
echo "node001: CC3M"
echo "node002: DROID -> MolmoAct sequential"
echo "node003: InternData shard 0/3"
echo "node004: InternData shard 1-2/3 + RoboCOIN + RoboInter"
