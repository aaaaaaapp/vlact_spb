#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="${REPO_ROOT}/logs/setup/wait_droid100_pretrain.nohup.log"
CACHE_LOG="${REPO_ROOT}/logs/setup/cache_job_droid_100_rerun.log"

exec >>"${LOG}" 2>&1
log() { echo "$(date -Is) [wait-pretrain] $*"; }

log "waiting for DROID_100 cache python to finish..."
while pgrep -f "compute_filtered_caches_droid_action_delta.py.*DROID_100" >/dev/null; do
  sleep 60
done

if [[ -f "${CACHE_LOG}" ]] && grep -q "Traceback" "${CACHE_LOG}"; then
  log "ERROR DROID_100 cache failed; see ${CACHE_LOG}"
  exit 1
fi

pkl="$(find "${REPO_ROOT}/playground/Datasets/DROID_100/meta" -name 'steps_data_index_filtered_*.pkl' | head -1)"
if [[ -z "${pkl}" ]]; then
  log "WARN no DROID_100 steps pkl; pretrain may rebuild stats on rank0"
else
  log "DROID_100 cache ready: ${pkl}"
fi

log "launching pretrain on node3"
bash "${REPO_ROOT}/scripts/setup/launch_pretrain_node3.sh"
log "done"
