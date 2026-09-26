#!/usr/bin/env bash
# Supplementary phase-A cache loop (InternA1/RoboCOIN/DROID_100) until DROID stable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP="${REPO_ROOT}/scripts/setup"
LOG_DIR="${REPO_ROOT}/logs/setup"
STABLE_FLAG="${LOG_DIR}/.droid_upload_stable"
LOG="${LOG_DIR}/watchdog_cache_phaseA.log"

exec >>"${LOG}" 2>&1
echo "========== $(date -Is) watchdog_cache_phaseA start =========="

while [[ ! -f "${STABLE_FLAG}" ]]; do
  SKIP_DROID=1 bash "${SETUP}/build_pretrain_caches_parallel.sh" || true
  sleep 180
done

echo "========== $(date -Is) watchdog_cache_phaseA done (DROID stable) =========="
