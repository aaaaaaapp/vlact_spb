#!/usr/bin/env bash
# Extract RoboInter lerobot_droid_anno chunk-*.tar archives to writable parquet dirs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROBOINTER_TAR_DIR:-/mnt/jfzn/ftp_upload/vlact/playground/Datasets/RoboInter-Data/Annotation_with_action_lerobotv21/lerobot_droid_anno/data}"
DST="${ROBOINTER_DATA_DIR:-${REPO_ROOT}/playground/Datasets/RoboInter-lerobot_droid_anno-data}"
LOG="${REPO_ROOT}/logs/setup/extract_robointer_droid_data.log"
JOBS="${EXTRACT_JOBS:-8}"

mkdir -p "${DST}" "$(dirname "${LOG}")"
exec >>"${LOG}" 2>&1

log() { echo "$(date -Is) [robointer-extract] $*"; }

extract_one() {
  local tar_path="$1"
  local name chunk_dir count
  name="$(basename "${tar_path}" .tar)"
  chunk_dir="${DST}/${name}"
  if [[ -d "${chunk_dir}" ]] && compgen -G "${chunk_dir}/*.parquet" >/dev/null; then
    log "skip ${name} (already extracted)"
    return 0
  fi
  rm -rf "${chunk_dir}"
  mkdir -p "${DST}"
  tar -xf "${tar_path}" -C "${DST}"
  count="$(find "${chunk_dir}" -maxdepth 1 -name '*.parquet' | wc -l)"
  log "done ${name} (${count} parquet)"
}

log "========== start src=${SRC} dst=${DST} jobs=${JOBS} =========="
mapfile -t TARS < <(find "${SRC}" -maxdepth 1 -name 'chunk-*.tar' | sort)
log "found ${#TARS[@]} tar archives"

if [[ ${#TARS[@]} -eq 0 ]]; then
  log "no tar archives; assuming parquet already extracted"
  exit 0
fi

running=0
for tar_path in "${TARS[@]}"; do
  while (( running >= JOBS )); do
    if wait -n 2>/dev/null; then
      running=$((running - 1))
    else
      wait || true
      running=0
    fi
  done
  extract_one "${tar_path}" &
  running=$((running + 1))
done
while (( running > 0 )); do
  if wait -n 2>/dev/null; then
    running=$((running - 1))
  else
    wait || true
    break
  fi
done

log "total parquet: $(find "${DST}" -name '*.parquet' | wc -l)"
log "========== done =========="
