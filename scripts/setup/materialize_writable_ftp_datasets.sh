#!/usr/bin/env bash
# Replace read-only ftp symlinks with composite dirs (writable meta + cache outputs).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/playground/Datasets"
FTP_DATA="${FTP_UPLOAD_DATA:-/mnt/jfzn/ftp_upload/vlact/playground/Datasets}"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"
cd "${REPO_ROOT}"

log() { echo "$(date -Is) [materialize] $*"; }

link_meta_files() {
  local src_meta="$1" dst_meta="$2"
  mkdir -p "${dst_meta}"
  shopt -s nullglob
  for f in "${src_meta}"/*; do
    local name
    name="$(basename "${f}")"
    case "${name}" in
      modality.json|steps_data_index*.pkl|stats_gr00t*.json) continue ;;
    esac
    ln -sfn "${f}" "${dst_meta}/${name}"
  done
}

materialize_droid() {
  local name="DROID"
  local dst="${DATA_ROOT}/DROID"
  local ftp="${FTP_DATA}/DROID"
  [[ -d "${ftp}" ]] || { log "skip ${name} (missing ftp)"; return 0; }
  if [[ -L "${dst}" ]]; then rm -f "${dst}"; fi
  if [[ -d "${dst}/meta" ]] && [[ -L "${dst}/data" ]]; then
    log "${name} already materialized"
    if [[ ! -f "${dst}/meta/modality.json" ]]; then
      log "generating missing ${name} modality.json"
      python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from examples.DROID.compute_stats_gr00t_mp import ensure_modality_json
ensure_modality_json(Path("playground/Datasets/DROID"), overwrite=False)
PY
    fi
    return 0
  fi
  rm -rf "${dst}"
  mkdir -p "${dst}/meta"
  ln -sfn "${ftp}/data" "${dst}/data"
  ln -sfn "${ftp}/videos" "${dst}/videos"
  link_meta_files "${ftp}/meta" "${dst}/meta"
  log "materialized ${name}"
  python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from examples.DROID.compute_stats_gr00t_mp import ensure_modality_json
ensure_modality_json(Path("playground/Datasets/DROID"), overwrite=False)
PY
}

materialize_robocoin() {
  local name="RoboCOIN"
  local dst="${DATA_ROOT}/RoboCOIN"
  local ftp="${FTP_DATA}/RoboCOIN"
  [[ -d "${ftp}" ]] || return 0
  if [[ -L "${dst}" ]]; then rm -f "${dst}"; fi
  mkdir -p "${dst}"
  for task in "${ftp}"/*; do
    [[ -d "${task}" ]] || continue
    [[ -f "${task}/meta/info.json" ]] || continue
    local task_name="${dst}/$(basename "${task}")"
    mkdir -p "${task_name}/meta"
    ln -sfn "${task}/data" "${task_name}/data" 2>/dev/null || true
    ln -sfn "${task}/videos" "${task_name}/videos" 2>/dev/null || true
    link_meta_files "${task}/meta" "${task_name}/meta"
  done
  log "materialized RoboCOIN tasks"
  python examples/RoboCoin/gen_modality_json.py --data-root "${dst}"
}

materialize_molmoact() {
  local name="MolmoAct-Dataset"
  local dst="${DATA_ROOT}/MolmoAct-Dataset"
  local ftp="${FTP_DATA}/MolmoAct-Dataset"
  [[ -d "${ftp}" ]] || return 0
  if [[ -L "${dst}" ]]; then rm -f "${dst}"; fi
  mkdir -p "${dst}"
  for sub in molmoact_dataset_household molmoact_dataset_tabletop; do
    [[ -d "${ftp}/${sub}" ]] && ln -sfn "${ftp}/${sub}" "${dst}/${sub}"
  done
  mkdir -p "${dst}/molmoact_household_v21" "${dst}/molmoact_tabletop_v21"
  log "materialized MolmoAct-Dataset (v21 dirs writable)"
}

fix_intern_tasks() {
  python - <<'PY'
from pathlib import Path
root = Path("playground/Datasets/InternData-A1")
tasks = []
for p in sorted(root.rglob("meta/info.json")):
    rel = p.parent.parent.relative_to(root)
    if "split_aloha" in str(rel):
        tasks.append(str(rel).replace("\\", "/"))
out = root / "split_aloha_tasks.txt"
out.write_text("\n".join(tasks) + ("\n" if tasks else ""))
print(f"Wrote {len(tasks)} tasks -> {out}")
print("sample:", tasks[0] if tasks else "none")
PY
}

log "========== start =========="
materialize_droid
materialize_robocoin
materialize_molmoact
fix_intern_tasks
log "========== done =========="
