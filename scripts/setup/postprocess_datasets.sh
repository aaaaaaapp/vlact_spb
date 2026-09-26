#!/usr/bin/env bash
# Post-download dataset preparation for VLAct pretraining.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/playground/Datasets"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"

cd "$REPO_ROOT"

echo "========== $(date -Is) postprocess_datasets start =========="

# Writable composite dirs for ftp datasets (meta/cache must not live on read-only ftp).
bash "${REPO_ROOT}/scripts/setup/materialize_writable_ftp_datasets.sh" || true

# 1) LLaVA-OneVision-COCO unzip
COCO_DIR="${DATA_ROOT}/LLaVA-OneVision-COCO"
if [[ -d "${COCO_DIR}/images/sharegpt4v_coco" ]] && [[ -n "$(ls -A "${COCO_DIR}/images/sharegpt4v_coco" 2>/dev/null)" ]]; then
  echo "[COCO] images already present, skip unzip"
elif [[ -f "${COCO_DIR}/sharegpt4v_coco.zip" ]]; then
  echo "[COCO] Unzipping sharegpt4v_coco.zip"
  unzip -o "${COCO_DIR}/sharegpt4v_coco.zip" -d "${COCO_DIR}"
fi

# 2) LLaVA-ReCap-CC3M: validate parquet shards + write manifest (training-ready marker)
CC3M_DATA="${DATA_ROOT}/LLaVA-ReCap-CC3M/data"
CC3M_MANIFEST="${REPO_ROOT}/logs/setup/cc3m_manifest.json"
CC3M_READY="${REPO_ROOT}/logs/setup/.cc3m_ready"
if [[ -d "${CC3M_DATA}" ]]; then
  shard_count="$(find "${CC3M_DATA}" -maxdepth 1 -name 'train-*-of-*.parquet' 2>/dev/null | wc -l)"
  if [[ "${shard_count}" -eq 0 ]]; then
    echo "[CC3M] WARN: no parquet shards under ${CC3M_DATA}"
  elif [[ -f "${CC3M_READY}" ]] && [[ -f "${CC3M_MANIFEST}" ]]; then
    echo "[CC3M] already validated (${shard_count} shards), skip"
  else
    echo "[CC3M] Validating ${shard_count} parquet shards -> ${CC3M_MANIFEST}"
    python "${REPO_ROOT}/scripts/setup/postprocess_cc3m.py" \
      --data-dir "${CC3M_DATA}" \
      --out "${CC3M_MANIFEST}" \
      --ready-marker "${CC3M_READY}"
  fi
else
  echo "[CC3M] WARN: missing ${CC3M_DATA}"
fi

# 3) InternData-A1: extract split_aloha tarballs + build task list
INTERN_RAW="${DATA_ROOT}/InternData-A1_raw"
INTERN_DST="${DATA_ROOT}/InternData-A1"
SPLIT_ALOHA_RAW="${INTERN_RAW}/sim_updated/pick_and_place_tasks/split_aloha"
SPLIT_ALOHA_DST="${INTERN_DST}/sim_updated/pick_and_place_tasks/split_aloha"
mkdir -p "$SPLIT_ALOHA_DST"

extract_one_tar() {
  local archive="$1" dst="$2"
  local task_name marker
  task_name="$(basename "${archive}" .tar.gz)"
  marker="${dst}/${task_name}"
  if [[ -d "${marker}" ]] && find "${marker}" -name info.json -path '*/meta/info.json' -quit 2>/dev/null; then
    echo "[InternA1] skip extract (exists): ${task_name}"
    return 0
  fi
  echo "[InternA1] extracting ${archive} -> ${dst}/"
  tar -xzf "${archive}" -C "${dst}"
}

if [[ -d "${SPLIT_ALOHA_RAW}" ]]; then
  shopt -s nullglob
  INTERN_EXTRACT_JOBS="${INTERN_EXTRACT_JOBS:-4}"
  pending=()
  for archive in "${SPLIT_ALOHA_RAW}"/*.tar.gz; do
    pending+=("${archive}")
  done
  if [[ "${#pending[@]}" -gt 0 ]]; then
    echo "[InternA1] parallel extract jobs=${INTERN_EXTRACT_JOBS} archives=${#pending[@]}"
    export -f extract_one_tar
    export SPLIT_ALOHA_DST
    printf '%s\n' "${pending[@]}" | xargs -P "${INTERN_EXTRACT_JOBS}" -I{} \
      bash -c 'extract_one_tar "$1" "$SPLIT_ALOHA_DST"' _ {}
  fi

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
print(f"Wrote {len(tasks)} split_aloha tasks -> {out}")
PY
fi

# 4) MolmoAct v2.1: convert into writable local dir, symlink back for training paths.
MOLMOACT_LOG="${REPO_ROOT}/logs/setup/molmoact_convert.log"
MOLMOACT_V21_LOCAL="${MOLMOACT_V21_LOCAL:-${DATA_ROOT}/MolmoAct-v21-local}"
MOLMOACT_LINK="${DATA_ROOT}/MolmoAct-Dataset"
MOLMOACT_FTP="${FTP_UPLOAD_DATA:-/mnt/jfzn/ftp_upload/vlact/playground/Datasets}/MolmoAct-Dataset"
mkdir -p "$(dirname "${MOLMOACT_LOG}")" "${MOLMOACT_V21_LOCAL}"

# Training expects MolmoAct-Dataset/{molmoact_household_v21,...}; ftp root is read-only.
if [[ -L "${MOLMOACT_LINK}" ]]; then
  rm -f "${MOLMOACT_LINK}"
fi
mkdir -p "${MOLMOACT_LINK}"
for sub in molmoact_dataset_household molmoact_dataset_tabletop; do
  if [[ -d "${MOLMOACT_FTP}/${sub}" ]] && [[ ! -e "${MOLMOACT_LINK}/${sub}" ]]; then
    ln -sfn "${MOLMOACT_FTP}/${sub}" "${MOLMOACT_LINK}/${sub}"
  fi
done
link_molmoact_v21() {
  local subset="$1"
  local name="molmoact_${subset}_v21"
  local src="${MOLMOACT_V21_LOCAL}/${name}"
  local dst="${MOLMOACT_LINK}/${name}"
  if [[ -d "${src}/meta" ]]; then
    ln -sfn "${src}" "${dst}"
    echo "[MolmoAct] linked ${dst} -> ${src}"
  fi
}

if [[ -f "${MOLMOACT_LINK}/molmoact_household_v21/meta/info.json" ]]; then
  echo "[MolmoAct] v21 already present, skip convert"
  link_molmoact_v21 household
  link_molmoact_v21 tabletop
elif pgrep -f "convert_molmoact_to_lerobot_v21.py" >/dev/null 2>&1; then
  echo "[MolmoAct] convert already running, log: ${MOLMOACT_LOG}"
else
  echo "[MolmoAct] Starting convert dst=${MOLMOACT_V21_LOCAL} -> ${MOLMOACT_LOG}"
  MOLMOACT_PROCS="${MOLMOACT_PROCS:-16}"
  (
    python examples/MolmoAct/convert_molmoact_to_lerobot_v21.py \
      --src-root "${MOLMOACT_LINK}" \
      --dst-root "${MOLMOACT_V21_LOCAL}" \
      --subsets household tabletop \
      --num-processes "${MOLMOACT_PROCS}"
    link_molmoact_v21 household
    link_molmoact_v21 tabletop
  ) >> "${MOLMOACT_LOG}" 2>&1 &
  echo "[MolmoAct] pid=$!"
fi

# 5) DROID_100 wrapper from RoboInter (only when official lerobot/droid_100 is absent)
# Do NOT overwrite an existing DROID_100 — training expects lerobot/droid_100 (~100 episodes),
# not the full RoboInter annotated corpus.
if [[ ! -f "${DATA_ROOT}/DROID_100/meta/info.json" ]]; then
  ROBOINTER_SRC="${DATA_ROOT}/RoboInter-Data/Annotation_with_action_lerobotv21/lerobot_droid_anno"
  if [[ -d "$ROBOINTER_SRC" ]]; then
    echo "[DROID_100] Preparing wrapper from RoboInter annotated DROID"
    python examples/DROID/prepare_droid_100.py \
      --source-dir "$ROBOINTER_SRC" \
      --skip-existing || true
  fi
fi

echo "========== $(date -Is) postprocess_datasets done =========="
