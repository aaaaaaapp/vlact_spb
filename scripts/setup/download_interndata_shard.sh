#!/usr/bin/env bash
# Download a shard of InternData-A1 split_aloha tarballs (for multi-node split).
# INTERNDATA_SHARD=0 INTERNDATA_SHARDS=3  -> shard 0 of 3
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DST="${REPO_ROOT}/playground/Datasets/InternData-A1_raw"
LOG="${REPO_ROOT}/logs/hf_downloads/interndata_shard_${INTERNDATA_SHARD:-0}.log"
SHARD="${INTERNDATA_SHARD:-0}"
SHARDS="${INTERNDATA_SHARDS:-1}"
PARALLEL="${INTERNDATA_PARALLEL:-2}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/setup/proxy.env"

if [[ -f "${HOME}/.huggingface/token" ]]; then
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.huggingface/token")"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "$DST" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}" 2>/dev/null || true
PYTHON="${CONDA_PREFIX}/bin/python"

echo "========== $(date -Is) interndata shard ${SHARD}/${SHARDS} host=$(hostname) parallel=$PARALLEL python=$PYTHON =========="

PYTHONUNBUFFERED=1 "$PYTHON" -u - <<PY
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import hf_hub_download

shard = int("${SHARD}")
shards = int("${SHARDS}")
parallel = int("${PARALLEL}")
token = os.environ["HF_TOKEN"]
endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
local_dir = Path("${DST}")
repo_id = "InternRobotics/InternData-A1"

ALL = [
    "sim_updated/pick_and_place_tasks/split_aloha/continues_pick_and_place_part1.tar.gz",
    "sim_updated/pick_and_place_tasks/split_aloha/continues_pick_and_place_part2.tar.gz",
    "sim_updated/pick_and_place_tasks/split_aloha/parallel_pick_and_place.tar.gz",
    "sim_updated/pick_and_place_tasks/split_aloha/pick_and_place_part1.tar.gz",
    "sim_updated/pick_and_place_tasks/split_aloha/pick_and_place_part2.tar.gz",
    "sim_updated/pick_and_place_tasks/split_aloha/pick_and_place_part3.tar.gz",
    "sim_updated/pick_and_place_tasks/split_aloha/pick_and_place_part4.tar.gz",
    "sim_updated/pick_and_place_tasks/split_aloha/pick_and_place_part5.tar.gz",
    "sim_updated/pick_and_place_tasks/split_aloha/pick_and_place_part6.tar.gz",
    "sim_updated/pick_and_place_tasks/split_aloha/pick_and_place_part7.tar.gz",
]
files = [f for i, f in enumerate(ALL) if i % shards == shard]

def log(msg): print(msg, flush=True)

def dl(filename):
    target = local_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return filename, "skip"
    while True:
        try:
            hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename,
                token=token, endpoint=endpoint, local_dir=str(local_dir))
            return filename, "OK"
        except Exception as e:
            log(f"retry {filename}: {e}")
            time.sleep(15)

log(f"shard {shard}/{shards}: {len(files)} files on {os.uname().nodename}")
with ThreadPoolExecutor(max_workers=parallel) as ex:
    futs = {ex.submit(dl, f): f for f in files}
    for i, fut in enumerate(as_completed(futs), 1):
        fn, st = fut.result()
        log(f"[{i}/{len(files)}] {fn}: {st}")
log("SHARD DONE")
PY

echo "========== $(date -Is) interndata shard ${SHARD} done =========="
