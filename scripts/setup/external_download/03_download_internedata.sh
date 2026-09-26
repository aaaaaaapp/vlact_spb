#!/usr/bin/env bash
# InternData-A1 split_aloha 10 个 tar.gz（gated，需要 HF_TOKEN）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_env.sh"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] InternData-A1 需要 HF_TOKEN 或 ~/.huggingface/token" >&2
  exit 1
fi

DST="${STAGING_DATASETS}/InternData-A1_raw"
LOG="${STAGING_DIR}/logs/interndata_a1.log"
SHARD="${INTERNDATA_SHARD:-0}"
SHARDS="${INTERNDATA_SHARDS:-1}"
PARALLEL="${INTERNDATA_PARALLEL:-2}"

mkdir -p "${DST}" "$(dirname "${LOG}")"
exec >>"${LOG}" 2>&1

echo "========== $(date -Is) InternData shard ${SHARD}/${SHARDS} =========="

python3 -u - <<PY
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import hf_hub_download

shard = int("${SHARD}")
shards = int("${SHARDS}")
parallel = int("${PARALLEL}")
token = os.environ["HF_TOKEN"]
endpoint = os.environ.get("HF_ENDPOINT") or None
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
            hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=filename,
                token=token, endpoint=endpoint, local_dir=str(local_dir),
            )
            return filename, "OK"
        except Exception as e:
            log(f"retry {filename}: {e}")
            time.sleep(15)

log(f"shard {shard}/{shards}: {len(files)} files")
with ThreadPoolExecutor(max_workers=parallel) as ex:
    futs = {ex.submit(dl, f): f for f in files}
    for i, fut in enumerate(as_completed(futs), 1):
        fn, st = fut.result()
        log(f"[{i}/{len(files)}] {fn}: {st}")
log("SHARD DONE")
PY

echo "[DONE] 日志: ${LOG}"
