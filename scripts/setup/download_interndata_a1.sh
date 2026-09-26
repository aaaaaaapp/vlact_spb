#!/usr/bin/env bash
# Download InternData-A1 split_aloha tarballs in parallel (gated, requires HF token).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DST="${REPO_ROOT}/playground/Datasets/InternData-A1_raw"
LOG="${REPO_ROOT}/logs/hf_downloads/interndata_a1_manual.log"
PARALLEL="${INTERNDATA_PARALLEL:-3}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/setup/proxy.env"

if [[ -f "${HOME}/.huggingface/token" ]]; then
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.huggingface/token")"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

mkdir -p "$DST" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}" 2>/dev/null || true
PYTHON="${CONDA_PREFIX}/bin/python"

echo "========== $(date -Is) download_interndata_a1 start parallel=$PARALLEL hf_transfer=$HF_HUB_ENABLE_HF_TRANSFER python=$PYTHON =========="

PYTHONUNBUFFERED=1 "$PYTHON" -u - <<PY
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

parallel = int(os.environ.get("INTERNDATA_PARALLEL", "${PARALLEL}"))
token = os.environ["HF_TOKEN"]
endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
local_dir = Path("${DST}")
repo_id = "InternRobotics/InternData-A1"

def log(msg: str) -> None:
    print(msg, flush=True)

def download_file(filename: str) -> tuple[str, bool, str]:
    target = local_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return filename, True, "skip existing"
    while True:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                token=token,
                endpoint=endpoint,
                local_dir=str(local_dir),
            )
            return filename, True, f"OK {path}"
        except Exception as exc:
            log(f"  retry {filename}: {exc}")
            time.sleep(15)

# Hardcoded fallback avoids API call when proxy returns 503
FALLBACK = [
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

files = []
api = HfApi(token=token, endpoint=endpoint)
for attempt in range(10):
    try:
        files = sorted(
            x.path
            for x in api.list_repo_tree(
                repo_id,
                repo_type="dataset",
                path_in_repo="sim_updated/pick_and_place_tasks/split_aloha",
                recursive=False,
            )
            if str(getattr(x, "path", "")).endswith(".tar.gz")
        )
        break
    except Exception as exc:
        log(f"list_repo_tree attempt {attempt + 1}/10 failed: {exc}")
        time.sleep(30)
if not files:
    log("using hardcoded split_aloha tarball list")
    files = FALLBACK
log(f"Found {len(files)} split_aloha tarballs, parallel={parallel}")

with ThreadPoolExecutor(max_workers=parallel) as pool:
    futures = {pool.submit(download_file, f): f for f in files}
    done = 0
    for fut in as_completed(futures):
        done += 1
        fn, ok, msg = fut.result()
        log(f"[{done}/{len(files)}] {fn}: {msg}")

log("ALL DONE")
PY

echo "========== $(date -Is) download_interndata_a1 done =========="
