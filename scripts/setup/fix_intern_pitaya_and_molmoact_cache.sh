#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASKS="${REPO_ROOT}/playground/Datasets/InternData-A1/split_aloha_tasks.txt"
LOG_DIR="${REPO_ROOT}/logs/setup"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"
cd "${REPO_ROOT}"

# Drop tasks missing episodes.jsonl (incomplete upload/extract).
python - <<'PY'
from pathlib import Path
tasks_file = Path("playground/Datasets/InternData-A1/split_aloha_tasks.txt")
root = Path("playground/Datasets/InternData-A1")
kept, dropped = [], []
for line in tasks_file.read_text().splitlines():
    t = line.strip()
    if not t:
        continue
    ep = root / t / "meta" / "episodes.jsonl"
    if ep.exists():
        kept.append(t)
    else:
        dropped.append(t)
tasks_file.write_text("\n".join(kept) + ("\n" if kept else ""))
print(f"tasks kept={len(kept)} dropped={len(dropped)}")
for d in dropped:
    print(f"  dropped: {d}")
PY

if [[ -f "${REPO_ROOT}/playground/Datasets/MolmoAct-Dataset/molmoact_household_v21/meta/info.json" ]]; then
  echo "[MolmoAct] starting cache"
  python examples/MolmoAct/compute_caches_molmoact_delta_eef.py \
    --data-root "${REPO_ROOT}/playground/Datasets/MolmoAct-Dataset" --mode filtered \
    --robot-types \
      molmoact_franka_exterior1_wrist_manualvel_strict_50 \
      molmoact_franka_exterior2_wrist_manualvel_strict_50 \
    --filter-delta-eef-steps --skip-existing \
    >> "${LOG_DIR}/cache_job_molmoact.log" 2>&1
  echo "[MolmoAct] cache done"
else
  echo "[MolmoAct] meta/info.json still missing" >&2
  exit 1
fi
