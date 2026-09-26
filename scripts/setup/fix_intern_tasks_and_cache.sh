#!/usr/bin/env bash
# Regenerate InternData split_aloha_tasks.txt (with sim_updated/ prefix) and rerun cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/playground/Datasets"
LOG="${REPO_ROOT}/logs/setup/cache_job_interna1_rerun.log"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"
cd "${REPO_ROOT}"

python - <<'PY'
import json
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
if tasks:
    print(f"sample: {tasks[0]}")
    assert tasks[0].startswith("sim_updated/"), f"expected sim_updated/ prefix, got: {tasks[0]}"
MODALITY = {
    "state": {
        "left_joints": {"start": 0, "end": 6, "original_key": "states.left_joint.position"},
        "left_gripper": {"start": 0, "end": 1, "original_key": "states.left_gripper.position"},
        "right_joints": {"start": 0, "end": 6, "original_key": "states.right_joint.position"},
        "right_gripper": {"start": 0, "end": 1, "original_key": "states.right_gripper.position"},
    },
    "action": {
        "left_joints": {"start": 0, "end": 6, "original_key": "actions.left_joint.position"},
        "left_gripper": {"start": 0, "end": 1, "original_key": "actions.left_gripper.position"},
        "right_joints": {"start": 0, "end": 6, "original_key": "actions.right_joint.position"},
        "right_gripper": {"start": 0, "end": 1, "original_key": "actions.right_gripper.position"},
    },
    "video": {
        "rgb_head": {"original_key": "images.rgb.head"},
        "rgb_hand_left": {"original_key": "images.rgb.hand_left"},
        "rgb_hand_right": {"original_key": "images.rgb.hand_right"},
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}
n = 0
for t in tasks:
    meta = root / t / "meta"
    if meta.is_dir():
        (meta / "modality.json").write_text(json.dumps(MODALITY, indent=4) + "\n")
        n += 1
print(f"Wrote {n} modality.json files")
PY

python examples/InternA1/preprocess/compute_caches_split_aloha.py \
  --data-root "${DATA_ROOT}/InternData-A1" \
  --task-list "${DATA_ROOT}/InternData-A1/split_aloha_tasks.txt" \
  --robot-type split_aloha_flip_wrap50 \
  --skip-existing \
  2>&1 | tee "${LOG}"
