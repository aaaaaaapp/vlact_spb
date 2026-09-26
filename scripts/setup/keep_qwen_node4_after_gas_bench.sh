#!/usr/bin/env bash
# Watch current node4 Qwen GAS=4, then bench GAS=2, keep the faster one alive.
# Must run ON node4. Does not touch node3 (run_id vlact_qwen3_pretrain).
set -u
REPO=/mnt/jfzn/liujn/code/VLAct-main
LOG_DIR="${REPO}/logs/setup"
STATUS="${LOG_DIR}/qwen_node4_status.txt"
KEEPLOG="${LOG_DIR}/qwen_node4_keeper.log"
GAS4_LOG="${LOG_DIR}/pretrain_qwen3_node4_long.log"
GAS2_LOG="${LOG_DIR}/pretrain_qwen3_node4_gas2.log"
WIN_LOG="${LOG_DIR}/pretrain_qwen3_node4_winner.log"
RESULT="${LOG_DIR}/qwen_gas_speedbench_node4.result"
mkdir -p "${LOG_DIR}"

klog() {
  local msg="$(date -Is) [n4-qwen] $*"
  echo "${msg}" | tee -a "${KEEPLOG}"
  echo "${msg}" >> "${STATUS}"
}

write_status() { printf '%s\n' "$*" > "${STATUS}.tmp" && mv "${STATUS}.tmp" "${STATUS}"; }

node4_qwen_pids() {
  ps -u liujn -o pid=,cmd= | awk '
    /\/envs\/vlact\/bin\/python/ && /train_starvla_cotrain.py/ && /vlact_qwen3_(pretrain_node4|node4_)/ {print $1}
  '
}

node4_launch_pids() {
  ps -u liujn -o pid=,cmd= | awk '
    (/\/envs\/vlact\/bin\/accelerate/ && /train_starvla_cotrain.py/) ||
    (/VLAct-main\/scripts\/run_scripts\/Pretrain\/pretrain_qwen3_single_node.sh/) {print $1}
  '
}

is_running() {
  [[ -n "$(node4_qwen_pids)" ]]
}

kill_node4_qwen() {
  local pids
  pids="$(node4_qwen_pids) $(node4_launch_pids)"
  pids="$(echo ${pids})"
  if [[ -n "${pids}" ]]; then
    klog "kill node4 qwen pids: ${pids}"
    kill -TERM ${pids} 2>/dev/null || true
    sleep 10
    pids="$(node4_qwen_pids) $(node4_launch_pids)"
    pids="$(echo ${pids})"
    [[ -n "${pids}" ]] && kill -KILL ${pids} 2>/dev/null || true
    sleep 4
  fi
}

wait_gpu_quiet() {
  local i used
  for i in $(seq 1 40); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')"
    if [[ "${used}" -lt 12000 ]]; then
      return 0
    fi
    sleep 3
  done
  klog "WARN gpu still ${used}MiB"
}

log_has_oom() {
  local log="$1"
  tail -n 120 "${log}" 2>/dev/null | grep -qE 'out of memory|CUDA out of memory|Triton Error \[CUDA\]: out of memory'
}

latest_step() {
  local log="$1"
  python3 - "$log" <<'PY'
import re, sys
p = sys.argv[1]
try:
    text = open(p, errors="replace").read()
except FileNotFoundError:
    print(0); raise SystemExit
best = 0
# Training tqdm only (has data_times). Dataset stats look like "10/24 [...s/it]eef_..."
for m in re.finditer(r"\|\s*(\d+)/\d+\s+\[[^\]]+,\s*[0-9.]+s/it, data_times=", text):
    best = max(best, int(m.group(1)))
print(best)
PY
}

median_sit() {
  local log="$1"
  python3 - "$log" <<'PY'
import re, statistics, sys
p = sys.argv[1]
seen = {}
try:
    text = open(p, errors="replace").read()
except FileNotFoundError:
    print("NA"); raise SystemExit
for m in re.finditer(r"\|\s*(\d+)/\d+\s+\[[^\]]+,\s*([0-9.]+)s/it, data_times=", text):
    seen[int(m.group(1))] = float(m.group(2))
vals = [seen[s] for s in sorted(seen) if s >= 4]
if not vals:
    print("NA")
else:
    print(f"{statistics.median(vals):.2f} n={len(vals)} last={vals[-1]:.2f} min={min(vals):.2f} max={max(vals):.2f}")
PY
}

parse_median() {
  local s="$1"
  python3 -c "s='$s'; print('nan' if s.startswith('NA') or not s else s.split()[0])"
}

wait_until_step() {
  local log="$1" target="$2" timeout_s="$3" label="$4"
  local i step
  klog "wait ${label}: step>=${target} log=${log} timeout=${timeout_s}s"
  # first wait for ranks to appear (dataset init can take a while)
  for i in $(seq 1 90); do
    if is_running; then
      break
    fi
    if [[ -f "${log}" ]] && log_has_oom "${log}"; then
      klog "${label} OOM before ranks"
      return 2
    fi
    sleep 10
  done
  if ! is_running; then
    klog "${label} never started"
    return 1
  fi
  for i in $(seq 1 $((timeout_s / 15))); do
    if ! is_running; then
      if log_has_oom "${log}"; then
        klog "${label} died OOM"
        return 2
      fi
      step="$(latest_step "${log}")"
      klog "${label} trainer gone at step=${step}"
      [[ "${step}" -ge "${target}" ]] && return 0
      return 1
    fi
    step="$(latest_step "${log}")"
    if [[ "${step}" -ge "${target}" ]]; then
      klog "${label} reached step ${step}"
      return 0
    fi
    if (( i % 8 == 0 )); then
      klog "${label} still waiting step=${step}/${target}"
      write_status "$(date -Is) phase=${label} step=${step}/${target}"
    fi
    sleep 15
  done
  klog "${label} timeout step=$(latest_step "${log}")"
  return 1
}

launch_qwen() {
  local tag="$1" vla="$2" vlm="$3" gas="$4" steps="$5" port="$6" run_id="$7" log="$8"
  klog "launch ${tag}: VLA=${vla} VLM=${vlm} GAS=${gas} steps=${steps} port=${port} run_id=${run_id}"
  : >> "${log}"
  (
    cd "${REPO}"
    source /mnt/jfzn/liujn/miniconda3/etc/profile.d/conda.sh
    conda activate vlact
    # shellcheck source=/dev/null
    source "${REPO}/scripts/setup/proxy.env"
    export VLA_BS="${vla}" VLM_BS="${vlm}" GRAD_ACCUM="${gas}"
    export NUM_PROCESSES=8
    export RUN_ID="${run_id}"
    export MAIN_PROCESS_PORT="${port}"
    export MAX_TRAIN_STEPS="${steps}"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTHONUNBUFFERED=1
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_PROJECT=VLAct_Qwen3_Pretrain
    export WANDB_ENTITY=app123
    nohup bash -c "stdbuf -oL -eL bash scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh 2>&1 | stdbuf -oL -eL bash scripts/setup/prefix_log.sh >> '${log}'" </dev/null >/dev/null 2>&1 &
    echo $!
  )
}

pick_winner() {
  local a_s="$1" b_s="$2"
  python3 - "${a_s}" "${b_s}" <<'PY'
import sys
def parse(s):
    if not s or s.startswith("NA"):
        return None
    try:
        return float(str(s).split()[0])
    except Exception:
        return None
a, b = parse(sys.argv[1]), parse(sys.argv[2])
# lower s/it wins; missing loses
if a is None and b is None:
    print("gas4")
elif a is None:
    print("gas2")
elif b is None:
    print("gas4")
elif b + 0.3 < a:  # gas2 must beat gas4 by >0.3s to switch
    print("gas2")
else:
    print("gas4")
PY
}

# ---------- bench ----------
klog "keeper start on $(hostname); existing GAS4 job will be measured in-place"

if ! is_running; then
  klog "no live GAS4, launching"
  launch_qwen gas4 20 2 4 100000 29522 vlact_qwen3_pretrain_node4 "${GAS4_LOG}"
  sleep 5
fi

if ! wait_until_step "${GAS4_LOG}" 18 3600 gas4; then
  rc=$?
  if [[ "${rc}" -eq 2 ]]; then
    klog "GAS4 OOM unexpected; will still try GAS2 after cleanup"
  fi
fi
GAS4_STAT="$(median_sit "${GAS4_LOG}")"
klog "GAS4(20x8x4) ${GAS4_STAT}"

kill_node4_qwen
wait_gpu_quiet

launch_qwen gas2 40 4 2 25 29523 vlact_qwen3_node4_gas2 "${GAS2_LOG}"
sleep 8
GAS2_RC=0
wait_until_step "${GAS2_LOG}" 18 3600 gas2 || GAS2_RC=$?
GAS2_STAT="$(median_sit "${GAS2_LOG}")"
klog "GAS2(40x8x2) ${GAS2_STAT} rc=${GAS2_RC}"

{
  echo "GAS4(20x8x4)=${GAS4_STAT}"
  echo "GAS2(40x8x2)=${GAS2_STAT}"
} | tee "${RESULT}"

WINNER="$(pick_winner "${GAS4_STAT}" "${GAS2_STAT}")"
if [[ "${GAS2_RC}" -eq 2 ]]; then
  WINNER=gas4
  klog "GAS2 OOM -> force winner=gas4"
fi
klog "winner=${WINNER}"
echo "winner=${WINNER}" >> "${RESULT}"

kill_node4_qwen
wait_gpu_quiet

if [[ "${WINNER}" == "gas2" ]]; then
  WIN_VLA=40; WIN_VLM=4; WIN_GAS=2; WIN_PORT=29523
else
  WIN_VLA=20; WIN_VLM=2; WIN_GAS=4; WIN_PORT=29522
fi

# ---------- long run + restart loop ----------
launch_winner() {
  launch_qwen winner "${WIN_VLA}" "${WIN_VLM}" "${WIN_GAS}" 100000 "${WIN_PORT}" vlact_qwen3_pretrain_node4 "${WIN_LOG}"
}

launch_winner
sleep 15
if ! is_running; then
  klog "winner failed to start, retry once"
  sleep 10
  launch_winner
fi

klog "long-run ${WINNER} VLA=${WIN_VLA} GAS=${WIN_GAS} log=${WIN_LOG}"
write_status "$(date -Is) WINNER=${WINNER} VLA=${WIN_VLA} GAS=${WIN_GAS} running=$(is_running && echo yes || echo no)
GAS4=${GAS4_STAT}
GAS2=${GAS2_STAT}
log=${WIN_LOG}"

while true; do
  sleep 180
  if is_running; then
    continue
  fi
  if tail -n 30 "${WIN_LOG}" 2>/dev/null | grep -qE 'Training completed|max_train_steps reached|100000/100000'; then
    klog "training finished, keeper exit"
    exit 0
  fi
  if log_has_oom "${WIN_LOG}"; then
    klog "winner OOM; fallback to GAS4 if needed"
    if [[ "${WINNER}" == "gas2" ]]; then
      WINNER=gas4; WIN_VLA=20; WIN_VLM=2; WIN_GAS=4; WIN_PORT=29522
    else
      klog "GAS4 also OOM, keeper exit"
      exit 1
    fi
  fi
  klog "winner missing, relaunch ${WINNER}"
  launch_winner
  sleep 20
done
