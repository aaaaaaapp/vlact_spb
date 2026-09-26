#!/usr/bin/env bash
# Sequential Qwen VLAct speed bench on node4: GAS=4 (20x8x4) then GAS=2 (40x8x2).
# Does not touch node3. Leaves the faster recipe running afterwards.
set -u
REPO="${REPO:-/mnt/jfzn/liujn/code/VLAct-main}"
LOG_DIR="${REPO}/logs/setup"
BENCH_LOG="${LOG_DIR}/qwen_gas_speedbench_node4.log"
RESULT="${LOG_DIR}/qwen_gas_speedbench_node4.result"
mkdir -p "${LOG_DIR}"
touch "${BENCH_LOG}"

klog() { echo "$(date -Is) [qwen-bench] $*" | tee -a "${BENCH_LOG}"; }

kill_node4_qwen() {
  local pids
  pids="$(ps -u liujn -o pid=,cmd= | awk '/\/envs\/vlact\/bin\/python/ && /train_starvla_cotrain.py/ && /vlact_qwen3_node4/ {print $1}')"
  if [[ -n "${pids}" ]]; then
    kill -TERM ${pids} 2>/dev/null || true
    sleep 8
    pids="$(ps -u liujn -o pid=,cmd= | awk '/\/envs\/vlact\/bin\/python/ && /train_starvla_cotrain.py/ && /vlact_qwen3_node4/ {print $1}')"
    [[ -n "${pids}" ]] && kill -KILL ${pids} 2>/dev/null || true
  fi
  local acc sh
  acc="$(ps -u liujn -o pid=,cmd= | awk '/\/envs\/vlact\/bin\/accelerate/ && /train_starvla_cotrain.py/ {print $1}')"
  sh="$(ps -u liujn -o pid=,cmd= | awk '/VLAct-main\/scripts\/run_scripts\/Pretrain\/pretrain_qwen3_single_node.sh/ {print $1}')"
  [[ -n "${acc}${sh}" ]] && kill -TERM ${acc} ${sh} 2>/dev/null || true
  sleep 3
}

wait_gpu_free() {
  local i used
  for i in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')"
    if [[ "${used}" -lt 8000 ]]; then
      return 0
    fi
    sleep 2
  done
  klog "WARN GPUs still using ${used}MiB total"
}

launch_phase() {
  local tag="$1" vla="$2" vlm="$3" gas="$4" steps="$5" port="$6"
  local log="${LOG_DIR}/pretrain_qwen3_node4_${tag}.log"
  : > "${log}"
  klog "launch ${tag}: VLA=${vla} VLM=${vlm} GAS=${gas} steps=${steps} port=${port} log=${log}"
  export VLA_BS="${vla}"
  export VLM_BS="${vlm}"
  export GRAD_ACCUM="${gas}"
  export NUM_PROCESSES=8
  export RUN_ID="vlact_qwen3_node4_${tag}"
  export MAIN_PROCESS_PORT="${port}"
  export MAX_TRAIN_STEPS="${steps}"
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTHONUNBUFFERED=1
  export WANDB_MODE="${WANDB_MODE:-offline}"
  nohup bash -c "stdbuf -oL -eL bash scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh 2>&1 | stdbuf -oL -eL bash scripts/setup/prefix_log.sh >> '${log}'" </dev/null >/dev/null 2>&1 &
  echo $!
}

median_sit() {
  local log="$1"
  python3 - "$log" <<'PY'
import re, statistics, sys
path = sys.argv[1]
vals = []
pat = re.compile(r"\|\s*(\d+)/100000\s+\[[^\]]+,\s*([0-9.]+)s/it")
# also match MAX_TRAIN_STEPS like 22/22
pat2 = re.compile(r"\|\s*(\d+)/\d+\s+\[[^\]]+,\s*([0-9.]+)s/it")
seen = {}
try:
    text = open(path, errors="replace").read()
except FileNotFoundError:
    print("NA")
    raise SystemExit
for patt in (pat, pat2):
    for m in patt.finditer(text):
        step = int(m.group(1))
        sit = float(m.group(2))
        seen[step] = sit
for step, sit in sorted(seen.items()):
    if step >= 4:
        vals.append(sit)
if not vals:
    print("NA")
else:
    print(f"{statistics.median(vals):.2f} n={len(vals)} last={vals[-1]:.2f} min={min(vals):.2f} max={max(vals):.2f}")
PY
}

wait_phase_done() {
  local log="$1" target="$2" timeout_s="$3"
  local i step
  for i in $(seq 1 $((timeout_s / 10))); do
    if ! ps -u liujn -o cmd= | grep -q '[t]rain_starvla_cotrain.py.*vlact_qwen3_node4'; then
      klog "trainer exited for ${log}"
      return 0
    fi
    step="$(grep -oE '[0-9]+/(100000|[0-9]+) \[' "${log}" 2>/dev/null | tail -1 | awk -F/ '{print $1}')"
    if [[ -n "${step}" && "${step}" -ge "${target}" ]]; then
      klog "reached step ${step} >= ${target}"
      return 0
    fi
    sleep 10
  done
  klog "WARN timeout waiting for ${target} steps in ${log}"
  return 1
}

source /mnt/jfzn/liujn/miniconda3/etc/profile.d/conda.sh
conda activate vlact
cd "${REPO}"
# shellcheck source=/dev/null
source "${REPO}/scripts/setup/proxy.env"

klog "start speedbench on $(hostname)"
kill_node4_qwen
wait_gpu_free

launch_phase gas4 20 2 4 22 29522
wait_phase_done "${LOG_DIR}/pretrain_qwen3_node4_gas4.log" 18 2400
GAS4_STAT="$(median_sit "${LOG_DIR}/pretrain_qwen3_node4_gas4.log")"
klog "GAS4 result: ${GAS4_STAT}"
kill_node4_qwen
wait_gpu_free

launch_phase gas2 40 4 2 22 29523
wait_phase_done "${LOG_DIR}/pretrain_qwen3_node4_gas2.log" 18 2400
GAS2_STAT="$(median_sit "${LOG_DIR}/pretrain_qwen3_node4_gas2.log")"
klog "GAS2 result: ${GAS2_STAT}"
kill_node4_qwen
wait_gpu_free

{
  echo "GAS4(20x8x4)=${GAS4_STAT}"
  echo "GAS2(40x8x2)=${GAS2_STAT}"
} | tee "${RESULT}"
klog "wrote ${RESULT}"

# keep the faster one running as a long job (lower median s/it wins; NA loses)
python3 - "${GAS4_STAT}" "${GAS2_STAT}" <<'PY' > /tmp/qwen_bench_winner.env
import sys
def parse(s):
    if not s or s.startswith("NA"):
        return None
    try:
        return float(s.split()[0])
    except Exception:
        return None
a, b = parse(sys.argv[1]), parse(sys.argv[2])
if a is None and b is None:
    print("WINNER=gas4")
    print("VLA_BS=20")
    print("VLM_BS=2")
    print("GRAD_ACCUM=4")
    print("PORT=29522")
elif a is None:
    print("WINNER=gas2")
    print("VLA_BS=40")
    print("VLM_BS=4")
    print("GRAD_ACCUM=2")
    print("PORT=29523")
elif b is None or a <= b:
    print("WINNER=gas4")
    print("VLA_BS=20")
    print("VLM_BS=2")
    print("GRAD_ACCUM=4")
    print("PORT=29522")
else:
    print("WINNER=gas2")
    print("VLA_BS=40")
    print("VLM_BS=4")
    print("GRAD_ACCUM=2")
    print("PORT=29523")
PY
# shellcheck disable=SC1091
source /tmp/qwen_bench_winner.env
klog "winner=${WINNER} VLA=${VLA_BS} GAS=${GRAD_ACCUM}; launching long run"
export VLA_BS VLM_BS GRAD_ACCUM
export NUM_PROCESSES=8
export RUN_ID="vlact_qwen3_pretrain_node4"
export MAIN_PROCESS_PORT="${PORT}"
export MAX_TRAIN_STEPS=100000
export WANDB_MODE="${WANDB_MODE:-offline}"
nohup bash -c "stdbuf -oL -eL bash scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh 2>&1 | stdbuf -oL -eL bash scripts/setup/prefix_log.sh >> '${LOG_DIR}/pretrain_qwen3_node4_long.log'" </dev/null >/dev/null 2>&1 &
klog "long-run pid=$! log=${LOG_DIR}/pretrain_qwen3_node4_long.log"
klog "speedbench finished"
