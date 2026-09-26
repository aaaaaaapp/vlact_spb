#!/usr/bin/env bash
# TOS 拉取 VLAct PI 权重 → 就绪后自动跑 RobotWin smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP="${ROOT}/scripts/setup"
CKPT="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
LOG="${ROOT}/logs/setup/pull_robotwin_and_smoke.log"
mkdir -p "$(dirname "$LOG")"

exec >>"$LOG" 2>&1
echo "========== $(date -Is) pull_robotwin_and_smoke start =========="

# 可选：外部上传路径与脚本默认不一致时设置
# export TOS_ROBOTWIN_CKPT="tos://c20250405/ljn/transfer/xxx/steps_100000_pytorch_model.pt"

if [[ -n "${TOS_ROBOTWIN_CKPT:-}" ]]; then
  # shellcheck source=/dev/null
  source "${SETUP}/external_download/_tosutil_env.sh"
  mkdir -p "$(dirname "$CKPT")"
  tos_retry robotwin_ckpt_direct \
    "${TOSUTIL}" cp "${TOS_ROBOTWIN_CKPT}" "${CKPT}" -u -conf="${TOSUTIL_RUNTIME_CONF}"
else
  bash "${SETUP}/external_download/06_download_from_tos.sh" --only robotwin
fi

if [[ ! -f "${CKPT}" ]] || (( $(stat -c%s "${CKPT}") < 10000000000 )); then
  echo "[ERROR] checkpoint not ready: ${CKPT}" >&2
  ls -lh "${CKPT}" 2>/dev/null || true
  exit 1
fi

echo "[OK] checkpoint ready: $(ls -lh "${CKPT}")"
bash "${SETUP}/run_robotwin_smoke_vlact_pi.sh"
echo "========== $(date -Is) pull_robotwin_and_smoke done =========="
