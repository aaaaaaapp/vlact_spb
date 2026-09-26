#!/usr/bin/env bash
# 将外部下载目录 rsync 到集群（默认 dry-run，加 --execute 才真正传输）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_env.sh"

DRY_RUN=1
ONLY="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --only) ONLY="${2:-all}"; shift 2 ;;
    --only=*) ONLY="${1#*=}"; shift ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

RSYNC=(rsync -avh --progress --partial)
(( DRY_RUN )) && RSYNC+=(--dry-run)

remote() {
  local src="$1"
  local dst="$2"
  [[ -d "${src}" ]] || { echo "[SKIP] 目录不存在: ${src}"; return 0; }
  echo ""
  echo ">>> ${src} -> ${CLUSTER_HOST}:${dst}"
  "${RSYNC[@]}" "${src}" "${CLUSTER_HOST}:${dst}"
}

echo "STAGING_DIR=${STAGING_DIR}"
echo "CLUSTER_HOST=${CLUSTER_HOST}"
echo "MODE=$([[ ${DRY_RUN} -eq 1 ]] && echo dry-run || echo EXECUTE)"
echo "ONLY=${ONLY}"

case "${ONLY}" in
  robotwin|all)
    remote "${STAGING_MODELS}/VLAct_Qwen3PI_Robotwin_all_Finetune/" \
      "${CLUSTER_MODEL_ROOT}/VLAct_Qwen3PI_Robotwin_all_Finetune/"
    ;;&
  pretrain|all)
    remote "${STAGING_PRETRAINED}/" \
      "${CLUSTER_VLACT_ROOT}/playground/Pretrained_models/"
    remote "${STAGING_DATASETS}/" \
      "${CLUSTER_VLACT_ROOT}/playground/Datasets/"
    ;;
  *)
    echo "未知 --only 值: ${ONLY}（可选: robotwin, pretrain, all）" >&2
    exit 1
    ;;
esac

echo ""
if (( DRY_RUN )); then
  echo "以上为预览。确认无误后执行:"
  echo "  bash $0 --execute --only ${ONLY}"
else
  echo "[DONE] 同步完成。集群上可运行:"
  echo "  bash ${CLUSTER_VLACT_ROOT}/scripts/setup/run_robotwin_smoke_vlact_pi.sh"
fi
