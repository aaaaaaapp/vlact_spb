#!/usr/bin/env bash
# 按名称下载单个数据集（支持 --include 部分下载）
# 用法: bash 02_download_one.sh llava_coco
#       bash 02_download_one.sh --group vlm
#       bash 02_download_one.sh --list
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_env.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/datasets.conf.sh"

LIST=0
GROUP=""
NAMES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST=1; shift ;;
    --group) GROUP="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: bash 02_download_one.sh [--list] [--group models|vlm|vla_core|vla_optional|robocoin] [name ...]"
      exit 0
      ;;
    *) NAMES+=("$1"); shift ;;
  esac
done

if (( LIST )); then
  echo "=== 数据集清单（name | include）==="
  for entry in "${VLACT_DATASET_ENTRIES[@]}"; do
    IFS='|' read -r name repo rtype relpath includes <<< "${entry}"
    echo "${name}"
    echo "  repo: ${repo} (${rtype})"
    echo "  path: ${STAGING_DIR}/${relpath}"
    if [[ -n "${includes}" ]]; then
      echo "  include: ${includes//,/, }"
    else
      echo "  include: (全量)"
    fi
    echo
  done
  echo "InternData-A1: 用 03_download_internedata.sh（10 个 split_aloha tar.gz）"
  exit 0
fi

if [[ -n "${GROUP}" ]]; then
  mapfile -t NAMES < <(resolve_dataset_group "${GROUP}")
  if (( ${#NAMES[@]} == 0 )); then
    echo "未知 group: ${GROUP}" >&2
    exit 1
  fi
fi

if (( ${#NAMES[@]} == 0 )); then
  echo "请指定数据集名，或 --group / --list" >&2
  exit 1
fi

download_entry() {
  local entry="$1"
  local name repo rtype relpath includes
  IFS='|' read -r name repo rtype relpath includes <<< "${entry}"
  local target="${STAGING_DIR}/${relpath}"
  local -a extra=()

  if [[ -n "${includes}" ]]; then
    local IFS=','
    for glob in ${includes}; do
      extra+=(--include "${glob}")
    done
  fi

  echo ">>> ${name}: ${repo} -> ${target}"
  [[ ${#extra[@]} -gt 0 ]] && echo "    include: ${includes}"
  bash "${SCRIPT_DIR}/_hf_download.sh" "${name}" "${repo}" "${target}" "${rtype}" "${extra[@]}"
}

for name in "${NAMES[@]}"; do
  entry="$(dataset_entry_by_name "${name}")" || {
    echo "未知数据集: ${name}（bash 02_download_one.sh --list 查看）" >&2
    exit 1
  }
  download_entry "${entry}"
done
