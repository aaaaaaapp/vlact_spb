#!/usr/bin/env bash
# tosutil 公共配置（上传/下载共用）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
[[ -f "${SCRIPT_DIR}/00_env.sh" ]] && source "${SCRIPT_DIR}/00_env.sh" || true

export TOSUTIL="${TOSUTIL:-/mnt/jfzn/liujn/code/Starvla/Starvla/tosutil}"
export TOSUTIL_BASE_CONF="${TOSUTIL_BASE_CONF:-/mnt/jfzn/liujn/.tosutilconfig}"
export TOSUTIL_RUNTIME_CONF="${TOSUTIL_RUNTIME_CONF:-/tmp/vlact-tosutil.conf}"
export TOS_STATE_DIR="${TOS_STATE_DIR:-/tmp/vlact-tos-transfer}"
export TOS_CHECKPOINT_DIR="${TOS_CHECKPOINT_DIR:-/tmp/vlact-tosutil-checkpoint}"
export TOS_RETRY_SECONDS="${TOS_RETRY_SECONDS:-15}"
export TOS_BUCKET_PREFIX="${TOS_BUCKET_PREFIX:-tos://c20250405/ljn/transfer/vlact}"

if [[ ! -x "${TOSUTIL}" ]]; then
  echo "[ERROR] tosutil 不存在: ${TOSUTIL}" >&2
  exit 1
fi
if [[ ! -f "${TOSUTIL_BASE_CONF}" ]]; then
  echo "[ERROR] 缺少配置: ${TOSUTIL_BASE_CONF}" >&2
  exit 1
fi

mkdir -p "${TOS_STATE_DIR}/output" "${TOS_CHECKPOINT_DIR}"
# TOS 国内 endpoint 建议直连；HF 外网才需要 proxy
# 用法: TOSUTIL_DISABLE_PROXY=1 source _tosutil_env.sh
sed -e "s#^utilLogPath=.*#utilLogPath=${TOS_STATE_DIR}/tosutil.log#" \
    -e "s#^sdkLogPath=.*#sdkLogPath=${TOS_STATE_DIR}/tossdk.log#" \
    -e "s#^defaultOutputDir=.*#defaultOutputDir=${TOS_STATE_DIR}/output#" \
    -e "s#^defaultCheckpointDir=.*#defaultCheckpointDir=${TOS_CHECKPOINT_DIR}#" \
    "${TOSUTIL_BASE_CONF}" > "${TOSUTIL_RUNTIME_CONF}"
if [[ "${TOSUTIL_DISABLE_PROXY:-0}" == "1" ]]; then
  sed -i 's|^proxyUrl=.*|proxyUrl=|' "${TOSUTIL_RUNTIME_CONF}"
fi

# 本地目录 -> TOS
tos_upload_dir() {
  local src="$1"
  local dst="$2"
  "${TOSUTIL}" cp "${src}" "${dst}" -r -u -vchecksum \
    -j=20 -nfj=20 -p=20 -tempFileDir=/tmp -conf="${TOSUTIL_RUNTIME_CONF}"
}

# TOS -> 本地目录
tos_download_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "${dst}"
  "${TOSUTIL}" cp "${src}" "${dst}" -r -u -vchecksum \
    -j=20 -nfj=20 -p=20 -tempFileDir=/tmp -conf="${TOSUTIL_RUNTIME_CONF}"
}

tos_retry() {
  local name="$1"
  shift
  local log="${TOS_STATE_DIR}/${name}.log"
  local marker="${TOS_STATE_DIR}/${name}.done"

  if [[ -f "${marker}" ]]; then
    echo "[${name}] already complete"
    return 0
  fi

  while true; do
    echo "[$(date -Is)] [${name}] starting/resuming" | tee -a "${log}"
    if "$@" >>"${log}" 2>&1; then
      date -Is > "${marker}"
      echo "[$(date -Is)] [${name}] complete" | tee -a "${log}"
      return 0
    fi
    echo "[$(date -Is)] [${name}] failed; retry in ${TOS_RETRY_SECONDS}s" | tee -a "${log}"
    sleep "${TOS_RETRY_SECONDS}"
  done
}
