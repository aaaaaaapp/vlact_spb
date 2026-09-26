#!/usr/bin/env bash
# Point VLAct-main playground/Datasets at /mnt/jfzn/ftp_upload/vlact (shared storage).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN_DATA="${REPO_ROOT}/playground/Datasets"
FTP_DATA="${FTP_UPLOAD_DATA:-/mnt/jfzn/ftp_upload/vlact/playground/Datasets}"
BACKUP_ROOT="${MAIN_DATA}/.local_backup_$(date +%Y%m%d_%H%M%S)"
LOG="${REPO_ROOT}/logs/setup/link_datasets_from_ftp_upload.log"

mkdir -p "${MAIN_DATA}" "$(dirname "${LOG}")"

exec > >(tee -a "${LOG}") 2>&1
echo "========== $(date -Is) link_datasets_from_ftp_upload start =========="
echo "MAIN_DATA=${MAIN_DATA}"
echo "FTP_DATA=${FTP_DATA}"

if [[ ! -d "${FTP_DATA}" ]]; then
  echo "[ERROR] FTP data root missing: ${FTP_DATA}" >&2
  exit 1
fi

# Keep small/local-only datasets in VLAct-main (not mirrored on ftp).
KEEP_LOCAL=(
  LLaVA-OneVision-COCO
  DROID_100
)

# Large datasets synced via SFTP → ftp_upload.
LINK_FROM_FTP=(
  DROID
  MolmoAct-Dataset
  InternData-A1_raw
  RoboCOIN
  LLaVA-ReCap-CC3M
  RoboInter-Data
)

mkdir -p "${BACKUP_ROOT}"

link_one() {
  local name="$1"
  local src="${FTP_DATA}/${name}"
  local dst="${MAIN_DATA}/${name}"

  if [[ ! -d "${src}" ]]; then
    echo "[SKIP] missing on ftp: ${name}"
    return 0
  fi

  if [[ -L "${dst}" ]]; then
    local cur
    cur="$(readlink -f "${dst}")"
    if [[ "${cur}" == "$(readlink -f "${src}")" ]]; then
      echo "[OK] already linked: ${name} -> ${src}"
      return 0
    fi
    rm -f "${dst}"
  elif [[ -e "${dst}" ]]; then
    echo "[BACKUP] ${name} -> ${BACKUP_ROOT}/"
    mv "${dst}" "${BACKUP_ROOT}/"
  fi

  ln -sfn "${src}" "${dst}"
  echo "[LINK] ${name} -> ${src} ($(du -sh "${src}" | cut -f1))"
}

for name in "${KEEP_LOCAL[@]}"; do
  if [[ -e "${MAIN_DATA}/${name}" ]]; then
    echo "[KEEP] local ${name} ($(du -sh "${MAIN_DATA}/${name}" 2>/dev/null | cut -f1))"
  else
    echo "[WARN] local-only dataset missing: ${name}"
  fi
done

for name in "${LINK_FROM_FTP[@]}"; do
  link_one "${name}"
done

echo ""
echo "========== $(date -Is) link summary =========="
for name in "${LINK_FROM_FTP[@]}" "${KEEP_LOCAL[@]}"; do
  p="${MAIN_DATA}/${name}"
  if [[ -e "${p}" ]]; then
    printf "  %-24s %s -> %s\n" "${name}:" "$(du -sh "${p}" 2>/dev/null | cut -f1)" "$(readlink -f "${p}")"
  fi
done
echo "Backup (if any): ${BACKUP_ROOT}"
echo "Log: ${LOG}"
echo "========== $(date -Is) link_datasets_from_ftp_upload done =========="
