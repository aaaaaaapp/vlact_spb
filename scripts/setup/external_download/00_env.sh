#!/usr/bin/env bash
# 外部下载环境配置 — source 此文件后再运行其他脚本
# 用法: source scripts/setup/external_download/00_env.sh

# ========== 必填 / 常用 ==========
export STAGING_DIR="${STAGING_DIR:-${HOME}/vlact_external_download}"

# HuggingFace Token（InternData 等 gated 资源）
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.huggingface/token" ]]; then
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.huggingface/token")"
fi

# ========== 下载参数 ==========
# 外部机器通常直连，不要继承集群代理
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

# 国内可开镜像；海外留空走 huggingface.co
export HF_ENDPOINT="${HF_ENDPOINT:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-7200}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-600}"
export MAX_WORKERS="${MAX_WORKERS:-8}"

# ========== 集群同步目标（04_sync_to_cluster.sh 用）==========
export CLUSTER_HOST="${CLUSTER_HOST:-liujn@10.120.154.139}"
export CLUSTER_VLACT_ROOT="${CLUSTER_VLACT_ROOT:-/mnt/jfzn/liujn/code/VLAct-main}"
export CLUSTER_MODEL_ROOT="${CLUSTER_MODEL_ROOT:-/mnt/jfzn/liujn/models/StarVLA}"

# ========== 目录布局（与集群一致）==========
export STAGING_MODELS="${STAGING_DIR}/models/StarVLA"
export STAGING_PRETRAINED="${STAGING_DIR}/playground/Pretrained_models"
export STAGING_DATASETS="${STAGING_DIR}/playground/Datasets"

mkdir -p "${STAGING_MODELS}" "${STAGING_PRETRAINED}" "${STAGING_DATASETS}"

HF_CMD="${HF_CMD:-hf}"
command -v "${HF_CMD}" >/dev/null 2>&1 || HF_CMD="huggingface-cli"
command -v "${HF_CMD}" >/dev/null 2>&1 || { echo "[ERROR] 请先安装: pip install huggingface-hub"; return 1 2>/dev/null || exit 1; }

echo "[env] STAGING_DIR=${STAGING_DIR}"
echo "[env] HF_ENDPOINT=${HF_ENDPOINT:-<default huggingface.co>}"
echo "[env] HF_CMD=${HF_CMD}"
echo "[env] HF_TOKEN=$([[ -n "${HF_TOKEN:-}" ]] && echo set || echo MISSING)"
