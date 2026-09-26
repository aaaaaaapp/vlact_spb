#!/usr/bin/env bash
# Official VLAct conda environment setup (README.md Installation section)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}"
ENV_NAME="${ENV_NAME:-vlact}"
LOG_DIR="${REPO_ROOT}/logs/setup"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/create_conda_env.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== $(date -Is) create_conda_env start =========="
echo "REPO_ROOT=$REPO_ROOT"
echo "CONDA_BASE=$CONDA_BASE"
echo "ENV_NAME=$ENV_NAME"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/setup/proxy.env"

CONDA_MIRROR="${CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[1/5] Creating conda env: $ENV_NAME (python=3.10) via mirror"
  conda create -n "$ENV_NAME" python=3.10 -y \
    --override-channels \
    -c "$CONDA_MIRROR" \
    -c "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r"
else
  echo "[1/5] Conda env $ENV_NAME already exists"
fi

conda activate "$ENV_NAME"

PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export http_proxy="${http_proxy:-http://10.119.18.5:3128}"
export https_proxy="${https_proxy:-http://10.119.18.5:3128}"

echo "[2/5] Installing PyTorch (CUDA 12.4 wheels)"
python -m pip install --upgrade pip setuptools wheel -i "$PIP_INDEX"
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] Installing requirements.txt"
python -m pip install -r "${REPO_ROOT}/requirements.txt" -i "$PIP_INDEX"

echo "[4/5] Installing flash-attn (may take several minutes)"
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation || {
  echo "WARN: flash-attn build failed; training may still work without it on some configs."
}

echo "[5/5] Installing starVLA package (editable)"
cd "$REPO_ROOT"
python -m pip install -e .

echo "[check] Verifying imports"
python - <<'PY'
import torch
import transformers
import accelerate
import deepspeed
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("deepspeed", deepspeed.__version__)
PY

python -m pip install -U "huggingface-hub[cli]"

echo "========== $(date -Is) create_conda_env done =========="
