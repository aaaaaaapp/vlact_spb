#!/usr/bin/env bash
# Install vlact dependencies with cluster proxy.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="${REPO_ROOT}/logs/setup/pip_install_vlact.log"

source "${CONDA_BASE:-/mnt/jfzn/liujn/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-vlact}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/setup/proxy.env"

export PIP_DEFAULT_TIMEOUT=600
cd "$REPO_ROOT"

exec > >(tee -a "$LOG") 2>&1
echo "========== $(date -Is) install_vlact_deps start proxy=$http_proxy =========="

python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[1/4] Installing PyTorch cu124 wheels via proxy"
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[2/4] Installing requirements.txt"
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[3/4] Installing flash-attn (optional)"
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation \
  -i https://pypi.tuna.tsinghua.edu.cn/simple || echo "WARN: flash-attn failed"

echo "[4/4] Installing starVLA editable"
python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# transformers 4.57 requires huggingface-hub<1.0; do not upgrade to 1.x
python -m pip install "huggingface-hub>=0.34.0,<1.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

python - <<'PY'
import torch, transformers, deepspeed
print("OK torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("OK transformers", transformers.__version__)
print("OK deepspeed", deepspeed.__version__)
PY

echo "========== $(date -Is) install_vlact_deps done =========="
