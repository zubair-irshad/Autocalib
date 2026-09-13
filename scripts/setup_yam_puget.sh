#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv-yam-puget
  uv pip install --python .venv-yam-puget/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  uv pip install --python .venv-yam-puget/bin/python -r requirements-yam.txt
else
  python3.11 -m venv .venv-yam-puget
  .venv-yam-puget/bin/python -m pip install --upgrade pip
  .venv-yam-puget/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  .venv-yam-puget/bin/python -m pip install -r requirements-yam.txt
fi
.venv-yam-puget/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name())'
command -v ffmpeg >/dev/null || { echo 'Install ffmpeg before running: sudo apt install ffmpeg'; exit 1; }

if [[ ! -f third_party/yam/i2rt/assets/base.stl ]]; then
  .venv-yam-puget/bin/python scripts/fetch_yam_models.py
fi
