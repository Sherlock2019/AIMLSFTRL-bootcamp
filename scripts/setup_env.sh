#!/usr/bin/env bash
# Creates the local Python environment for the training lab.
# GTX 10xx (Pascal, sm_61) GPUs need the CUDA 12.6 PyTorch build; newer cu128+ builds dropped them.
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
if [ "$(uname)" = "Darwin" ]; then TORCH_INDEX=""; fi
if ! command -v uv >/dev/null; then
  echo "uv not found - using python -m venv + pip"
  python3 -m venv .venv
  PIP=".venv/bin/pip install"
else
  uv venv --python 3.12 .venv
  PIP="uv pip install --python .venv/bin/python"
fi
if [ -n "$TORCH_INDEX" ]; then
  $PIP "torch==${TORCH_VERSION:-2.14.1}" --index-url "$TORCH_INDEX"
else
  $PIP torch
fi
$PIP -r requirements.txt
echo "Environment ready: .venv"
