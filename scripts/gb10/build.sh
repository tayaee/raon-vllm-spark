#!/usr/bin/env bash
# Build vLLM from this repo (GB10 / aarch64 / CUDA 13.0).
# Follows AGENTS.md: never system python3/pip, everything via uv + .venv.
#
#   ./build.sh          # Python-only build (precompiled kernels, fast)
#   FULL=1 ./build.sh   # Full C++/CUDA source build (needed for csrc changes)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# 1. venv (python 3.12 per AGENTS.md)
if [[ ! -d .venv ]]; then
  uv venv --python 3.12
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 2. install / build
if [[ "${FULL:-0}" == "1" ]]; then
  echo "==> full source build (C++/CUDA, slow on first run)"
  uv pip install -e . --torch-backend=auto
else
  echo "==> precompiled-kernel build (Python changes only, fast)"
  echo "    uses VLLM_PRECOMPILED_WHEEL_COMMIT default pinned in setup.py"
  VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
fi

# 3. smoke test
.venv/bin/python -c "
import torch, vllm
print('vllm:', vllm.__version__)
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
from vllm.model_executor.models.registry import ModelRegistry
print('raon registered:', 'Raon1MiniForCausalLM' in ModelRegistry.get_supported_archs())
"
echo "build OK: use .venv/bin/vllm ... or ./serve-raon.sh"
