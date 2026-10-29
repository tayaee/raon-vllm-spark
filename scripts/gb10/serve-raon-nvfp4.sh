#!/usr/bin/env bash
# Serve the Raon-1-mini base model from REMOTE (Hugging Face) on local port 8000.
# Verification purpose: check that THIS repo state loads/serves the base model
# before making any changes.
#
# GB10 default is the NVFP4 checkpoint: the MXFP8 base needs sm_100-family
# (B300) kernels and is EXPECTED TO FAIL on GB10/sm_121. That failure is a
# hardware gate, not repo breakage.
#
#   ./scripts/gb10/serve-raon-nvfp4.sh                                              # NVFP4 (GB10-servable)
#   MODEL_ID=KRAFTON/Raon-1-mini-preview ./scripts/gb10/serve-raon-nvfp4.sh         # MXFP8 base (B300-only)
#   REVISION="" ./scripts/gb10/serve-raon-nvfp4.sh                                  # remote main tip
#   SPEC_TOKENS=7 ./scripts/gb10/serve-raon-nvfp4.sh                                # + DFlash draft
#   HOST=0.0.0.0 ./scripts/gb10/serve-raon-nvfp4.sh "$@"                            # extra vllm args
#
# Pinned revisions (see DGX Spark recipe):
#   MXFP8 base : KRAFTON/Raon-1-mini-preview          da4cf162c8613993f2189ebb438a5f9a23f3e5ca
#   NVFP4      : KRAFTON/Raon-1-mini-preview-nvfp4     0b0f06ac2b3d03abea62dd221fe7839fddf36dfe
#   DFlash2    : KRAFTON/Raon-1-mini-preview-dflash2   6450d1546d17ee1bb1cb942f316c7fb91cdd868a
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL_ID="${MODEL_ID:-KRAFTON/Raon-1-mini-preview-nvfp4}"
# Pin revisions per known model; unknown models follow main tip.
# NOTE: ${REVISION+x} distinguishes unset (auto-pin) from REVISION="" (main tip),
# because :- treats empty as unset.
if [[ -z "${REVISION+x}" ]]; then
  case "$MODEL_ID" in
    KRAFTON/Raon-1-mini-preview-nvfp4) REVISION="0b0f06ac2b3d03abea62dd221fe7839fddf36dfe" ;;
    KRAFTON/Raon-1-mini-preview) REVISION="da4cf162c8613993f2189ebb438a5f9a23f3e5ca" ;;
    *) REVISION="" ;;
  esac
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# GB10 (sm_121) JIT targets; harmless toolchain hint, not a model workaround.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ ! -x .venv/bin/python ]]; then
  echo "missing .venv — run ./scripts/gb10/build.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

args=(
  serve "$MODEL_ID"
  --host "$HOST" --port "$PORT"
  --served-model-name Raon-1-mini-preview
  --trust-remote-code
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}"
  --max-num-seqs "${MAX_NUM_SEQS:-8}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}"
  --kv-cache-dtype fp8_e4m3
  --kv-cache-memory "${KV_CACHE_BYTES:-3221225472}"
  --enable-prefix-caching --async-scheduling
  --reasoning-parser raon_1
  --enable-auto-tool-choice --tool-call-parser raon_1
  --default-chat-template-kwargs '{"enable_thinking":true,"truncate_history_thinking":false}'
)
if [[ -n "$REVISION" ]]; then
  args+=(--revision "$REVISION")
fi
if [[ "${SPEC_TOKENS:-0}" != "0" ]]; then
  args+=(
    --spec-method dflash
    --spec-model "${DRAFT_MODEL_ID:-KRAFTON/Raon-1-mini-preview-dflash2}"
    --spec-tokens "$SPEC_TOKENS"
  )
fi

echo "==> MODEL_ID=$MODEL_ID REVISION=${REVISION:-<main>} ${HOST}:${PORT}"
exec .venv/bin/vllm "${args[@]}" "$@"
