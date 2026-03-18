#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MODEL_PATH="$HOME/huggingface/Qwen3-0.6B/"

if [[ $# -ge 1 ]]; then
  export NANOVLLM_MODEL="$1"
fi

export NANOVLLM_MODEL="${NANOVLLM_MODEL:-$DEFAULT_MODEL_PATH}"

if [[ -z "${NANOVLLM_MODEL:-}" ]]; then
  echo "Usage: $0 /path/to/model"
  echo "Or set NANOVLLM_MODEL before running."
  exit 1
fi

if [[ ! -d "${NANOVLLM_MODEL}" ]]; then
  echo "Model directory does not exist: ${NANOVLLM_MODEL}"
  exit 1
fi

export NANOVLLM_HOST="${NANOVLLM_HOST:-0.0.0.0}"
export NANOVLLM_PORT="${NANOVLLM_PORT:-8000}"
export NANOVLLM_TP_SIZE="${NANOVLLM_TP_SIZE:-1}"
export NANOVLLM_MAX_BATCHED_TOKENS="${NANOVLLM_MAX_BATCHED_TOKENS:-2048}"
export NANOVLLM_MAX_NUM_SEQS="${NANOVLLM_MAX_NUM_SEQS:-256}"
export NANOVLLM_CHUNKED_PREFILL="${NANOVLLM_CHUNKED_PREFILL:-true}"
export NANOVLLM_ENFORCE_EAGER="${NANOVLLM_ENFORCE_EAGER:-false}"
export NANOVLLM_GPU_IDS="${NANOVLLM_GPU_IDS:-5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$NANOVLLM_GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT_DIR}"

echo "Starting nano-vLLM server"
echo "  model: ${NANOVLLM_MODEL}"
echo "  host: ${NANOVLLM_HOST}"
echo "  port: ${NANOVLLM_PORT}"
echo "  tensor_parallel_size: ${NANOVLLM_TP_SIZE}"
echo "  max_num_batched_tokens: ${NANOVLLM_MAX_BATCHED_TOKENS}"
echo "  max_num_seqs: ${NANOVLLM_MAX_NUM_SEQS}"
echo "  chunked_prefill: ${NANOVLLM_CHUNKED_PREFILL}"
echo "  enforce_eager: ${NANOVLLM_ENFORCE_EAGER}"
echo "  cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
echo "  pytorch_cuda_alloc_conf: ${PYTORCH_CUDA_ALLOC_CONF}"

if command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn server.app:app --host "${NANOVLLM_HOST}" --port "${NANOVLLM_PORT}"
fi

if python -c "import uvicorn" >/dev/null 2>&1; then
  exec python -m uvicorn server.app:app --host "${NANOVLLM_HOST}" --port "${NANOVLLM_PORT}"
fi

echo "uvicorn is not installed in the current environment."
echo "Install dependencies with:"
echo "  pip install -e ."
echo "or:"
echo "  pip install fastapi uvicorn"
exit 1
