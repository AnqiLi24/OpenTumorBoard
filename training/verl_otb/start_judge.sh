#!/usr/bin/env bash
# Serve the reward judge with vLLM on one GPU.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_SCRATCH=${PROJECT_SCRATCH:-$HERE/outputs/rl}
: "${GPU:?Set GPU explicitly for the judge, for example GPU=1 after that card is released}"
PORT=${PORT:-8094}
# Complete reference discussions can exceed 22k tokens before the
# candidate and rubric are added. The old 16k context was only sufficient for
# conclusion-only judging.
MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN:-65536}
MAX_NUM_SEQS=${JUDGE_MAX_NUM_SEQS:-16}
GPU_MEMORY_UTILIZATION=${JUDGE_GPU_MEMORY_UTILIZATION:-0.90}
WEIGHTS=${JUDGE_WEIGHTS:?set JUDGE_WEIGHTS to a local copy of the judge model}
SERVED_NAME=${OTB_JUDGE_MODEL:-judge}
VLLM_BIN=${JUDGE_VLLM_BIN:-$(command -v vllm)}

if [ ! -x "$VLLM_BIN" ]; then
  echo "Judge vLLM executable not found: $VLLM_BIN" >&2
  exit 2
fi
if [ ! -f "$WEIGHTS/config.json" ]; then
  echo "Judge weights not found: $WEIGHTS" >&2
  exit 2
fi

mkdir -p "$PROJECT_SCRATCH/cache/judge" "$PROJECT_SCRATCH/logs"
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=$PROJECT_SCRATCH/cache/judge/huggingface
export XDG_CACHE_HOME=$PROJECT_SCRATCH/cache/judge/xdg
export VLLM_CACHE_ROOT=$PROJECT_SCRATCH/cache/judge/vllm
export TMPDIR=$PROJECT_SCRATCH/cache/judge/tmp
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"

exec "$VLLM_BIN" serve "$WEIGHTS" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  --generation-config vllm \
  --host 127.0.0.1 \
  --port "$PORT"
