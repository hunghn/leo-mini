#!/bin/bash
# Stage 1 — Warmup Visual Projector
#
# Usage:
#   bash scripts/run_stage1.sh                                   # Llama3-8B (paper default)
#   bash scripts/run_stage1.sh configs/models/llama3_2_1b.yaml  # Llama-3.2-1B
#   bash scripts/run_stage1.sh configs/models/llama3_2_3b.yaml  # Llama-3.2-3B
#   bash scripts/run_stage1.sh configs/models/phi3_5_mini.yaml  # Phi-3.5-mini
#
# Set NUM_GPUS=1 for a single 48 GB GPU (A6000 / A100).
# Estimated time: 4–8 h on 8×A100 | 1–2 h (1B) / 5–7 h (3B/Phi) on 1×48GB.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

MODEL_CONFIG="${1:-}"          # optional first argument
NUM_GPUS="${NUM_GPUS:-1}"      # default to 1 GPU for single-machine setup

MODEL_CONFIG_ARG=""
if [ -n "$MODEL_CONFIG" ]; then
  MODEL_CONFIG_ARG="--model_config $MODEL_CONFIG"
  echo "[Stage 1] Using model config: $MODEL_CONFIG"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  torchrun \
    --nproc_per_node "$NUM_GPUS" \
    --master_port 29500 \
    -m src.train.trainer \
    --config "$ROOT/configs/config_stage1.yaml" \
    $MODEL_CONFIG_ARG
else
  python -m src.train.trainer \
    --config "$ROOT/configs/config_stage1.yaml" \
    $MODEL_CONFIG_ARG
fi
