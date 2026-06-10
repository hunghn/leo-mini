#!/bin/bash
# Stage 2 — Full Supervised Fine-tuning
#
# Usage:
#   bash scripts/run_stage2.sh                                   # Llama3-8B (paper default)
#   bash scripts/run_stage2.sh configs/models/llama3_2_1b.yaml  # Llama-3.2-1B
#   bash scripts/run_stage2.sh configs/models/llama3_2_3b.yaml  # Llama-3.2-3B
#   bash scripts/run_stage2.sh configs/models/phi3_5_mini.yaml  # Phi-3.5-mini
#
# WARNING: Stage 2 trains the entire model. On a single 48 GB GPU:
#   - Llama-3.2-1B: standard Adam OK  (~14 GB)
#   - Llama-3.2-3B: 8-bit Adam required  (~27 GB)
#   - Phi-3.5-mini:  8-bit Adam required  (~30 GB)
#   model configs already set optim: adamw_bnb_8bit + gradient_checkpointing: true.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

MODEL_CONFIG="${1:-}"
NUM_GPUS="${NUM_GPUS:-1}"

MODEL_CONFIG_ARG=""
if [ -n "$MODEL_CONFIG" ]; then
  MODEL_CONFIG_ARG="--model_config $MODEL_CONFIG"
  echo "[Stage 2] Using model config: $MODEL_CONFIG"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  torchrun \
    --nproc_per_node "$NUM_GPUS" \
    --master_port 29501 \
    -m src.train.trainer \
    --config "$ROOT/configs/config_stage2.yaml" \
    $MODEL_CONFIG_ARG
else
  python -m src.train.trainer \
    --config "$ROOT/configs/config_stage2.yaml" \
    $MODEL_CONFIG_ARG
fi
