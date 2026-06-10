#!/bin/bash
# Stage 3 — Token Reduction Fine-tuning (CoTR + MMoE-LLM)
# This is the core LEO-MINI contribution (paper Section 3.2–3.3).
#
# Usage:
#   bash scripts/run_stage3.sh                                   # Llama3-8B (paper default)
#   bash scripts/run_stage3.sh configs/models/llama3_2_1b.yaml  # Llama-3.2-1B
#   bash scripts/run_stage3.sh configs/models/llama3_2_3b.yaml  # Llama-3.2-3B
#   bash scripts/run_stage3.sh configs/models/phi3_5_mini.yaml  # Phi-3.5-mini
#
# Stage 3 freezes LLM + vision experts; only CoTR + MMoE-LLM adapters are
# trained (~100–200 M params). VRAM usage is low for all model sizes.
# Estimated time: 2–3 h (1B) / 5–7 h (3B/Phi) on 1×48 GB GPU.

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
  echo "[Stage 3] Using model config: $MODEL_CONFIG"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  torchrun \
    --nproc_per_node "$NUM_GPUS" \
    --master_port 29502 \
    -m src.train.trainer \
    --config "$ROOT/configs/config_stage3.yaml" \
    $MODEL_CONFIG_ARG
else
  python -m src.train.trainer \
    --config "$ROOT/configs/config_stage3.yaml" \
    $MODEL_CONFIG_ARG
fi
