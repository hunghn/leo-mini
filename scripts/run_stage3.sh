#!/bin/bash
# Stage 3 — Token Reduction Fine-tuning (CoTR + MMoE-LLM)
# Estimated time: 8–16 hours on 8×A100 40GB
# This is the core LEO-MINI contribution.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=${NUM_GPUS:-8}

# Optional: start from EAGLE checkpoint instead of Stage 2 output
# Set LLM_PATH to override the path in config_stage3.yaml
if [ -n "${LLM_PATH:-}" ]; then
  sed -i "s|^llm_path:.*|llm_path: \"$LLM_PATH\"|" "$ROOT/configs/config_stage3.yaml"
fi

torchrun \
  --nproc_per_node "$NUM_GPUS" \
  --master_port 29502 \
  "$ROOT/src/train/trainer.py" \
  --config "$ROOT/configs/config_stage3.yaml"
