#!/bin/bash
# Stage 2 — Full Supervised Fine-tuning
# Estimated time: 1–2 days on 8×A100 40GB

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=${NUM_GPUS:-8}

torchrun \
  --nproc_per_node "$NUM_GPUS" \
  --master_port 29501 \
  "$ROOT/src/train/trainer.py" \
  --config "$ROOT/configs/config_stage2.yaml"
