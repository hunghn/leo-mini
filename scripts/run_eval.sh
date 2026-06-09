#!/bin/bash
# Full benchmark evaluation on GPU server (12 benchmarks from paper)
# Usage:
#   bash scripts/run_eval.sh [--model_path PATH] [--tasks TASKS] [--limit N]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$ROOT:$PYTHONPATH"

MODEL_PATH=${MODEL_PATH:-"checkpoints/stage3/stage3_adapter_weights.pt"}
TASKS=${TASKS:-"mme,mmbench,seedbench,gqa,scienceqa,mmmu,pope,ai2d,textvqa,chartqa,ocrbench"}
LIMIT=${LIMIT:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"results/full_eval"}

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

python -m src.eval.evaluator \
  --model_path "$MODEL_PATH" \
  --tasks      "$TASKS" \
  --output_dir "$OUTPUT_DIR" \
  $LIMIT_ARG

echo ""
echo "Evaluation complete. Results in $OUTPUT_DIR/results.json"
