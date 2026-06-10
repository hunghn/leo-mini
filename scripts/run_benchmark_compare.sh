#!/bin/bash
# Compare LEO-MINI across backbone models (Llama-3.2-1B, 3B, Phi-3.5-mini).
#
# Each model requires TWO pieces of information:
#   BASE_*  — HuggingFace model ID (or full checkpoint dir) for the base LLM.
#             Passed to AutoModelForCausalLM.from_pretrained().
#   CKPT_*  — Path to stage3_adapter_weights.pt produced by
#             LeoMini.save_stage3_weights() at the end of Stage 3.
#             Leave empty to evaluate the base LLM without Stage-3 adapters
#             (useful for a Stage-2 baseline comparison).
#
# Usage examples:
# ─────────────────────────────────────────────────────────────────────────────
# Full LEO-MINI evaluation (all three models):
#   BASE_1B="meta-llama/Llama-3.2-1B-Instruct" \
#   CKPT_1B="checkpoints/llama3_2_1b/stage3/stage3_adapter_weights.pt" \
#   BASE_3B="meta-llama/Llama-3.2-3B-Instruct" \
#   CKPT_3B="checkpoints/llama3_2_3b/stage3/stage3_adapter_weights.pt" \
#   BASE_PHI="microsoft/Phi-3.5-mini-instruct" \
#   CKPT_PHI="checkpoints/phi3_5_mini/stage3/stage3_adapter_weights.pt" \
#   bash scripts/run_benchmark_compare.sh
#
# Quick sanity check (200 samples/task, 4-bit, 3B only):
#   BASE_3B="meta-llama/Llama-3.2-3B-Instruct" \
#   CKPT_3B="checkpoints/llama3_2_3b/stage3/stage3_adapter_weights.pt" \
#   TASKS="pope,scienceqa,textvqa" LIMIT=200 LOAD_4BIT=1 \
#   bash scripts/run_benchmark_compare.sh
#
# Stage-2 baseline (no adapters, just the fine-tuned LLM):
#   BASE_3B="checkpoints/llama3_2_3b/stage2/checkpoint-last" \
#   bash scripts/run_benchmark_compare.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# ── Per-model env vars ────────────────────────────────────────────────────────
# Llama-3.2-1B
BASE_1B="${BASE_1B:-}"
CKPT_1B="${CKPT_1B:-}"
# Llama-3.2-3B
BASE_3B="${BASE_3B:-}"
CKPT_3B="${CKPT_3B:-}"
# Phi-3.5-mini
BASE_PHI="${BASE_PHI:-}"
CKPT_PHI="${CKPT_PHI:-}"

# ── Evaluation settings ───────────────────────────────────────────────────────
TASKS="${TASKS:-pope,scienceqa,textvqa,mme,mmmu}"
LIMIT="${LIMIT:-}"
LOAD_4BIT="${LOAD_4BIT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/comparison}"

# ── Build --model_specs string ───────────────────────────────────────────────
# Format: "Name:base_path:stage3_weights" (stage3_weights may be empty → 2-part)
SPECS=""

_add_model() {
  local label="$1" base="$2" ckpt="$3"
  [ -z "$base" ] && return  # skip if base path not provided
  if [ -n "$ckpt" ]; then
    SPECS="${SPECS:+$SPECS,}${label}:${base}:${ckpt}"
  else
    SPECS="${SPECS:+$SPECS,}${label}:${base}"
  fi
}

_add_model "Llama-3.2-1B" "$BASE_1B" "$CKPT_1B"
_add_model "Llama-3.2-3B" "$BASE_3B" "$CKPT_3B"
_add_model "Phi-3.5-mini"  "$BASE_PHI" "$CKPT_PHI"

if [ -z "$SPECS" ]; then
  echo "ERROR: No models specified."
  echo ""
  echo "Set at least one BASE_* env var, e.g.:"
  echo "  BASE_3B=\"meta-llama/Llama-3.2-3B-Instruct\" \\"
  echo "  CKPT_3B=\"checkpoints/llama3_2_3b/stage3/stage3_adapter_weights.pt\" \\"
  echo "  bash $0"
  exit 1
fi

echo "============================================================"
echo "  LEO-MINI Benchmark Comparison"
echo "  Tasks:      $TASKS"
echo "  Models:     $SPECS"
echo "  Output dir: $OUTPUT_DIR"
[ -n "$LIMIT" ]        && echo "  Limit:      $LIMIT samples/task"
[ "$LOAD_4BIT" = "1" ] && echo "  Quant:      4-bit"
echo "============================================================"

# ── Build and run command ────────────────────────────────────────────────────
CMD=(python -m src.eval.evaluator
  --compare
  --model_specs "$SPECS"
  --tasks       "$TASKS"
  --output_dir  "$OUTPUT_DIR"
)

[ -n "$LIMIT"       ] && CMD+=(--limit "$LIMIT")
[ "$LOAD_4BIT" = "1" ] && CMD+=(--load_in_4bit)

"${CMD[@]}"
