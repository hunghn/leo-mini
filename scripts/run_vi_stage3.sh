#!/usr/bin/env bash
# Vi-LEO-MINI — Stage 3: Token Reduction (CoTR + MMoE-LLM)
#
# Requires Stage 2 to have completed (trainer will exit(1) if checkpoint missing).
#
# Usage:
#   bash scripts/run_vi_stage3.sh [model_config]

set -euo pipefail

MODEL_CONFIG="${1:-configs/models/qwen2_5_3b_vi.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "========================================================"
echo "  Vi-LEO-MINI  |  Stage 3  |  Token Reduction"
echo "  Model config : ${MODEL_CONFIG}"
echo "  GPUs         : ${NUM_GPUS}"
echo "========================================================"

if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=29502 \
        -m src.train.trainer \
        --config configs/config_vi_stage3.yaml \
        --model_config "${MODEL_CONFIG}"
else
    python -m src.train.trainer \
        --config configs/config_vi_stage3.yaml \
        --model_config "${MODEL_CONFIG}"
fi

echo ""
echo "[Stage 3] Done. Adapter weights saved to checkpoints/qwen2_5_3b_vi/stage3/"
echo ""
echo "Run evaluation:"
echo "  python -m src.eval.vi_evaluator \\"
echo "    --model_path Qwen/Qwen2.5-3B-Instruct \\"
echo "    --stage3_weights checkpoints/qwen2_5_3b_vi/stage3/stage3_adapter_weights.pt \\"
echo "    --split test \\"
echo "    --output_dir results/vi_leomini/"
