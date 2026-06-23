#!/usr/bin/env bash
# Vi-LEO-MINI — Stage 2: Full SFT on ViTextVQA
#
# Requires Stage 1 to have completed (trainer will exit(1) if checkpoint missing).
#
# Usage:
#   bash scripts/run_vi_stage2.sh [model_config]

set -euo pipefail

MODEL_CONFIG="${1:-configs/models/qwen2_5_3b_vi.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "========================================================"
echo "  Vi-LEO-MINI  |  Stage 2  |  Full SFT"
echo "  Model config : ${MODEL_CONFIG}"
echo "  GPUs         : ${NUM_GPUS}"
echo "========================================================"

if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=29501 \
        -m src.train.trainer \
        --config configs/config_vi_stage2.yaml \
        --model_config "${MODEL_CONFIG}"
else
    python -m src.train.trainer \
        --config configs/config_vi_stage2.yaml \
        --model_config "${MODEL_CONFIG}"
fi

echo ""
echo "[Stage 2] Done. Checkpoint saved to checkpoints/qwen2_5_3b_vi/stage2/"
