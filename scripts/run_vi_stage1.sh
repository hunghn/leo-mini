#!/usr/bin/env bash
# Vi-LEO-MINI — Stage 1: Projector Warmup on ViTextVQA
#
# Usage:
#   bash scripts/run_vi_stage1.sh [model_config]
#
# Default model config: configs/models/qwen2_5_3b_vi.yaml
# Override: bash scripts/run_vi_stage1.sh configs/models/my_custom.yaml

set -euo pipefail

MODEL_CONFIG="${1:-configs/models/qwen2_5_3b_vi.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "========================================================"
echo "  Vi-LEO-MINI  |  Stage 1  |  Projector Warmup"
echo "  Model config : ${MODEL_CONFIG}"
echo "  GPUs         : ${NUM_GPUS}"
echo "========================================================"

if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=29500 \
        -m src.train.trainer \
        --config configs/config_vi_stage1.yaml \
        --model_config "${MODEL_CONFIG}"
else
    python -m src.train.trainer \
        --config configs/config_vi_stage1.yaml \
        --model_config "${MODEL_CONFIG}"
fi

echo ""
echo "[Stage 1] Done. Checkpoint saved to checkpoints/qwen2_5_3b_vi/stage1/"
