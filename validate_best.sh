#!/bin/bash
# Validate the best checkpoint

cd /workspace

CHECKPOINT="checkpoints/epoch-47_miou_67.15.pth"
CONFIG="local_configs.JARVIS.DFormerv2_Base_custom"
GPUS=1
PORT=29158

echo "Running validation on: ${CHECKPOINT}"
echo "Config: ${CONFIG}"

PYTHONPATH=/workspace \
    torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/eval.py \
    --config=${CONFIG} \
    --gpus=$GPUS \
    --continue_fpath=${CHECKPOINT}
