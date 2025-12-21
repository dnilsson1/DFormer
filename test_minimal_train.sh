#!/bin/bash
# Minimal actual training test

cd /workspace

# Kill any existing processes
pkill -f train.py 2>/dev/null || true
sleep 2

# Set environment for single GPU, non-distributed training  
export CUDA_VISIBLE_DEVICES="0"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
export WORLD_SIZE="1"
export RANK="0"
export LOCAL_RANK="0"

echo "=== Minimal Training Test ==="
echo "⚙️  Configuration: single GPU, no workers, batch_size=2"
echo "⏱️  Will run for maximum 90 seconds then stop"
echo ""

# Set Python path and use timeout to prevent infinite hang
export PYTHONPATH="/workspace:$PYTHONPATH"
timeout 90s python3 utils/train.py \
    --config=local_configs.JARVIS.DFormer_Large_custom \
    --gpus=1 \
    --epochs=1 \
    --no-sliding \
    --no-compile \
    --no-syncbn \
    --no-mst \
    --compile_mode="default" \
    --no-amp \
    --no-val_amp \
    --no-use_seed 2>&1 | head -50

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 124 ]; then
    echo "⚠️  Training timed out after 90 seconds"
    echo "🔍 This suggests the training is stuck in the data loading loop"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"  
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
fi

echo ""
echo "🔍 Checking for any remaining processes..."
ps aux | grep train | grep -v grep || echo "No training processes running"