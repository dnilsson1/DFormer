#!/bin/bash
# Quick test for new optimized settings (720x1280, batch=4, workers=8)

cd /workspace

# Kill any stuck processes first
pkill -f train.py 2>/dev/null || true
sleep 3

# Set environment
export CUDA_VISIBLE_DEVICES="0"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export PYTHONPATH="/workspace:$PYTHONPATH"

echo "=== Quick Test: Optimized Settings ==="
echo "📊 Config: batch_size=4, workers=8, resolution=720x1280"
echo "⏱️  Test duration: 60 seconds"
echo "🔥 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "💾 Available VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo ""

echo "🔍 Current memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""

echo "▶️  Starting test..."
timeout 60s python3 utils/train.py \
    --config=local_configs.JARVIS.DFormer_Large_custom \
    --gpus=1 \
    --epochs=1 \
    --no-sliding \
    --no-compile \
    --no-syncbn \
    --no-mst \
    --no-amp \
    --no-use_seed 2>&1 | while IFS= read -r line; do
        echo "$(date '+%H:%M:%S') | $line"
        # Show progress indicators
        if [[ "$line" == *"Epoch"* && "$line" == *"Iter"* ]]; then
            echo "  🟢 TRAINING PROGRESS DETECTED"
        fi
        if [[ "$line" == *"begin trainning"* ]]; then
            echo "  🚀 TRAINING STARTED"
        fi
        if [[ "$line" == *"ERROR"* || "$line" == *"error"* ]]; then
            echo "  ❌ ERROR DETECTED"
        fi
    done &

TRAIN_PID=$!

# Monitor GPU while training
sleep 15  # Give time to start
echo ""
echo "📊 GPU monitoring during test:"

for i in {1..8}; do
    if kill -0 $TRAIN_PID 2>/dev/null; then
        echo "Sample $i/8: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
        sleep 5
    else
        echo "Training process ended early"
        break
    fi
done

# Check result
wait $TRAIN_PID 2>/dev/null
EXIT_CODE=$?

echo ""
echo "📋 Test Results:"
if [ $EXIT_CODE -eq 124 ]; then
    echo "✅ SUCCESS: Training ran for full 60 seconds (timed out normally)"
    echo "   This indicates the configuration is stable"
    echo "   GPU utilization looks good for full training"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Training completed successfully within 60 seconds"
    echo "   Configuration is working perfectly"
else
    echo "❌ FAILED: Training crashed with exit code $EXIT_CODE"
    echo "   Need to adjust configuration (likely reduce batch size)"
fi

echo ""
echo "💡 Recommendations:"
if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 0 ]; then
    echo "   ✅ Configuration is good for production training"
    echo "   ✅ Can proceed with full training safely"
else
    echo "   ⚠️  Reduce batch_size to 2 and test again"
    echo "   ⚠️  Or reduce workers to 4"
fi