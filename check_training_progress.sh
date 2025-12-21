#!/bin/bash
# Check what's happening after "begin trainning:"

cd /workspace

echo "=== Training Progress Diagnostic ==="
echo "🔍 Checking if training is actually progressing..."
echo ""

# Check if any training processes are running
TRAIN_PROCS=$(ps aux | grep "python.*train.py" | grep -v grep | wc -l)
echo "📊 Active training processes: $TRAIN_PROCS"

if [ $TRAIN_PROCS -gt 0 ]; then
    echo "🔄 Training processes found:"
    ps aux | grep "python.*train.py" | grep -v grep
    echo ""
    
    # Check GPU activity
    echo "🔥 Current GPU status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
    
    # Check if log files are being written
    echo "📝 Recent log activity:"
    LOG_FILE=$(ls -t /workspace/checkpoints/Dformer_format_DFormer-Large/log_*.log 2>/dev/null | head -1)
    
    if [ -f "$LOG_FILE" ]; then
        echo "Latest log: $(basename "$LOG_FILE")"
        echo "File size: $(du -h "$LOG_FILE" | cut -f1)"
        echo "Last modified: $(stat -c %y "$LOG_FILE")"
        echo ""
        echo "Last 5 lines:"
        tail -5 "$LOG_FILE"
        echo ""
        
        # Check if file is still growing
        INITIAL_SIZE=$(stat -c %s "$LOG_FILE")
        sleep 5
        CURRENT_SIZE=$(stat -c %s "$LOG_FILE")
        
        if [ $CURRENT_SIZE -gt $INITIAL_SIZE ]; then
            echo "✅ Log file is growing - training is progressing!"
            echo "   Growth: $((CURRENT_SIZE - INITIAL_SIZE)) bytes in 5 seconds"
        else
            echo "⚠️  Log file not growing - might be stuck"
        fi
    else
        echo "❌ No log files found"
    fi
else
    echo "❌ No training processes running"
fi

echo ""
echo "💡 What to expect after 'begin trainning:':"
echo "   • 15-30 seconds: DataLoader workers starting up"
echo "   • 30-60 seconds: First batch preprocessing and loading"
echo "   • 60+ seconds: First forward pass (large model + high resolution)"
echo "   • 90+ seconds: Should see first 'Epoch 1/500 Iter 1/81' message"
echo ""
echo "🚨 If stuck longer than 2 minutes, training is likely hung"