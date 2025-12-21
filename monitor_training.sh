#!/bin/bash
# Monitor training logs in real-time

echo "=== DFormer Training Monitor ==="
echo "Monitoring training logs in real-time..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# Find the most recent log file
LOG_FILE=$(ls -t /workspace/checkpoints/Dformer_format_DFormer-Large/log_*.log 2>/dev/null | head -1)

if [ -f "$LOG_FILE" ]; then
    echo "Monitoring: $LOG_FILE"
    echo "----------------------------------------"
    tail -f "$LOG_FILE"
else
    echo "No training log files found. Start training first."
    echo "Available files:"
    ls -la /workspace/checkpoints/Dformer_format_DFormer-Large/ 2>/dev/null || echo "Checkpoint directory not found"
fi