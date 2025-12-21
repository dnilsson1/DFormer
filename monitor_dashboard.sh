#!/bin/bash
# Training progress dashboard

echo "=== DFormer Training Dashboard ==="

# Function to extract training metrics
extract_metrics() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        echo "📈 Recent Training Metrics:"
        
        # Extract loss values
        echo "   Latest Loss Values:"
        grep -E "(loss|Loss)" "$log_file" | tail -5 | sed 's/^/     /'
        
        echo ""
        
        # Extract accuracy/mIoU if available
        echo "   Latest Accuracy/mIoU:"
        grep -E "(accuracy|mIoU|Accuracy)" "$log_file" | tail -3 | sed 's/^/     /'
        
        echo ""
        
        # Extract epoch information
        echo "   Epoch Progress:"
        grep -E "epoch|Epoch" "$log_file" | tail -3 | sed 's/^/     /'
        
    else
        echo "❌ No log file found: $log_file"
    fi
}

# Main monitoring loop
while true; do
    clear
    echo "=== DFormer Training Dashboard - $(date) ==="
    echo ""
    
    # Find latest log file
    LOG_FILE=$(ls -t /workspace/checkpoints/Dformer_format_DFormer-Large/log_*.log 2>/dev/null | head -1)
    
    if [ -f "$LOG_FILE" ]; then
        echo "📄 Current Log: $(basename "$LOG_FILE")"
        echo "📊 File Size: $(du -h "$LOG_FILE" | cut -f1)"
        echo "⏱️  Last Modified: $(stat -c %y "$LOG_FILE" | cut -d'.' -f1)"
        echo ""
        
        extract_metrics "$LOG_FILE"
        
        echo ""
        echo "🔄 Live Log Tail (last 10 lines):"
        echo "----------------------------------------"
        tail -10 "$LOG_FILE" | sed 's/^/   /'
        
    else
        echo "❌ No training logs found"
        echo "💡 Start training with: docker-compose exec dformer bash /workspace/train.sh"
        echo ""
        echo "📁 Available files in checkpoint directory:"
        ls -la /workspace/checkpoints/Dformer_format_DFormer-Large/ 2>/dev/null || echo "   Directory not found"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "🔄 Refreshing in 10 seconds... (Ctrl+C to exit)"
    
    sleep 10
done