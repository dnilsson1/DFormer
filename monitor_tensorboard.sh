#!/bin/bash
# Start TensorBoard for training visualization

echo "=== TensorBoard Monitoring Setup ==="

TB_DIR="/workspace/checkpoints/Dformer_format_DFormerv2_B/tb/Nov15_15-14-55"

# Check if TensorBoard logs exist
if [ -d "$TB_DIR" ]; then
    echo "✅ TensorBoard directory found: $TB_DIR"
    echo "📊 Starting TensorBoard server..."
    echo ""
    echo "🌐 Access TensorBoard at: http://localhost:6006"
    echo "💡 Port 6006 needs to be exposed in docker-compose.yml"
    echo ""
    
    # Start TensorBoard using tensorboardX (already installed)
    python -c "from tensorboardX import SummaryWriter; import subprocess; subprocess.run(['python', '-m', 'tensorboard', '--logdir=$TB_DIR', '--host=0.0.0.0', '--port=6006'])" 2>/dev/null || \
    python -c "import os; os.system('pip install tensorboard && tensorboard --logdir=$TB_DIR --host=0.0.0.0 --port=6006')"
    
else
    echo "❌ TensorBoard directory not found: $TB_DIR"
    echo "💡 TensorBoard logs will be created during training"
    echo "📁 Current checkpoint structure:"
    find /workspace/checkpoints -type d 2>/dev/null || echo "No checkpoints directory found"
fi