#!/bin/bash
echo "🔮 Running inference with your trained DFormer model..."
echo "📂 Dataset: Dformer_format"
echo ""

# Parse command-line arguments (allow override of defaults)
CONFIG="${CONFIG:-local_configs.JARVIS.DFormerv2_Base_custom}"
CHECKPOINT="${CHECKPOINT:-/workspace/checkpoints/Dformer_format_DFormerv2_B/epoch-10_miou_46.43.pth}"
GPUS="${GPUS:-1}"
SAVE_PATH="${SAVE_PATH:-inference_results_epoch-10_miou_46.43}"
SHOW_IMAGE="${SHOW_IMAGE:---show_image}"
VERBOSE="${VERBOSE:---verbose}"

echo "🏆 Using config: $CONFIG"
echo "📊 Using checkpoint: $CHECKPOINT"
echo "💾 Output directory: $SAVE_PATH"
echo ""

cd /workspace
export LOCAL_RANK=0
export RANK=0 
export WORLD_SIZE=1
export PYTHONPATH="/workspace:$PYTHONPATH"

# Create output directory for inference results
mkdir -p /workspace/$SAVE_PATH

# Run inference with your best trained model
python utils/infer.py \
    --config "$CONFIG" \
    --gpus "$GPUS" \
    --continue_fpath "$CHECKPOINT" \
    --save_path "$SAVE_PATH" \
    $SHOW_IMAGE \
    $VERBOSE

# choose the dataset and DFormer for evaluating

# NYUv2 DFormers
# --config=local_configs.NYUDepthv2.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/trained/NYUv2_DFormer_Large/Base/Small/Tiny.pth

# SUNRGBD DFormers
# --config=local_configs.SUNRGBD.DFormer_Large/Base/Small/Tiny
# --continue_fpath=checkpoints/trained/SUNRGBD_DFormer_Large/Base/Small/Tiny.pth
