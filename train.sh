GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0"
export TORCHDYNAMO_VERBOSE=1
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export PYTHONPATH="/workspace:$PYTHONPATH"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.JARVIS.DFormerv2_Base_custom --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --use_seed \
    --tensorboard
    
# To resume from checkpoint, add this line before --no-sliding:
# --continue_fpath=checkpoints/Dformer_format_DFormerv2_B/epoch-1_miou_8.79.pth \

# config for DFormers on NYUDepthv2
# local_configs.NYUDepthv2.DFormer_Large
# local_configs.NYUDepthv2.DFormer_Base
# local_configs.NYUDepthv2.DFormer_Small
# local_configs.NYUDepthv2.DFormer_Tiny
# local_configs.NYUDepthv2.DFormer_v2_S
# local_configs.NYUDepthv2.DFormer_v2_B
# local_configs.NYUDepthv2.DFormer_v2_L

# config for DFormers on SUNRGBD
# local_configs.SUNRGBD.DFormer_Large
# local_configs.SUNRGBD.DFormer_Base
# local_configs.SUNRGBD.DFormer_Small
# local_configs.SUNRGBD.DFormer_Tiny
# local_configs.SUNRGBD.DFormer_v2_S
# local_configs.SUNRGBD.DFormer_v2_B
# local_configs.SUNRGBD.DFormer_v2_L