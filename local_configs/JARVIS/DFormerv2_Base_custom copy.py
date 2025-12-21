import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

remoteip = os.popen("pwd").read()
C.root_dir = "datasets"
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = "Dformer_format"
C.dataset_path = osp.join(C.root_dir, "Dformer_format")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
# JARVIS labels are already 0-indexed (0..17), so do NOT apply gt_transform
# which subtracts 1 (designed for 1-indexed datasets like NYU).
# If True, label 0 would become 255 (ignored), breaking training!
C.gt_transform = False
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 6424
C.num_eval_imgs = 1606
# JARVIS labels are 0-indexed with values 0-18 (class 8 is unused - empty Blender Collection)
# Need 19 output channels to cover all possible label values (0..18)
C.num_classes = 19
C.class_names = [
    "battery",            # 0
    "battery_plate",      # 1
    "battery_strap",      # 2
    "bushing",            # 3
    "cable",              # 4
    "cable_holder",       # 5
    "cable_tie",          # 6
    "Collection",         # 7
    "UNUSED_COLLECTION",  # 8 - empty Blender Collection, no pixels have this label
    "connector",          # 9
    "cooler_pipe_sensor", # 10
    "hose_clamp",       # 11
    "invertor",         # 12
    "invertor_cover",   # 13
    "nut",              # 14
    "pump_bracket",     # 15
    "pump_valve",       # 16
    "screw",            # 17
    "tube",             # 18
]

"""Image Config"""
C.background = 255
C.image_height = 512  # Safe resolution with good performance   
C.image_width = 768   # Safe resolution with good performance
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

"""Random Seed"""
C.seed = 12345  # Fixed seed for reproducibility

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormerv2_B"  # DFormerv2 Base (underscore not hyphen!)
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Base_pretrained.pth"  # ImageNet-1K pretrained weights
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 3e-5  # Reduced from 6e-5 to prevent NaN loss
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01

# Loss function configuration
# Using Focal Loss with per-class alpha weights for severe class imbalance
# Previous run with focal_alpha=0.25 (uniform) caused battery overfitting
C.use_focal_loss = True

# Focal Loss parameters
# gamma: Focus on hard examples (higher = more focus on misclassified)
# alpha: Per-class weights based on inverse frequency (replaces class_weights)
C.focal_gamma = 2.0

# Per-class alpha weights based on inverse pixel frequency
# Battery (71%) gets very low alpha, rare classes get high alpha
# These act as class-balancing weights within focal loss
C.focal_alpha = [
    0.02,   # 0: battery (71.24%) - heavily downweight
    0.40,   # 1: battery_plate (3.51%)
    0.18,   # 2: battery_strap (7.86%)
    6.00,   # 3: bushing (0.21%)
    9.00,   # 4: cable (0.14%)
    0.25,   # 5: cable_holder (5.62%)
    1.20,   # 6: cable_tie (1.05%)
    4.20,   # 7: Collection (0.30%)
    0.00,   # 8: UNUSED_COLLECTION (0%) - no pixels
    0.65,   # 9: connector (2.00%)
    15.00,  # 10: cooler_pipe_sensor (0.08%)
    4.30,   # 11: hose_clamp (0.29%)
    1.45,   # 12: invertor (0.86%)
    0.50,   # 13: invertor_cover (2.53%)
    14.00,  # 14: nut (0.09%)
    3.10,   # 15: pump_bracket (0.40%)
    3.70,   # 16: pump_valve (0.34%)
    20.00,  # 17: screw (0.06%) - rarest class
    0.37,   # 18: tube (3.42%)
]

# Don't use class_weights with focal loss - use focal_alpha instead
C.class_weights = None

C.batch_size = 2  # Reduced from 3 to avoid OOM during validation
C.nepochs = 150  # Fine-tuning from NYUDepthv2 pretrained DFormerv2-B
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0  # Previously stable with batch_size=2
C.train_scale_array = [0.75, 1, 1.25, 1.5]
C.warm_up_epoch = 10  # Increased to prevent NaN and stabilize training
C.channels = [96, 192, 288, 576]


C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.1  # Reduced from 0.2 to allow model to learn with focal loss
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 10
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  # Single scale to reduce memory
C.eval_flip = False  # Disable flip to reduce VRAM usage (was causing VRAM spike)
C.eval_crop_size = [512, 768]  # [height weight] - matches training dimensions

"""Store Config"""
C.checkpoint_start_epoch = 20  # Start saving at epoch 10
C.checkpoint_step = 10        # Save every 10 epochs (10, 20, 30, 40, 50)

"""Path Config"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(
    osp.join(C.log_dir, "checkpoint")
)  #'/mnt/sda/repos/2023_RGBX/pretrained/'#osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"

if __name__ == "__main__":
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument("-tb", "--tensorboard", default=False, action="store_true")
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
