"""
DFormerv2_Base Resume Training Config
=====================================
Resume from epoch 10 checkpoint (63.47% mIoU) with lower learning rate
to prevent overfitting collapse observed after warm-up ends.

Key changes from DFormerv2_Base_custom.py:
- Resume from epoch-10_miou_63.47.pth checkpoint
- Lower LR: 5e-6 (was 3e-5) for gentle fine-tuning
- More frequent validation: every 2 epochs (was 10)
- No warm-up needed (already warmed up)
- Train for 40 more epochs (epochs 11-50)
"""

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
C.gt_transform = False
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 6424
C.num_eval_imgs = 1606
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
    "hose_clamp",         # 11
    "invertor",           # 12
    "invertor_cover",     # 13
    "nut",                # 14
    "pump_bracket",       # 15
    "pump_valve",         # 16
    "screw",              # 17
    "tube",               # 18
]

"""Image Config"""
C.background = 255
C.image_height = 512
C.image_width = 768
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

"""Random Seed"""
C.seed = 12345

""" Settings for network """
C.backbone = "DFormerv2_B"
# NOTE: Using --continue_fpath in train_resume.sh to load full checkpoint
# (including decoder and optimizer state). This pretrained_model is NOT used.
C.pretrained_model = None
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config - MODIFIED FOR RESUME"""
# REDUCED LR: 5e-6 (was 3e-5) - gentle fine-tuning to avoid overfitting
C.lr = 5e-6
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01

# Loss function configuration - same as before (worked well!)
C.use_focal_loss = True
C.focal_gamma = 2.0

# Per-class alpha weights (conservative range 0.1-5.0)
C.focal_alpha = [
    0.10,   # 0: battery (71.24%)
    0.50,   # 1: battery_plate (3.51%)
    0.35,   # 2: battery_strap (7.86%)
    2.20,   # 3: bushing (0.21%)
    2.70,   # 4: cable (0.14%)
    0.40,   # 5: cable_holder (5.62%)
    1.00,   # 6: cable_tie (1.05%)
    1.80,   # 7: Collection (0.30%)
    0.00,   # 8: UNUSED_COLLECTION (0%)
    0.70,   # 9: connector (2.00%)
    3.50,   # 10: cooler_pipe_sensor (0.08%)
    1.85,   # 11: hose_clamp (0.29%)
    1.10,   # 12: invertor (0.86%)
    0.60,   # 13: invertor_cover (2.53%)
    3.30,   # 14: nut (0.09%)
    1.60,   # 15: pump_bracket (0.40%)
    1.70,   # 16: pump_valve (0.34%)
    5.00,   # 17: screw (0.06%)
    0.55,   # 18: tube (3.42%)
]

C.class_weights = None

C.batch_size = 2
# REDUCED EPOCHS: Train for 40 more epochs (effectively epochs 11-50)
C.nepochs = 40
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0
C.train_scale_array = [0.75, 1, 1.25, 1.5]
# NO WARM-UP: Model is already warmed up from epoch 10
C.warm_up_epoch = 0
C.channels = [96, 192, 288, 576]

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.2  # Keep regularization high
C.aux_rate = 0.0

"""Eval Config - MORE FREQUENT VALIDATION"""
# VALIDATE EVERY 2 EPOCHS to catch overfitting early
C.eval_iter = 2
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [512, 768]

"""Store Config"""
C.checkpoint_start_epoch = 1  # Save from start since we're resuming
C.checkpoint_step = 2         # Save every 2 epochs to track progress

"""Path Config"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir))

# Use different log directory to avoid overwriting
C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone + "_resume")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

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
