# DFormerv2-Base configuration for JARVIS dataset
# FRESH TRAINING with FREQUENT VALIDATION to catch the optimal epoch
# Validates every 2 epochs starting from epoch 2

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
C.gt_transform = False  # JARVIS labels are already 0-indexed
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
    "battery", "battery_plate", "battery_strap", "bushing", "cable",
    "cable_holder", "cable_tie", "Collection", "UNUSED_COLLECTION",
    "connector", "cooler_pipe_sensor", "hose_clamp", "invertor",
    "invertor_cover", "nut", "pump_bracket", "pump_valve", "screw", "tube",
]

"""Image Config"""
C.background = 255
C.image_height = 512
C.image_width = 768
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

"""Random Seed"""
C.seed = 12345

""" Settings for network"""
C.backbone = "DFormerv2_B"
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Base_pretrained.pth"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 3e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01

# Loss function - Focal Loss with conservative alpha (same as previous successful run)
C.use_focal_loss = True
C.focal_gamma = 2.0
C.focal_alpha = [
    0.10, 0.50, 0.35, 2.20, 2.70, 0.40, 1.00, 1.80, 0.00, 0.70,
    3.50, 1.85, 1.10, 0.60, 3.30, 1.60, 1.70, 5.00, 0.55,
]
C.class_weights = None

C.batch_size = 2
C.nepochs = 30  # Reduced since overfitting starts early
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0
C.train_scale_array = [0.75, 1, 1.25, 1.5]
C.warm_up_epoch = 10
C.channels = [96, 192, 288, 576]

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.2
C.aux_rate = 0.0

"""Eval Config - FREQUENT VALIDATION"""
C.eval_iter = 2  # Validate every 2 epochs (was 10) - KEY CHANGE
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]
C.eval_flip = False
C.eval_crop_size = [512, 768]

"""Store Config - FREQUENT CHECKPOINTING"""
C.checkpoint_start_epoch = 2  # Start saving at epoch 2 (was 10)
C.checkpoint_step = 2  # Save every 2 epochs (was 10) - KEY CHANGE

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir))

# New experiment directory
C.log_dir = osp.abspath("checkpoints/Dformer_format_DFormerv2_B_freqval")
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

