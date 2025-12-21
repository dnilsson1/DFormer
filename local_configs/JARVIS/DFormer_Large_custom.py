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
C.gt_transform = True
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 6424
C.num_eval_imgs = 1606
C.num_classes = 18
C.class_names = [
    "battery",
    "battery_plate",
    "battery_strap",
    "bushing",
    "cable",
    "cable_holder",
    "cable_tie",
    "Collection",
    "connector",
    "cooler_pipe_sensor",
    "hose_clamp",
    "invertor",
    "invertor_cover",
    "nut",
    "pump_bracket",
    "pump_valve",
    "screw",
    "tube",
]

"""Image Config"""
C.background = 255
C.image_height = 512  # Safe resolution with good performance
C.image_width = 768   # Safe resolution with good performance
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormer-Large"  # Remember change the path below.
C.pretrained_model = "checkpoints/Dformer_format_DFormer-Large/epoch-1_miou_6.47.pth"
#C.pretrained_model = "checkpoints/Dformer_format_DFormer-Large/epoch-1_miou_6.47.pth"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01

# Class weights to handle severe imbalance (background=90.4%, others=0.01-2.19%)
# Using SQUARE ROOT of inverse frequency to prevent gradient explosion
# Original weights caused NaN at epoch 3 - sqrt scaling is more stable
# NOTE: Only 18 classes (0-17), class 8 is missing from dataset
C.class_weights = [
    1.0000,    # class 0 - background (90.40%) - sqrt(1.0) = 1.0
    6.4948,    # class 1 (2.14%) - sqrt(42.18)
    6.4206,    # class 2 (2.19%) - sqrt(41.22)
    25.0524,   # class 3 (0.14%) - sqrt(627.62)
    44.1716,   # class 4 (0.05%) - sqrt(1951.15)
    6.6786,    # class 5 (2.03%) - sqrt(44.60)
    12.5821,   # class 6 (0.57%) - sqrt(158.30)
    31.9869,   # class 7 (0.09%) - sqrt(1023.17)
    12.7606,   # class 9 (0.56%) - sqrt(162.83)
    57.5393,   # class 10 (0.03%) - sqrt(3310.77)
    31.2449,   # class 11 (0.09%) - sqrt(976.23)
    28.7409,   # class 12 (0.11%) - sqrt(826.07)
    14.2616,   # class 13 (0.44%) - sqrt(203.38)
    72.3684,   # class 14 (0.02%) - sqrt(5237.08)
    38.2558,   # class 15 (0.06%) - sqrt(1463.59)
    39.3877,   # class 16 (0.06%) - sqrt(1551.35)
    95.1335,   # class 17 (0.01%) - sqrt(9049.38) - MAX weight now 95x instead of 9049x
    9.4433,    # class 18 (1.02%) - sqrt(89.17)
]

C.batch_size = 3  # Safe batch size for RTX 3090
C.nepochs = 100
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 1
C.train_scale_array = [0.75, 1, 1.25, 1.5]
C.warm_up_epoch = 10
C.channels = [96, 192, 288, 576]


C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.15
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  # [0.75, 1, 1.25] #
C.eval_flip = True  # False #
C.eval_crop_size = [512, 768]  # [height weight] - matches training dimensions

"""Store Config"""
C.checkpoint_start_epoch = 20  # Start saving at epoch 20
C.checkpoint_step = 20         # Save every 20 epochs (epochs: 20, 40, 60, 80, 100)

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
