#!/usr/bin/env python3

"""Simple script to check label encoding consistency for a dataset folder.

Usage:
    python utils/check_dataset_labels.py --dataset-root /workspace/datasets/D-Former_dataset

It will sample a few GTs and print a summary report including warnings for inconsistent encodings.
"""

import argparse
import os
import pprint
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataloader.RGBXDataset import RGBXDataset


def run_check(dataset_root: str, dataset_name='D-Former_dataset', sample_n: int = 12):
    # minimal settings required by RGBXDataset
    data_setting = {
        'rgb_root': os.path.join(dataset_root, 'RGB'),
        'rgb_format': '.png',
        'gt_root': os.path.join(dataset_root, 'Label'),
        'gt_format': '.png',
        'transform_gt': True,
        'x_root': os.path.join(dataset_root, 'Depth'),
        'x_format': '.png',
        'x_single_channel': True,
        'class_names': [],
        'train_source': os.path.join(dataset_root, 'train.txt') if os.path.exists(os.path.join(dataset_root, 'train.txt')) else '',
        'eval_source': os.path.join(dataset_root, 'test.txt') if os.path.exists(os.path.join(dataset_root, 'test.txt')) else '',
        'dataset_name': dataset_name,
        'backbone': 'DFormerv2_B',
    }

    ds = RGBXDataset(data_setting, 'val', preprocess=None)
    summary = ds._validate_label_encoding(sample_n=sample_n)
    pprint.pprint(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', '-d', default='/workspace/datasets/D-Former_dataset')
    parser.add_argument('--sample-n', '-n', default=12, type=int)

    args = parser.parse_args()
    run_check(args.dataset_root, sample_n=args.sample_n)
