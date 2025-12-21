#!/usr/bin/env python3
"""Analyze class pixel distribution in JARVIS dataset labels."""

import cv2
import numpy as np
import os
from collections import Counter

label_dir = '/workspace/datasets/Dformer_format/Label'
files = os.listdir(label_dir)

total_pixels = Counter()
for f in files:
    gt = cv2.imread(os.path.join(label_dir, f), cv2.IMREAD_GRAYSCALE)
    if gt is not None:
        unique, counts = np.unique(gt, return_counts=True)
        for u, c in zip(unique, counts):
            total_pixels[u] += c

total = sum(total_pixels.values())
print('=== Class Pixel Distribution ===')
for cls in sorted(total_pixels.keys()):
    pct = 100 * total_pixels[cls] / total
    print(f'Class {cls:2d}: {total_pixels[cls]:12,} pixels ({pct:6.2f}%)')
print(f'Total: {total:,} pixels')
