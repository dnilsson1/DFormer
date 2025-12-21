#!/usr/bin/env python
import cv2
import numpy as np
import sys

depth_path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/Realsense_inference_test/Depth/depth1.png'

# Check depth values - try different read modes
print(f"Checking depth file: {depth_path}")

# Grayscale
depth_gray = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
print("\n=== GRAYSCALE MODE ===")
print('Shape:', depth_gray.shape)
print('Dtype:', depth_gray.dtype)
print('Min:', depth_gray.min())
print('Max:', depth_gray.max())
print('Mean:', depth_gray.mean())
print('Pixels <= 10:', (depth_gray <= 10).sum())
print('Pixels > 10:', (depth_gray > 10).sum())

# Unchanged (16-bit if available)
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
print("\n=== UNCHANGED MODE ===")
print('Shape:', depth_raw.shape)
print('Dtype:', depth_raw.dtype)
print('Min:', depth_raw.min())
print('Max:', depth_raw.max())
print('Mean:', depth_raw.mean())

# Check if 16-bit
if depth_raw.dtype == np.uint16:
    print('Pixels == 0:', (depth_raw == 0).sum())
    print('Pixels > 0:', (depth_raw > 0).sum())
    print('Non-zero min:', depth_raw[depth_raw > 0].min() if (depth_raw > 0).any() else 'N/A')
