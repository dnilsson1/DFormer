#!/usr/bin/env python
"""Analyze depth ranges in training data and test images."""
import cv2
import numpy as np
import os
import sys

def analyze_depth(path, name=""):
    """Analyze a single depth image."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"Could not read: {path}")
        return None
    
    valid = depth[depth > 0]
    if len(valid) == 0:
        print(f"{name}: All zeros!")
        return None
    
    stats = {
        'min': valid.min(),
        'max': valid.max(),
        'mean': valid.mean(),
        'median': np.median(valid),
        'zeros': (depth == 0).sum(),
        'total': depth.size
    }
    print(f"{name}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.0f}, "
          f"median={stats['median']:.0f}, zeros={stats['zeros']}/{stats['total']} ({100*stats['zeros']/stats['total']:.1f}%)")
    return stats

def main():
    print("=" * 60)
    print("TEST IMAGES")
    print("=" * 60)
    test_dir = "datasets/Realsense_inference_test/Depth"
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            if f.endswith(('.png', '.jpg')):
                analyze_depth(os.path.join(test_dir, f), f)
    
    print("\n" + "=" * 60)
    print("TRAINING DATA SAMPLE (first 20 images)")
    print("=" * 60)
    train_dir = "datasets/Dformer_format/Depth"
    if os.path.exists(train_dir):
        files = sorted([f for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg'))])[:20]
        all_max = []
        for f in files:
            stats = analyze_depth(os.path.join(train_dir, f), f)
            if stats:
                all_max.append(stats['max'])
        if all_max:
            print(f"\nMax depth across sample: {max(all_max)}")
            print(f"Mean of max depths: {np.mean(all_max):.0f}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("For RealSense D435 (typical working range 0.1m - 10m):")
    print("  near_threshold = 100 (10cm, filters noise)")
    print("  background_threshold = 10000 (10m, or remove entirely)")

if __name__ == "__main__":
    main()
