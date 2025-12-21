#!/bin/bash
# Ultra-minimal test - just try to load one sample manually

cd /workspace

echo "=== Manual Dataset Test ==="

python3 -c "
import sys
sys.path.insert(0, '/workspace')
import cv2
import numpy as np

print('🔍 Testing manual sample loading...')

# Load config
from local_configs.JARVIS.DFormer_Large_custom import C

# Read sample files directly
rgb_path = '/workspace/datasets/Dformer_format/RGB/24_8_frame_0000.jpg'
depth_path = '/workspace/datasets/Dformer_format/Depth/24_8_frame_0000.png'
label_path = '/workspace/datasets/Dformer_format/Label/24_8_frame_0000.png'

print('📸 Loading RGB image...')
rgb = cv2.imread(rgb_path)
print(f'   RGB shape: {rgb.shape}')

print('🏔️ Loading Depth image...')  
depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
print(f'   Depth shape: {depth.shape}')

print('🏷️ Loading Label image...')
label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  
print(f'   Label shape: {label.shape}')

# Resize to target resolution
target_h, target_w = C.image_height, C.image_width
print(f'🔄 Resizing to target: {target_h}x{target_w}')

rgb_resized = cv2.resize(rgb, (target_w, target_h))
depth_resized = cv2.resize(depth, (target_w, target_h))
label_resized = cv2.resize(label, (target_w, target_h))

print(f'   RGB resized: {rgb_resized.shape}')
print(f'   Depth resized: {depth_resized.shape}')
print(f'   Label resized: {label_resized.shape}')

# Check label values
unique_labels = np.unique(label_resized)
print(f'📊 Unique label values: {unique_labels[:10]}...' if len(unique_labels) > 10 else f'📊 Unique label values: {unique_labels}')
print(f'   Number of classes found: {len(unique_labels)}')
print(f'   Expected classes: {C.num_classes}')

print('✅ Manual loading test completed successfully!')
"