#!/bin/bash
# Minimal test to check data loading

cd /workspace

echo "=== Testing Data Loading ==="
python3 -c "
import sys
sys.path.insert(0, '/workspace')
print('Importing configuration...')
from local_configs.JARVIS.DFormer_Large_custom import C

print(f'Dataset path: {C.dataset_path}')
print(f'Batch size: {C.batch_size}')
print(f'Workers: {C.num_workers}')
print(f'Image size: {C.image_height}x{C.image_width}')

print('Testing file access...')
import os
rgb_path = os.path.join(C.rgb_root_folder, '24_8_frame_0000.jpg')
depth_path = os.path.join(C.x_root_folder, '24_8_frame_0000.png')
label_path = os.path.join(C.gt_root_folder, '24_8_frame_0000.png')

print(f'RGB exists: {os.path.exists(rgb_path)}')
print(f'Depth exists: {os.path.exists(depth_path)}')
print(f'Label exists: {os.path.exists(label_path)}')

if os.path.exists(rgb_path):
    print(f'RGB size: {os.path.getsize(rgb_path)} bytes')

print('Testing image loading...')
try:
    import cv2
    import numpy as np
    
    if os.path.exists(rgb_path):
        img = cv2.imread(rgb_path)
        print(f'RGB loaded: {img.shape if img is not None else \"Failed\"}')
    
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        print(f'Depth loaded: {depth.shape if depth is not None else \"Failed\"}')
        
    print('✅ Basic data loading test passed!')
    
except Exception as e:
    print(f'❌ Error during image loading: {e}')

print('Testing dataloader import...')
try:
    from utils.dataloader import get_train_loader
    print('✅ Dataloader import successful!')
except Exception as e:
    print(f'❌ Dataloader import failed: {e}')
"