import os
import numpy as np
from PIL import Image

def analyze_colors():
    base_dir = '/workspace/inference_results_epoch-8_miou_66.35/Dformer_format/RGB'
    if not os.path.exists(base_dir):
        print(f"Dir not found: {base_dir}")
        return

    files = [f for f in os.listdir(base_dir) if f.endswith('_pred.png')]
    if not files:
        print("No pred files found")
        return

    # Check first 5 files
    for f in files[:5]:
        path = os.path.join(base_dir, f)
        img = Image.open(path).convert('RGB')
        arr = np.array(img)
        # Reshape to list of colors
        colors = arr.reshape(-1, 3)
        unique_colors = np.unique(colors, axis=0)
        
        print(f"File: {f}")
        for c in unique_colors:
            print(f"  Color: {c}")

if __name__ == '__main__':
    analyze_colors()
