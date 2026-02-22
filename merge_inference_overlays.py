import os
import numpy as np
from PIL import Image
import sys

def merge_images():
    base_dir = '/workspace'
    pred_dir = os.path.join(base_dir, 'inference_results_epoch-8_miou_66.35', 'Dformer_format', 'RGB')
    orig_dir = os.path.join(base_dir, 'datasets', 'D-Former_dataset', 'RGB')
    out_dir = os.path.join(base_dir, 'inference_results', 'merged')
    
    print(f"Pred Dir: {pred_dir}")
    print(f"Orig Dir: {orig_dir}")
    print(f"Out Dir: {out_dir}")

    if not os.path.exists(pred_dir):
        print(f"Error: Prediction directory does not exist: {pred_dir}")
        return
    if not os.path.exists(orig_dir):
        print(f"Error: Original directory does not exist: {orig_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    
    # List pred files
    try:
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('_pred.png')]
    except Exception as e:
        print(f"Error listing prediction directory: {e}")
        return
    
    print(f"Found {len(pred_files)} prediction images.")
    
    count = 0
    for pred_file in pred_files:
        # Determine original filename
        # Remove '_pred.png' and add '.jpg'
        base_name = pred_file[:-9] # remove '_pred.png'
        orig_file = base_name + '.jpg'
        
        pred_path = os.path.join(pred_dir, pred_file)
        orig_path = os.path.join(orig_dir, orig_file)
        
        if not os.path.exists(orig_path):
            # Try finding it recursively or exact match failed
            # print(f"Original image not found for {pred_file} (looked for {orig_file})")
            continue
            
        try:
            # Open images
            pred_img = Image.open(pred_path).convert('RGBA')
            orig_img = Image.open(orig_path).convert('RGBA')
            
            # Resize pred to match orig if necessary
            if pred_img.size != orig_img.size:
                pred_img = pred_img.resize(orig_img.size, Image.BILINEAR)
            
            # --- COLOR REPLACEMENT START ---
            # Replace 'pipe' colors with Yellow [255, 255, 0]
            # Based on val_mm.py palette for Dformer_format/JARVIS:
            # Class 18 ('tube') -> [0, 192, 0] (Green)
            # Class 10 ('cooler_pipe_sensor') -> [64, 128, 0] (Olive)
            
            pred_arr = np.array(pred_img)
            
            # Define target color (Yellow)
            yellow = [255, 255, 0, 255] # RGBA
            
            # Colors to replace (RGB from check_colors.py, alpha assumed 255)
            # [0, 192, 0] and [64, 128, 0]
            
            # Create mask for Class 18 (Tube)
            mask18 = np.all(pred_arr[:, :, :3] == [0, 192, 0], axis=-1)
            pred_arr[mask18] = yellow
            
            # Create mask for Class 10 (Cooler Pipe Sensor)
            mask10 = np.all(pred_arr[:, :, :3] == [64, 128, 0], axis=-1)
            pred_arr[mask10] = yellow
            
            pred_img = Image.fromarray(pred_arr)
            # --- COLOR REPLACEMENT END ---

            # Blend
            merged = Image.blend(orig_img, pred_img, 0.5)
            
            # Save
            out_path = os.path.join(out_dir, base_name + '_merged.png')
            merged.save(out_path)
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} images...", end='\r')
            
        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
            
    print(f"\nDone. Proceeded {count} images. Saved to {out_dir}")

if __name__ == '__main__':
    merge_images()
