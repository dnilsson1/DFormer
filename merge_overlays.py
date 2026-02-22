import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def merge_images():
    # Define paths
    inference_dir = r"D:\PhD\Jarvis\DFormer\inference_results_epoch-8_miou_66.35\Dformer_format\RGB"
    original_dir = r"D:\PhD\Jarvis\DFormer\datasets\D-Former_dataset\RGB"
    output_dir = r"D:\PhD\Jarvis\DFormer\inference_results\merged"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of files in inference directory
    inference_files = [f for f in os.listdir(inference_dir) if f.endswith('_pred.png')]
    
    print(f"Found {len(inference_files)} inference files.")

    for inf_file in tqdm(inference_files, desc="Merging images"):
        # Construct the corresponding original image filename
        # Remove '_pred.png' and add '.jpg'
        base_name = inf_file.replace('_pred.png', '')
        orig_file = base_name + '.jpg'
        
        inf_path = os.path.join(inference_dir, inf_file)
        orig_path = os.path.join(original_dir, orig_file)
        out_path = os.path.join(output_dir, base_name + '_merged.jpg')

        # Check if original file exists
        if not os.path.exists(orig_path):
            print(f"Warning: Original file not found for {inf_file} -> Expected {orig_file}")
            continue

        # Read images
        img_inf = cv2.imread(inf_path)
        img_orig = cv2.imread(orig_path)

        if img_inf is None:
            print(f"Error reading inference image: {inf_path}")
            continue
        if img_orig is None:
            print(f"Error reading original image: {orig_path}")
            continue

        # Resize inference image to match original image size if they differ
        if img_inf.shape != img_orig.shape:
            img_inf = cv2.resize(img_inf, (img_orig.shape[1], img_orig.shape[0]))

        # Overlay images (0.5 weight for both)
        # You can adjust alpha and beta for transparency
        alpha = 0.6  # Weight for original image
        beta = 0.4   # Weight for inference overlay
        merged = cv2.addWeighted(img_orig, alpha, img_inf, beta, 0)

        # Save merged image
        cv2.imwrite(out_path, merged)

    print(f"Processing complete. Merged images saved to {output_dir}")

if __name__ == "__main__":
    merge_images()
