#!/usr/bin/env python3
"""
Visualize inference results - compare original images with predictions
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

def create_inference_comparison():
    """Create a comparison grid showing original images vs predictions"""
    
    # Paths
    rgb_dir = "datasets/Dformer_format/RGB"
    pred_dir = "inference_results/Dformer_format/_epoch-8"
    
    # Class names for JARVIS dataset (18 classes + background)
    class_names = [
        "background",
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
        "tube"
    ]
    
    # Color palette used in predictions (corresponding to class_names)
    # These match the colors used in val_mm.py for saving predictions
    color_palette = [
        [128, 64, 128],   # background - purple
        [244, 35, 232],   # battery - pink
        [70, 70, 70],     # battery_plate - dark gray
        [102, 102, 156],  # battery_strap - blue-gray
        [190, 153, 153],  # bushing - beige
        [153, 153, 153],  # cable - gray
        [250, 170, 30],   # cable_holder - orange
        [220, 220, 0],    # cable_tie - yellow
        [107, 142, 35],   # Collection - olive
        [152, 251, 152],  # connector - light green
        [70, 130, 180],   # cooler_pipe_sensor - steel blue
        [220, 20, 60],    # hose_clamp - crimson
        [255, 0, 0],      # invertor - red
        [0, 0, 142],      # invertor_cover - dark blue
        [0, 0, 70],       # nut - navy
        [0, 60, 100],     # pump_bracket - dark cyan
        [0, 80, 100],     # pump_valve - teal
        [0, 0, 230],      # screw - blue
        [119, 11, 32]     # tube - maroon
    ]
    
    # Get available prediction files
    pred_files = glob.glob(os.path.join(pred_dir, "*_pred.png"))
    
    if not pred_files:
        print("❌ No prediction files found. Run inference first!")
        return
    
    # Select first 6 images for comparison
    n_samples = min(6, len(pred_files))
    selected_preds = pred_files[:n_samples]
    
    # Create figure with extra space for legend
    fig = plt.figure(figsize=(22, 10))
    
    # Create grid: 2 rows for images, space on right for legend
    gs = fig.add_gridspec(2, n_samples + 1, width_ratios=[1]*n_samples + [0.3],
                          hspace=0.3, wspace=0.1)
    
    for i, pred_path in enumerate(selected_preds):
        # Get corresponding original image
        pred_name = os.path.basename(pred_path)
        orig_name = pred_name.replace("_pred.png", ".jpg")
        orig_path = os.path.join(rgb_dir, orig_name)
        
        if not os.path.exists(orig_path):
            print(f"⚠️  Original image not found: {orig_path}")
            continue
        
        # Load images
        try:
            orig_img = Image.open(orig_path).convert('RGB')
            pred_img = Image.open(pred_path).convert('RGB')
            
            # Display original image
            ax1 = fig.add_subplot(gs[0, i])
            ax1.imshow(orig_img)
            ax1.set_title(f'Original: {orig_name}', fontsize=10)
            ax1.axis('off')
            
            # Display prediction
            ax2 = fig.add_subplot(gs[1, i])
            ax2.imshow(pred_img)
            ax2.set_title(f'Prediction: {pred_name}', fontsize=10)
            ax2.axis('off')
            
        except Exception as e:
            print(f"❌ Error loading images: {e}")
            continue
    
    # Create color legend on the right side
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis('off')
    
    # Create color patches for legend
    from matplotlib.patches import Rectangle
    y_offset = 0.95
    y_step = 0.05
    
    legend_ax.text(0.1, 1.0, 'Class Legend:', fontsize=12, fontweight='bold',
                   transform=legend_ax.transAxes)
    
    for i, (class_name, color) in enumerate(zip(class_names, color_palette)):
        # Normalize color to [0, 1] range
        color_norm = [c/255.0 for c in color]
        
        # Add colored rectangle
        rect = Rectangle((0.1, y_offset - i*y_step), 0.15, 0.04,
                        facecolor=color_norm, edgecolor='black', linewidth=0.5,
                        transform=legend_ax.transAxes)
        legend_ax.add_patch(rect)
        
        # Add class name
        legend_ax.text(0.3, y_offset - i*y_step + 0.02, class_name,
                      fontsize=8, verticalalignment='center',
                      transform=legend_ax.transAxes)
    
    plt.suptitle(f'DFormer Inference Results - Best mIoU: 66.35%\nJARVIS Dataset (18 classes)', 
                 fontsize=16, fontweight='bold')
    
    # Save the comparison
    output_path = "inference_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Inference comparison saved: {output_path}")
    
    # Show statistics
    print(f"\n📊 Inference Results Summary:")
    print(f"   • Total predictions: {len(pred_files)}")
    print(f"   • Best mIoU: 66.35%")
    print(f"   • Model: epoch-8_miou_66.35")
    print(f"   • Dataset: JARVIS (18 classes)")
    print(f"\n🎨 Class Colors:")
    for i, (class_name, color) in enumerate(zip(class_names[:10], color_palette[:10])):
        print(f"   • {class_name:20s} → RGB{tuple(color)}")
    if len(class_names) > 10:
        print(f"   ... and {len(class_names)-10} more classes")
    
    plt.show()

if __name__ == "__main__":
    print("🔍 Creating inference results visualization...")
    create_inference_comparison()