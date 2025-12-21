#!/bin/bash
echo "🎨 Creating Inference Visualization Inside Docker Container..."
echo ""

# Run the visualization inside the container where matplotlib is available
docker-compose exec dformer python -c "
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

def create_inference_comparison():
    os.chdir('/workspace')
    
    # Paths
    rgb_dir = 'datasets/Dformer_format/RGB'
    pred_dir = 'inference_results/Dformer_format/RGB'
    
    # Get available prediction files
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*_pred.png')))
    
    if not pred_files:
        print('❌ No prediction files found!')
        return
    
    # Select first 6 images for comparison
    n_samples = min(6, len(pred_files))
    selected_preds = pred_files[:n_samples]
    
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 8))
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    success_count = 0
    for i, pred_path in enumerate(selected_preds):
        # Get corresponding original image
        pred_name = os.path.basename(pred_path)
        orig_name = pred_name.replace('_pred.png', '.jpg')
        orig_path = os.path.join(rgb_dir, orig_name)
        
        if not os.path.exists(orig_path):
            print(f'⚠️  Original image not found: {orig_path}')
            continue
        
        # Load images
        try:
            orig_img = Image.open(orig_path).convert('RGB')
            pred_img = Image.open(pred_path).convert('RGB')
            
            # Display original image
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original: {orig_name}', fontsize=8)
            axes[0, i].axis('off')
            
            # Display prediction
            axes[1, i].imshow(pred_img)
            axes[1, i].set_title(f'Prediction: {pred_name}', fontsize=8)
            axes[1, i].axis('off')
            
            success_count += 1
            
        except Exception as e:
            print(f'❌ Error loading images: {e}')
            continue
    
    plt.suptitle(f'DFormer Inference Results - Validation mIoU: 12.82%\\nJARVIS Dataset (18 classes) - {success_count} samples shown', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the comparison
    output_path = 'inference_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✅ Inference comparison saved: {output_path}')
    
    # Show statistics
    print(f'📊 Total predictions generated: {len(pred_files)}')
    print(f'📈 Validation mIoU: 12.82%')
    print(f'🏆 Best model used: epoch-30_miou_6.64.pth')
    
    return output_path

# Run the visualization
result = create_inference_comparison()
"

echo ""
echo "📁 Copying visualization to host system..."
docker-compose exec dformer cp /workspace/inference_comparison.png /workspace/inference_results/
echo "✅ Visualization saved to: inference_results/inference_comparison.png"