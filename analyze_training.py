#!/usr/bin/env python3
"""
DFormer Training Analysis Script
Extracts training metrics from log files and creates visualizations
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_training_log(log_file):
    """Parse training log file to extract metrics"""
    
    epochs = []
    losses = []
    total_losses = []
    learning_rates = []
    mious = []
    epoch_times = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Parse epoch completion lines
        epoch_match = re.search(r'Epoch (\d+)/\d+ Iter 107/107: lr=([\d.e-]+) loss=([\d.]+) total_loss=([\d.]+)', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            lr = float(epoch_match.group(2))
            loss = float(epoch_match.group(3))
            total_loss = float(epoch_match.group(4))
            
            epochs.append(epoch)
            learning_rates.append(lr)
            losses.append(loss)
            total_losses.append(total_loss)
        
        # Parse validation results
        miou_match = re.search(r'Epoch (\d+) validation result: mIoU ([\d.]+)', line)
        if miou_match:
            epoch = int(miou_match.group(1))
            miou = float(miou_match.group(2))
            mious.append((epoch, miou))
    
    return {
        'epochs': epochs,
        'losses': losses,
        'total_losses': total_losses,
        'learning_rates': learning_rates,
        'mious': mious
    }

def create_training_plots(data):
    """Create training visualization plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DFormer Training Analysis - 250 Epochs', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax1.plot(data['epochs'], data['losses'], 'b-', label='Loss', linewidth=2)
    ax1.plot(data['epochs'], data['total_losses'], 'r-', label='Total Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning Rate Schedule
    ax2.semilogy(data['epochs'], data['learning_rates'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate (log scale)')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: mIoU progression
    if data['mious']:
        miou_epochs, miou_values = zip(*data['mious'])
        ax3.plot(miou_epochs, miou_values, 'mo-', linewidth=2, markersize=4)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('mIoU (%)')
        ax3.set_title('Validation mIoU Progression')
        ax3.grid(True, alpha=0.3)
        
        # Highlight best mIoU
        best_miou = max(miou_values)
        best_epoch = miou_epochs[miou_values.index(best_miou)]
        ax3.axhline(y=best_miou, color='r', linestyle='--', alpha=0.7, label=f'Best mIoU: {best_miou:.2f} (Epoch {best_epoch})')
        ax3.legend()
    
    # Plot 4: Training summary stats
    ax4.axis('off')
    
    # Calculate statistics
    final_loss = data['losses'][-1] if data['losses'] else 0
    initial_loss = data['losses'][0] if data['losses'] else 0
    loss_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
    
    best_miou = max([miou for _, miou in data['mious']]) if data['mious'] else 0
    
    summary_text = f"""
    Training Summary:
    
    📊 Total Epochs: {max(data['epochs']) if data['epochs'] else 0}
    
    🎯 Loss Progression:
    • Initial Loss: {initial_loss:.4f}
    • Final Loss: {final_loss:.4f}
    • Reduction: {loss_reduction:.1f}%
    
    🏆 Best Performance:
    • Best mIoU: {best_miou:.2f}%
    • Training Time: ~6.4 hours
    • Avg. Time/Epoch: ~1.9 min
    
    ⚙️ Configuration:
    • Batch Size: 3
    • Workers: 3  
    • Resolution: 512×768
    • Model: DFormer-Large (37.68M params)
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/workspace/training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Training analysis plots saved to: /workspace/training_analysis.png")
    return fig

def main():
    """Main analysis function"""
    log_file = '/workspace/checkpoints/Dformer_format_DFormer-Large/log_2025_11_06_23_52_24.log'
    
    print("📊 Analyzing DFormer training logs...")
    print(f"📄 Log file: {log_file}")
    
    # Parse training data
    data = parse_training_log(log_file)
    
    print(f"✅ Found {len(data['epochs'])} training epochs")
    print(f"✅ Found {len(data['mious'])} validation results")
    
    # Create visualizations
    create_training_plots(data)
    
    # Print detailed metrics
    if data['mious']:
        print("\n📈 Validation mIoU Progress:")
        for epoch, miou in data['mious'][-10:]:  # Show last 10
            print(f"   Epoch {epoch:3d}: {miou:5.2f}%")
        
        best_miou = max([miou for _, miou in data['mious']])
        print(f"\n🏆 Best mIoU: {best_miou:.2f}%")

if __name__ == "__main__":
    main()