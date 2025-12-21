# Focal Loss Implementation Guide

## Overview

Focal loss has been integrated into the DFormer training pipeline to handle severe class imbalance. This guide explains how to use it and configure it for optimal results.

## Quick Start

### Enable Focal Loss

Edit `local_configs/JARVIS/DFormerv2_Base_custom.py`:

```python
# Set this to True to enable Focal Loss
C.use_focal_loss = True

# Configure focal loss parameters
C.focal_gamma = 2.0      # Focusing parameter (recommended: 2.0)
C.focal_alpha = 0.25     # Balancing parameter (recommended: 0.25)

# Use MODERATE class weights with focal loss (5-10x max instead of 20x)
C.class_weights = [1.0, 2.0, 2.0, 3.0, ...]  # Adjust as needed
```

### Run Training

```bash
docker-compose exec dformer bash -c "pkill -f train.py; rm -rf /workspace/local_configs/__pycache__ /workspace/local_configs/JARVIS/__pycache__; bash /workspace/train.sh"
```

## Configuration Options

### Loss Function Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `C.use_focal_loss` | bool | `False` | Enable/disable focal loss. `False` uses CrossEntropyLoss |

### Focal Loss Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `C.focal_gamma` | float | `2.0` | Focusing parameter γ. Controls how much to down-weight easy examples |
| `C.focal_alpha` | float | `0.25` | Balancing parameter α. Weights foreground vs background |
| `C.class_weights` | list | cube root | Optional class-specific weights (multiplicative with alpha) |

### Gamma (γ) Parameter Guide

The gamma parameter controls the **focusing strength** - how much the loss focuses on hard examples:

- **γ = 0**: Equivalent to standard cross-entropy (no focusing)
- **γ = 1**: Mild focusing on hard examples
- **γ = 2**: **Standard focal loss (recommended starting point)**
- **γ = 5**: Aggressive hard example mining

**Formula effect**: Loss is multiplied by `(1 - p_t)^γ` where `p_t` is the predicted probability of the true class.

Example with γ=2:
- Easy example (p_t=0.9): Loss weight = (1-0.9)² = 0.01 (99% reduction)
- Hard example (p_t=0.5): Loss weight = (1-0.5)² = 0.25 (75% reduction)
- Very hard (p_t=0.1): Loss weight = (1-0.1)² = 0.81 (19% reduction)

### Alpha (α) Parameter Guide

The alpha parameter provides **class balancing**:

- **α = 0.5**: No class balancing (equal weight)
- **α = 0.25**: **Common choice for foreground/background balancing**
- **α = [list]**: Per-class alpha values (advanced usage)

For JARVIS dataset with 90% background:
- Use α=0.25 to give less weight to the abundant background class
- Alternatively, use per-class alpha based on class frequency

### Class Weights Strategy

When **combining focal loss with class weights**, use **moderate weights** to avoid gradient instability:

| Scenario | Recommended Weight Range | Rationale |
|----------|-------------------------|-----------|
| Focal loss only | `C.class_weights = None` | Let focal loss handle imbalance automatically |
| Focal + moderate weights | Max 5-10x | Focal loss already down-weights easy examples |
| Extreme imbalance | Max 10-20x | Higher weights risk NaN gradients |
| CrossEntropyLoss only | Use cube/4th root weights | Need stronger weights without focal modulation |

**Current JARVIS weights (cube root)**: Max ~21x
- **Recommended for CrossEntropyLoss**: Keep as-is
- **Recommended for Focal Loss**: Reduce to 5-10x max

## Usage Patterns

### Pattern 1: Focal Loss Only (Recommended First Try)

```python
C.use_focal_loss = True
C.focal_gamma = 2.0
C.focal_alpha = 0.25
C.class_weights = None  # Let focal loss handle everything
```

**Pros**: Cleanest approach, automatic hard example mining
**Cons**: May not be sufficient for extreme imbalance (90%+ background)

### Pattern 2: Focal Loss + Moderate Weights (Recommended for JARVIS)

```python
C.use_focal_loss = True
C.focal_gamma = 2.0
C.focal_alpha = 0.25
C.class_weights = [
    1.0,     # background
    2.0,     # common classes
    3.0-5.0, # rare classes
    8.0-10.0 # very rare classes (0.01-0.05%)
]
```

**Pros**: Combines automatic focusing with static class balancing
**Cons**: Requires tuning both gamma and weights

### Pattern 3: CrossEntropyLoss + Strong Weights (Current Setup)

```python
C.use_focal_loss = False
C.class_weights = [1.0, 3.48, 3.46, ..., 20.83]  # Cube root weights
```

**Pros**: Simple, works with existing config
**Cons**: May cause gradient explosion with weights >20x

### Pattern 4: Aggressive Focal Loss (For Very Hard Cases)

```python
C.use_focal_loss = True
C.focal_gamma = 3.0  # Increased from 2.0
C.focal_alpha = 0.20  # Lower alpha for background class
C.class_weights = [1.0, 2.0, 2.0, 3.0, ...]  # Light weights
```

**Pros**: Maximum focus on hard examples
**Cons**: May be too aggressive, causing instability

## Troubleshooting

### NaN Gradients

**Symptoms**: Training shows NaN loss after some iterations

**Solutions**:
1. **Reduce class weights**: If using focal loss, reduce max weight to 5-10x
2. **Lower gamma**: Try γ=1.5 or γ=1.0 for gentler focusing
3. **Check learning rate**: May need to reduce lr with strong focal loss

### Background Overfitting

**Symptoms**: Model predicts mostly background class, low mIoU on rare classes

**Solutions**:
1. **Increase gamma**: Try γ=3.0 to focus more on hard examples
2. **Add moderate class weights**: Even with focal loss, use 5-10x weights
3. **Lower alpha**: Try α=0.20 to reduce background weight
4. **Increase rare class weights**: Target 10-15x for rarest classes

### Training Instability

**Symptoms**: Loss fluctuates wildly, mIoU inconsistent

**Solutions**:
1. **Lower gamma**: Use γ=1.5 or γ=1.0
2. **Remove class weights**: Set `C.class_weights = None`
3. **Reduce learning rate**: Try `C.lr = 4e-5` instead of 6e-5

## Implementation Details

### File Structure

```
models/losses/
├── segmentation_focal_loss.py  # New: Clean focal loss wrapper
├── focal_loss.py               # Existing: MMSeg focal loss implementation
└── __init__.py                 # Exports FocalLoss

utils/
└── train.py                    # Modified: Supports C.use_focal_loss flag

local_configs/JARVIS/
└── DFormerv2_Base_custom.py   # Modified: Added focal loss config
```

### SegmentationFocalLoss Class

Located in `models/losses/segmentation_focal_loss.py`

**Features**:
- Clean API designed for semantic segmentation
- Supports both single alpha and per-class alpha
- Optional class weights (multiplicative with alpha)
- Proper handling of ignore_index (default: 255)
- Efficient computation with PyTorch operations

**Signature**:
```python
SegmentationFocalLoss(
    num_classes: int,
    gamma: float = 2.0,
    alpha: float | list = 0.25,
    class_weights: list | None = None,
    ignore_index: int = 255,
    reduction: str = 'mean'
)
```

### Training Script Integration

The training script (`utils/train.py`) automatically selects the loss function based on `C.use_focal_loss`:

```python
if use_focal_loss:
    criterion = SegmentationFocalLoss(
        num_classes=config.num_classes,
        gamma=config.focal_gamma,
        alpha=config.focal_alpha,
        class_weights=config.class_weights,
        ignore_index=config.background,
        reduction='mean'
    )
else:
    criterion = nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=config.background,
        weight=class_weights
    )
```

## Recommended Testing Strategy

### Step 1: Baseline Focal Loss (No Weights)

```python
C.use_focal_loss = True
C.focal_gamma = 2.0
C.focal_alpha = 0.25
C.class_weights = None
```

Run for 1 epoch, check:
- ✅ No NaN gradients
- ✅ Loss decreasing
- 📊 Validation mIoU at epoch 1

**Expected**: mIoU 5-10%, stable training

### Step 2: Focal Loss + Light Weights

```python
C.use_focal_loss = True
C.focal_gamma = 2.0
C.focal_alpha = 0.25
C.class_weights = [1.0, 2.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 
                   3.0, 5.0, 4.0, 4.0, 3.0, 6.0, 5.0, 5.0, 8.0, 2.5]
```

Run for 1 epoch, check:
- ✅ No NaN gradients
- ✅ Better mIoU than Step 1
- 📊 Per-class mIoU breakdown

**Expected**: mIoU 8-12%, rare classes starting to appear

### Step 3: Fine-tune Gamma

If background still dominates:
```python
C.focal_gamma = 3.0  # Increase focusing
```

If training unstable:
```python
C.focal_gamma = 1.5  # Decrease focusing
```

### Step 4: Full 50-Epoch Training

Once you find stable config with good epoch-1 mIoU (>10%), run full training:

```bash
# Expected results with optimal focal loss config:
# Epoch 1: mIoU 10-15%
# Epoch 10: mIoU 18-25%
# Epoch 20: mIoU 22-30%
# Epoch 50: mIoU 28-38%
```

## Monitoring

### TensorBoard Metrics

Monitor these during training:

1. **Total Loss**: Should decrease smoothly (no sudden spikes)
2. **Validation mIoU**: Should increase steadily
3. **Per-class IoU**: Check that rare classes are being learned

Access TensorBoard:
```bash
# On host machine
http://localhost:6006
```

### Log Messages

Look for:
```
Using Focal Loss: gamma=2.0, alpha=0.25, class_weights=enabled
```

This confirms focal loss is active.

### Quick Validation Check

After epoch 1, manually check predictions:
```python
python visualize_inference.py
```

Look for:
- Rare classes (screws, nuts, connectors) should appear in predictions
- Background should not dominate 100% of pixels
- Hard boundaries should be detected

## Performance Comparison

| Configuration | Max Weight | Gradient Stability | Background Overfitting Risk | Expected Final mIoU |
|---------------|------------|-------------------|---------------------------|-------------------|
| CrossEntropy + 4th root (9.75x) | 9.75x | ✅ Stable | ⚠️ High risk | 15-20% |
| CrossEntropy + cube root (21x) | 20.8x | ⚠️ May diverge | ✅ Low risk | 20-28% |
| Focal (γ=2) alone | N/A | ✅ Very stable | ⚠️ Moderate risk | 18-25% |
| **Focal (γ=2) + light weights (8x)** | **8x** | **✅ Stable** | **✅ Low risk** | **25-35%** ⭐ |
| Focal (γ=3) + light weights (8x) | 8x | ⚠️ May be unstable | ✅ Very low risk | 28-38% |

⭐ **Recommended starting point for JARVIS dataset**

## References

- Original Paper: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- Implementation: Based on MMSegmentation FocalLoss with segmentation-specific wrapper
- JARVIS Dataset Stats: 90.4% background, 18 classes, severe imbalance

## Support

For issues or questions:
1. Check TensorBoard logs for loss/mIoU trends
2. Review training logs for NaN or instability warnings
3. Try the recommended testing strategy above
4. Adjust gamma/alpha/weights based on troubleshooting guide
