"""
Segmentation-specific Focal Loss wrapper for easy integration.

This is a simplified wrapper around the existing FocalLoss implementation,
specifically designed for semantic segmentation tasks with class imbalance.

Usage:
    from models.losses.segmentation_focal_loss import SegmentationFocalLoss
    
    criterion = SegmentationFocalLoss(
        num_classes=18,
        gamma=2.0,
        alpha=0.25,
        class_weights=[1.0, 2.5, ...],  # Optional
        ignore_index=255
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationFocalLoss(nn.Module):
    """
    Focal Loss for semantic segmentation with class imbalance.
    
    Combines focal loss modulation with optional class weights for handling
    extreme class imbalance. The focal loss automatically down-weights easy
    examples (high confidence predictions) and focuses on hard examples.
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        num_classes (int): Number of classes in the segmentation task.
        gamma (float): Focusing parameter for modulating loss. Higher gamma
            increases the penalty for misclassified examples. Default: 2.0
            - gamma=0: Equivalent to standard cross-entropy
            - gamma=2: Standard focal loss (recommended)
            - gamma=5: Very hard example mining
        alpha (float or list): Weighting factor for class imbalance.
            - If float: Same weight for all classes (0.25 is common)
            - If list: Per-class alpha values (length must equal num_classes)
            Default: 0.25
        class_weights (list or None): Additional class-specific weights for
            severe imbalance. These are multiplicative with alpha. Use moderate
            weights (2-20x) when combining with focal loss. Default: None
        ignore_index (int): Label index to ignore in loss computation. Default: 255
        reduction (str): Loss reduction mode ('mean', 'sum', 'none'). Default: 'mean'
    
    Example:
        # For JARVIS dataset with 18 classes and severe background imbalance
        criterion = SegmentationFocalLoss(
            num_classes=18,
            gamma=2.0,           # Standard focal loss
            alpha=0.25,          # Balanced weighting
            class_weights=[1.0, 3.0, 3.0, 5.0, ...],  # Moderate additional weights
            ignore_index=255
        )
        
        # During training
        pred = model(img)  # Shape: [B, 18, H, W]
        loss = criterion(pred, target)  # target shape: [B, H, W]
    """
    
    def __init__(
        self,
        num_classes,
        gamma=2.0,
        alpha=0.25,
        class_weights=None,
        ignore_index=255,
        reduction='none'
    ):
        super(SegmentationFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Convert class_weights to tensor if provided
        if class_weights is not None:
            assert len(class_weights) == num_classes, \
                f"class_weights length ({len(class_weights)}) must match num_classes ({num_classes})"
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = None
        
        # Convert alpha to tensor if it's a list
        if isinstance(alpha, list):
            assert len(alpha) == num_classes, \
                f"alpha length ({len(alpha)}) must match num_classes ({num_classes})"
            self.alpha = torch.FloatTensor(alpha)
        else:
            self.alpha = alpha
    
    def forward(self, pred, target):
        """
        Forward pass for focal loss computation.
        
        Args:
            pred (torch.Tensor): Model predictions with shape [B, C, H, W]
                where B=batch, C=num_classes, H=height, W=width
            target (torch.Tensor): Ground truth labels with shape [B, H, W]
                containing class indices in range [0, num_classes-1] or ignore_index
        
        Returns:
            torch.Tensor: Computed focal loss (scalar if reduction='mean' or 'sum')
        """
        assert pred.dim() == 4, f"Expected pred shape [B, C, H, W], got {pred.shape}"
        assert target.dim() == 3, f"Expected target shape [B, H, W], got {target.shape}"
        assert pred.size(0) == target.size(0), "Batch size mismatch"
        assert pred.size(2) == target.size(1) and pred.size(3) == target.size(2), \
            "Spatial dimensions mismatch"
        
        B, C, H, W = pred.shape
        
        # Move class_weights to same device as pred if needed
        if self.class_weights is not None:
            if self.class_weights.device != pred.device:
                self.class_weights = self.class_weights.to(pred.device)
        
        if isinstance(self.alpha, torch.Tensor) and self.alpha.device != pred.device:
            self.alpha = self.alpha.to(pred.device)
        
        # Reshape for easier computation
        # [B, C, H, W] -> [B*H*W, C]
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # [B, H, W] -> [B*H*W]
        target_flat = target.view(-1)
        
        # Create mask for valid pixels (ignore ignore_index)
        valid_mask = (target_flat != self.ignore_index)
        
        if valid_mask.sum() == 0:
            # No valid pixels, return zero loss that maintains gradient graph
            # Use mean of predictions * 0 to keep computation graph
            return (pred.mean(dim=1) * 0.0)  # Returns [B, H, W] with gradients
        
        # Filter out ignored pixels
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(pred_valid, target_valid, reduction='none')
        
        # Compute softmax probabilities and get probability of true class
        p = F.softmax(pred_valid, dim=1)
        p_t = p.gather(1, target_valid.unsqueeze(1)).squeeze(1)  # [N]
        
        # Clamp p_t to avoid numerical instability (log(0) = -inf, (1-1)^gamma = 0)
        # This prevents NaN in edge cases while maintaining gradient flow
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
        
        # Compute focal loss modulation factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal modulation
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            # Per-class alpha
            alpha_t = self.alpha[target_valid]
            focal_loss = alpha_t * focal_loss
        else:
            # Single alpha value
            focal_loss = self.alpha * focal_loss
        
        # Apply additional class weights if provided
        if self.class_weights is not None:
            class_weight_t = self.class_weights[target_valid]
            focal_loss = class_weight_t * focal_loss
        
        # Cleanup intermediate tensors to prevent memory accumulation
        del p, p_t, focal_weight, ce_loss, pred_valid, target_valid
        
        # Reduction - always return per-pixel loss for compatibility with builder.py
        # The model's forward() will handle masking ignore_index and calling .mean()
        # Reshape back to [B, H, W] and set ignored pixels to 0
        loss_map = pred_flat.new_zeros(B * H * W)
        loss_map[valid_mask] = focal_loss
        del focal_loss, pred_flat, target_flat, valid_mask
        return loss_map.view(B, H, W)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"gamma={self.gamma}, "
            f"alpha={self.alpha}, "
            f"class_weights={'None' if self.class_weights is None else 'enabled'}, "
            f"ignore_index={self.ignore_index}, "
            f"reduction='{self.reduction}')"
        )
