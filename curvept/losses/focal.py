"""
Focal Loss for confidence prediction.

Handles the severe class imbalance between real DLO queries (few) and
background queries (many). Without focal loss, the model would learn to
predict "no object" for everything.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(pred_logits: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Binary focal loss.

    Args:
        pred_logits: (B, Q) raw logits (before sigmoid)
        targets: (B, Q) binary targets (1 = real DLO, 0 = background)
        alpha: weighting factor for positive class
        gamma: focusing parameter (higher = more focus on hard examples)

    Returns:
        loss: scalar focal loss
    """
    prob = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy_with_logits(
        pred_logits, targets, reduction="none"
    )

    # Focal modulation
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * focal_weight * ce_loss
    return loss.mean()
