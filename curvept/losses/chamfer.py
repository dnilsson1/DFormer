"""
Chamfer Distance and Ordered Curve Loss.

Two complementary losses for 3D curve estimation:
- Chamfer: bidirectional nearest-neighbor distance (shape similarity, order-agnostic)
- Ordered L1: respects curve ordering (the robot needs to know which end is which)
"""

import torch
import torch.nn.functional as F


def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor,
                     mask: torch.Tensor = None) -> torch.Tensor:
    """
    Bidirectional Chamfer distance between two point sets.

    Args:
        pred: (B, N, 3) predicted points
        gt: (B, M, 3) ground truth points
        mask: (B, M) optional mask (1 = valid GT point, 0 = ignore)

    Returns:
        loss: scalar Chamfer distance
    """
    # (B, N, M) pairwise squared distances
    dist_sq = torch.cdist(pred, gt, p=2.0).pow(2)  # (B, N, M)

    # pred → gt: for each predicted point, find nearest GT point
    min_pred_to_gt, _ = dist_sq.min(dim=2)  # (B, N)

    # gt → pred: for each GT point, find nearest predicted point
    min_gt_to_pred, _ = dist_sq.min(dim=1)  # (B, M)

    if mask is not None:
        # Only consider valid GT points
        min_gt_to_pred = min_gt_to_pred * mask
        loss_gt = min_gt_to_pred.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss_pred = min_pred_to_gt.mean(dim=1)
    else:
        loss_pred = min_pred_to_gt.mean(dim=1)
        loss_gt = min_gt_to_pred.mean(dim=1)

    # Bidirectional: average of both directions
    loss = (loss_pred + loss_gt) / 2.0
    return loss.mean()


def ordered_l1_loss(pred: torch.Tensor, gt: torch.Tensor,
                    visibility: torch.Tensor = None) -> torch.Tensor:
    """
    Order-aware L1 loss between corresponding curve nodes.

    Handles direction ambiguity: computes loss in both directions
    and takes the minimum (a curve can be traversed either way).

    Args:
        pred: (B, N, 3) predicted ordered points
        gt: (B, N, 3) ground truth ordered points
        visibility: (B, N) optional weights (visible nodes contribute more)

    Returns:
        loss: scalar ordered L1 loss
    """
    # Forward direction
    forward_loss = (pred - gt).abs().sum(dim=-1)  # (B, N)

    # Reverse direction (flip GT ordering)
    gt_reversed = gt.flip(dims=[1])
    reverse_loss = (pred - gt_reversed).abs().sum(dim=-1)  # (B, N)

    if visibility is not None:
        forward_loss = forward_loss * visibility
        reverse_loss = reverse_loss * visibility.flip(dims=[1])
        # Normalize by number of visible nodes
        norm = visibility.sum(dim=1, keepdim=True).clamp(min=1)
        forward_total = forward_loss.sum(dim=1) / norm.squeeze(1)
        reverse_total = reverse_loss.sum(dim=1) / norm.squeeze(1)
    else:
        forward_total = forward_loss.mean(dim=1)
        reverse_total = reverse_loss.mean(dim=1)

    # Take minimum of both directions (handles direction ambiguity)
    loss = torch.min(forward_total, reverse_total)
    return loss.mean()


def tangent_loss(pred: torch.Tensor) -> torch.Tensor:
    """Scale-invariant cosine tangent continuity loss along the curve."""
    if pred.shape[0] == 0 or pred.shape[1] < 3:
        return torch.tensor(0.0, device=pred.device)
    v = torch.diff(pred, dim=1)  # (M, N-1, 3)
    cos_sim = F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1)  # (M, N-2)
    # Clamp to avoid negative loss from float precision overshoot
    return (1.0 - cos_sim.clamp(max=1.0)).mean()


def curvature_loss(pred: torch.Tensor) -> torch.Tensor:
    """Laplacian second-derivative penalty along the curve in millimeters."""
    if pred.shape[0] == 0 or pred.shape[1] < 3:
        return torch.tensor(0.0, device=pred.device)
    second_diff = torch.diff(pred, n=2, dim=1)  # (M, N-2, 3)
    # Add epsilon to avoid NaN gradients when nodes overlap exactly
    return torch.linalg.vector_norm(second_diff, ord=2, dim=-1).clamp(min=1e-8).mean() * 1000.0


def equidistant_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalizes non-uniform spacing between adjacent nodes (in mm)."""
    if pred.shape[0] == 0 or pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device)
    # clamp to avoid NaN gradients when adjacent nodes coincide
    seg_lengths = torch.linalg.vector_norm(torch.diff(pred, dim=1), ord=2, dim=-1).clamp(min=1e-8) * 1000.0
    mean_length = seg_lengths.mean(dim=-1, keepdim=True)
    return (seg_lengths - mean_length).abs().mean()
