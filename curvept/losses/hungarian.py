"""
Hungarian Matching for bipartite assignment between predicted and GT curves.

Like DETR, we solve a linear assignment problem to find the optimal 1-to-1
matching between Q predicted curves and K ground truth curves (K ≤ Q).
Unmatched predictions are assigned the "no object" target.
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_match(
    pred_coords: torch.Tensor,
    pred_conf: torch.Tensor,
    pred_vis: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_vis: torch.Tensor,
    num_curves: torch.Tensor = None,
    cost_chamfer: float = 5.0,
    cost_ordered: float = 2.0,
    cost_vis: float = 1.0,
    cost_conf: float = 1.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Perform Hungarian matching between predictions and ground truth.

    Vectorized implementation solving the assignment problem on the
    exact Q x K_b valid ground-truth curves per batch sample.

    Args:
        pred_coords: (B, Q, N, 3) predicted 3D curve points
        pred_conf: (B, Q, 1) predicted confidence logits
        pred_vis: (B, Q, N) predicted visibility logits
        gt_coords: (B, K_max, N, 3) ground truth curve points
        gt_vis: (B, K_max, N) ground truth visibility masks
        num_curves: (B,) number of valid GT curves per sample
        cost_chamfer: weight for Chamfer distance in cost matrix
        cost_ordered: weight for ordered L1 distance in cost matrix
        cost_vis: weight for visibility matching cost
        cost_conf: weight for confidence cost

    Returns:
        matches: list of (pred_indices, gt_indices) tuples per batch element
    """
    B, Q, N, _ = pred_coords.shape
    device = pred_coords.device
    matches = []

    for b in range(B):
        if num_curves is not None:
            K = int(num_curves[b].item())
        else:
            K = gt_coords.shape[1]

        if K == 0:
            matches.append((
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            ))
            continue

        gt_c = gt_coords[b, :K]  # (K, N, 3)
        gt_v = gt_vis[b, :K]     # (K, N)
        pred_c = pred_coords[b]  # (Q, N, 3)
        pred_v = pred_vis[b]     # (Q, N)
        pred_cf = pred_conf[b, :, 0]  # (Q,)

        # 1. Vectorized Chamfer Distance using torch.cdist: (Q, K)
        # Compute pairwise distances for each (q,k) pair without 5D broadcast
        # pred_c: (Q, N, 3), gt_c: (K, N, 3)
        # Expand and reshape for batched cdist: (Q*K, N, 3)
        pred_exp = pred_c.unsqueeze(1).expand(-1, K, -1, -1).reshape(Q * K, N, 3)
        gt_exp = gt_c.unsqueeze(0).expand(Q, -1, -1, -1).reshape(Q * K, N, 3)
        dist_sq = torch.cdist(pred_exp, gt_exp, p=2.0).pow(2)  # (Q*K, N, N)
        dist_sq = dist_sq.view(Q, K, N, N)

        min_pred_to_gt, _ = dist_sq.min(dim=-1)  # (Q, K, N)
        loss_pred = min_pred_to_gt.mean(dim=-1)   # (Q, K)

        min_gt_to_pred, _ = dist_sq.min(dim=-2)  # (Q, K, N)
        mask = gt_v[None, :, :]                   # (1, K, N)
        mask_sum = mask.sum(dim=-1).clamp(min=1e-6)
        loss_gt = (min_gt_to_pred * mask).sum(dim=-1) / mask_sum  # (Q, K)

        c_chamfer = 0.5 * (loss_pred + loss_gt)

        # 2. Vectorized Direction-Invariant Ordered L1: (Q, K)
        diff_fwd = (pred_c[:, None, :, :] - gt_c[None, :, :, :]).abs().sum(dim=-1)  # (Q, K, N)
        gt_c_rev = gt_c.flip(dims=[1])
        diff_rev = (pred_c[:, None, :, :] - gt_c_rev[None, :, :, :]).abs().sum(dim=-1)

        mask_rev = gt_v.flip(dims=[1])[None, :, :]
        fwd_total = (diff_fwd * mask).sum(dim=-1) / mask_sum
        rev_total = (diff_rev * mask_rev).sum(dim=-1) / mask_rev.sum(dim=-1).clamp(min=1e-6)
        c_ordered = torch.min(fwd_total, rev_total)

        # 3. Vectorized Visibility BCE: (Q, K)
        c_vis = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_v[:, None, :].expand(-1, K, -1),
            gt_v[None, :, :].expand(Q, -1, -1),
            reduction="none",
        ).mean(dim=-1)

        # 4. Confidence cost (higher confidence -> lower cost for real DLOs): (Q, K)
        c_conf = -torch.sigmoid(pred_cf)[:, None].expand(-1, K)

        # Total cost matrix (Q, K)
        cost = (cost_chamfer * c_chamfer +
                cost_ordered * c_ordered +
                cost_vis * c_vis +
                cost_conf * c_conf)

        # Linear sum assignment on rectangular (Q, K) cost matrix
        cost_np = cost.detach().cpu().numpy()
        pred_idx, gt_idx = linear_sum_assignment(cost_np)

        matches.append((
            torch.tensor(pred_idx, dtype=torch.long, device=device),
            torch.tensor(gt_idx, dtype=torch.long, device=device),
        ))

    return matches
