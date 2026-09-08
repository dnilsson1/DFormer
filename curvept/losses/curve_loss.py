"""
Composite Curve Loss: combines all loss components with Hungarian matching.

Training pipeline:
1. Run Hungarian matching to pair predicted curves with GT curves
2. Compute supervised losses on matched pairs
3. Compute "no object" loss on unmatched predictions
4. Apply auxiliary losses from intermediate decoder layers (deep supervision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chamfer import chamfer_distance, ordered_l1_loss, curvature_loss, equidistant_loss, tangent_loss
from .focal import focal_loss
from .hungarian import hungarian_match


class CurvePTLoss(nn.Module):
    """
    Full loss function for CurvePT training.

    Combines:
    - Chamfer distance (shape similarity)
    - Ordered L1 (curve ordering preservation)
    - Tangent loss (scale-invariant cosine continuity)
    - Curvature loss (Laplacian smoothness / bend penalty in mm)
    - Equidistant loss (uniform node spacing along curve in mm)
    - Visibility BCE (per-node occlusion prediction)
    - Confidence focal loss (real vs background detection)
    - Radius L1 (cross-section estimation)
    - Auxiliary losses from intermediate decoder layers
    """

    def __init__(self, config: dict):
        super().__init__()
        tc = config["training"]
        self.w_chamfer = tc["loss_chamfer"]
        self.w_ordered = tc["loss_ordered"]
        self.w_tangent = tc.get("loss_tangent", 2.0)
        self.w_curv = tc.get("loss_curvature", 0.1)
        self.w_equi = tc.get("loss_equidistant", 0.1)
        self.w_vis = tc["loss_visibility"]
        self.w_conf = tc["loss_confidence"]
        self.w_radius = tc["loss_radius"]
        self.w_aux = tc["loss_aux_weight"]

        self.match_cost_chamfer = tc["match_cost_chamfer"]
        self.match_cost_ordered = tc.get("match_cost_ordered", 2.0)
        self.match_cost_vis = tc["match_cost_visibility"]
        self.match_cost_conf = tc["match_cost_confidence"]

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Args:
            predictions: dict from CurvePT.forward() with lists of per-layer preds
            targets: dict with:
                'coords': (B, K, N, 3) GT 3D points
                'visibility': (B, K, N) GT visibility masks
                'radius': (B, K) GT radii
                'num_curves': (B,) number of valid GT curves per sample

        Returns:
            losses: dict with total loss and per-component breakdowns
        """
        gt_coords = targets["coords"]
        gt_vis = targets["visibility"]
        gt_radius = targets["radius"]
        num_curves = targets["num_curves"]

        num_layers = len(predictions["pred_coords"])
        total_loss = torch.tensor(0.0, device=gt_coords.device)
        loss_breakdown = {}

        for layer_idx in range(num_layers):
            pred_coords = predictions["pred_coords"][layer_idx]
            pred_vis = predictions["pred_vis"][layer_idx]
            pred_conf = predictions["pred_conf"][layer_idx]
            pred_radius = predictions["pred_radius"][layer_idx]

            # Layer-wise Hungarian matching on valid GT curves
            matches = hungarian_match(
                pred_coords,
                pred_conf,
                pred_vis,
                gt_coords,
                gt_vis,
                num_curves=num_curves,
                cost_chamfer=self.match_cost_chamfer,
                cost_ordered=self.match_cost_ordered,
                cost_vis=self.match_cost_vis,
                cost_conf=self.match_cost_conf,
            )

            # Weight: last layer gets full weight, earlier layers get aux_weight
            layer_weight = 1.0 if layer_idx == num_layers - 1 else self.w_aux

            # Compute losses per batch element
            layer_loss = self._compute_layer_loss(
                pred_coords, pred_vis, pred_conf, pred_radius,
                gt_coords, gt_vis, gt_radius, num_curves, matches
            )

            for key, val in layer_loss.items():
                total_loss = total_loss + val * layer_weight
                loss_key = f"L{layer_idx}_{key}"
                loss_breakdown[loss_key] = val.item()

        loss_breakdown["total"] = total_loss.item()
        return {"loss": total_loss, "breakdown": loss_breakdown}

    def _compute_layer_loss(
        self,
        pred_coords, pred_vis, pred_conf, pred_radius,
        gt_coords, gt_vis, gt_radius, num_curves, matches
    ) -> dict:
        """Compute all loss components for a single decoder layer."""
        B = pred_coords.shape[0]
        Q = pred_coords.shape[1]
        device = pred_coords.device

        loss_chamfer = torch.tensor(0.0, device=device)
        loss_ordered = torch.tensor(0.0, device=device)
        loss_tangent = torch.tensor(0.0, device=device)
        loss_curv = torch.tensor(0.0, device=device)
        loss_equi = torch.tensor(0.0, device=device)
        loss_vis = torch.tensor(0.0, device=device)
        loss_radius = torch.tensor(0.0, device=device)

        # Build confidence targets: 1 for matched queries, 0 for unmatched
        conf_targets = torch.zeros(B, Q, device=device)

        # Gather all matched pairs across the batch into contiguous tensors
        all_matched_pred = []
        all_matched_gt = []
        all_matched_vis_gt = []
        all_matched_vis_pred = []
        all_matched_radius_pred = []
        all_matched_radius_gt = []

        for b in range(B):
            pred_idx, gt_idx = matches[b]
            K = int(num_curves[b].item())

            if len(pred_idx) == 0 or K == 0:
                continue

            # Mark matched queries as positive for confidence loss
            conf_targets[b, pred_idx] = 1.0

            all_matched_pred.append(pred_coords[b, pred_idx])
            all_matched_gt.append(gt_coords[b, gt_idx])
            all_matched_vis_gt.append(gt_vis[b, gt_idx])
            all_matched_vis_pred.append(pred_vis[b, pred_idx])
            all_matched_radius_pred.append(pred_radius[b, pred_idx, 0])
            all_matched_radius_gt.append(gt_radius[b, gt_idx])

        if all_matched_pred:
            # Concatenate all matched pairs: (M_total, N, 3), etc.
            cat_pred = torch.cat(all_matched_pred, dim=0)
            cat_gt = torch.cat(all_matched_gt, dim=0)
            cat_vis_gt = torch.cat(all_matched_vis_gt, dim=0)
            cat_vis_pred = torch.cat(all_matched_vis_pred, dim=0)
            cat_rad_pred = torch.cat(all_matched_radius_pred, dim=0)
            cat_rad_gt = torch.cat(all_matched_radius_gt, dim=0)

            # Compute all geometry losses in single batched calls
            loss_chamfer = chamfer_distance(cat_pred, cat_gt, mask=cat_vis_gt)
            loss_ordered = ordered_l1_loss(cat_pred, cat_gt, visibility=cat_vis_gt)
            loss_tangent = tangent_loss(cat_pred)
            loss_curv = curvature_loss(cat_pred)
            loss_equi = equidistant_loss(cat_pred)
            loss_vis = F.binary_cross_entropy_with_logits(
                cat_vis_pred, cat_vis_gt, reduction="mean"
            )
            loss_radius = F.l1_loss(cat_rad_pred, cat_rad_gt)

        # Confidence focal loss (all queries, matched + unmatched)
        conf_logits = pred_conf.squeeze(-1)  # (B, Q)
        loss_conf = focal_loss(conf_logits, conf_targets)

        return {
            "chamfer": loss_chamfer * self.w_chamfer,
            "ordered": loss_ordered * self.w_ordered,
            "tangent": loss_tangent * self.w_tangent,
            "curv": loss_curv * self.w_curv,
            "equi": loss_equi * self.w_equi,
            "vis": loss_vis * self.w_vis,
            "conf": loss_conf * self.w_conf,
            "radius": loss_radius * self.w_radius,
        }
