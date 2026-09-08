"""
CurvePT: Main model assembling all components.

RGB + Depth/Normals → DINOv2 + GeoEncoder → Fusion → Curve Queries → 3D Curves
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rgb_encoder import RGBEncoder
from .geo_encoder import GeometricEncoder
from .curve_decoder import CurveQueryDecoder


class CurvePT(nn.Module):
    """
    3D Curve Perception Transformer.

    End-to-end model: RGBD input → set of 3D DLO curves with visibility + confidence.
    """

    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        dc = config["data"]

        self.image_height = dc["image_height"]
        self.image_width = dc["image_width"]
        self.patch_size = dc["patch_size"]

        # 1. RGB Encoder (DINOv2 + LoRA)
        self.rgb_encoder = RGBEncoder(
            model_name=mc["rgb_backbone"],
            embed_dim=mc["rgb_embed_dim"],
            patch_size=dc["patch_size"],
            lora_rank=mc["lora_rank"],
            lora_alpha=mc["lora_alpha"],
            lora_num_blocks=mc["lora_num_blocks"],
            freeze_backbone=mc["freeze_backbone"],
            image_height=dc["image_height"],
            image_width=dc["image_width"],
        )

        # 2. Geometric Encoder (depth + normals)
        self.geo_encoder = GeometricEncoder(
            in_channels=mc["geo_in_channels"],
            embed_dim=mc["geo_embed_dim"],
            patch_size=dc["patch_size"],
            num_layers=mc["geo_num_layers"],
            image_height=dc["image_height"],
            image_width=dc["image_width"],
        )

        # 3. Cross-Modal Fusion
        fusion_type = mc.get("fusion_type", "spatial")
        if fusion_type == "spatial":
            from .fusion import SpatialCrossModalFusion
            self.fusion = SpatialCrossModalFusion(
                rgb_dim=mc["rgb_embed_dim"],
                geo_dim=mc["geo_embed_dim"],
                dropout=mc["decoder_dropout"],
                h_tokens=dc["image_height"] // dc["patch_size"],
                w_tokens=dc["image_width"] // dc["patch_size"],
            )
        else:
            self.fusion = CrossModalFusion(
                rgb_dim=mc["rgb_embed_dim"],
                geo_dim=mc["geo_embed_dim"],
                num_heads=mc["fusion_num_heads"],
                num_layers=mc["fusion_num_layers"],
                dropout=mc["decoder_dropout"],
            )

        # 4. Curve Query Decoder
        self.decoder = CurveQueryDecoder(
            d_model=mc["rgb_embed_dim"],
            num_queries=mc["num_queries"],
            num_nodes=mc["num_nodes"],
            num_decoder_layers=mc["num_decoder_layers"],
            nhead=mc["decoder_num_heads"],
            dim_feedforward=mc["decoder_dim_feedforward"],
            dropout=mc["decoder_dropout"],
            use_bspline=mc.get("use_bspline", True),
            num_ctrl_points=mc.get("num_ctrl_points", 12),
        )

        # Store config
        self.config = config

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor,
                normals: torch.Tensor) -> dict:
        """
        Args:
            rgb: (B, 3, H, W) in [0, 1]
            depth: (B, 1, H, W) in meters
            normals: (B, 3, H, W) unit surface normals

        Returns:
            dict with predictions from all decoder layers:
                'pred_coords': list of (B, Q, N, 3) — 3D skeleton points
                'pred_vis': list of (B, Q, N) — visibility logits
                'pred_conf': list of (B, Q, 1) — confidence logits
                'pred_radius': list of (B, Q, 1) — radius estimates
        """
        pad_height = (-rgb.shape[-2]) % self.patch_size
        pad_width = (-rgb.shape[-1]) % self.patch_size
        if pad_height or pad_width:
            padding = (0, pad_width, 0, pad_height)
            rgb = F.pad(rgb, padding, mode="replicate")
            depth = F.pad(depth, padding, mode="replicate")
            normals = F.pad(normals, padding, mode="replicate")

        h_t = rgb.shape[-2] // self.patch_size
        w_t = rgb.shape[-1] // self.patch_size

        # Encode RGB
        rgb_tokens = self.rgb_encoder(rgb)       # (B, N_tokens, 768)

        # Encode depth + normals
        geo_tokens = self.geo_encoder(depth, normals)  # (B, N_tokens, 256)

        # Fuse modalities
        scene_features = self.fusion(rgb_tokens, geo_tokens, h=h_t, w=w_t)  # (B, N_tokens, 768)

        # Decode curve queries
        predictions = self.decoder(scene_features)

        return predictions

    def predict(self, rgb: torch.Tensor, depth: torch.Tensor,
                normals: torch.Tensor, confidence_threshold: float = 0.5
                ) -> list[dict]:
        """
        Inference-friendly prediction: returns filtered curves per image.

        Returns:
            list of dicts (one per image in batch), each containing:
                'curves': list of {
                    'points_3d': (N, 3) ndarray,
                    'visibility': (N,) ndarray,
                    'confidence': float,
                    'radius': float,
                }
        """
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                preds = self.forward(rgb, depth, normals)

            # Use last decoder layer's predictions
            coords = preds["pred_coords"][-1]      # (B, Q, N, 3)
            vis = torch.sigmoid(preds["pred_vis"][-1])  # (B, Q, N)
            conf = torch.sigmoid(preds["pred_conf"][-1]).squeeze(-1)  # (B, Q)
            radius = preds["pred_radius"][-1].squeeze(-1)  # (B, Q)

            B = coords.shape[0]
            results = []
            for b in range(B):
                curves = []
                for q in range(coords.shape[1]):
                    if conf[b, q].item() > confidence_threshold:
                        curves.append({
                            "points_3d": coords[b, q].cpu().numpy(),
                            "visibility": vis[b, q].cpu().numpy(),
                            "confidence": conf[b, q].item(),
                            "radius": radius[b, q].item(),
                        })
                # Sort by confidence (highest first)
                curves.sort(key=lambda c: c["confidence"], reverse=True)
                results.append({"curves": curves})
        finally:
            if was_training:
                self.train()

        return results

    def get_param_groups(self, lr: float, backbone_lr_factor: float = 0.1
                        ) -> list[dict]:
        """
        Get parameter groups with different learning rates.

        - LoRA params: lr * backbone_lr_factor (slow adaptation)
        - Other trainable params: lr (full learning rate)
        """
        lora_params = self.rgb_encoder.get_lora_params()
        lora_ids = {id(p) for p in lora_params}

        other_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in lora_ids
        ]

        return [
            {"params": lora_params, "lr": lr * backbone_lr_factor,
             "name": "backbone_lora"},
            {"params": other_params, "lr": lr,
             "name": "main"},
        ]


def build_curvept(config: dict) -> CurvePT:
    """Factory function to build CurvePT from config dict."""
    return CurvePT(config)
