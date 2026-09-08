"""
Curve Query Decoder: DETR-style transformer decoder with iterative curve refinement.

The key innovation: learnable curve queries attend to fused scene features,
and each decoder layer refines the predicted 3D curve coordinates.
Self-attention between queries enables inter-DLO reasoning (critical for
separating parallel cables in bundles).

Inspired by:
- DETR (Carion et al., 2020) — learnable object queries + Hungarian matching
- 3PT (Kalra et al., 2026) — iterative render-and-compare refinement
- Deformable DETR (Zhu et al., 2021) — sampling features at predicted locations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import CoordinateHead, VisibilityHead, ConfidenceHead, RadiusHead


class CurveDecoderLayer(nn.Module):
    """
    Single decoder layer with:
    1. Self-attention among curve queries (inter-DLO reasoning)
    2. Cross-attention from queries to scene features (localization)
    3. Feed-forward network
    """

    def __init__(self, d_model: int = 768, nhead: int = 12,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: queries attend to scene features
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor,
                scene_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, Q, d_model) — curve query embeddings
            scene_features: (B, N, d_model) — fused scene features

        Returns:
            refined_queries: (B, Q, d_model)
        """
        # 1. Self-attention (queries interact with each other)
        q = queries
        attn_out, _ = self.self_attn(q, q, q)
        queries = self.norm1(queries + attn_out)

        # 2. Cross-attention (queries attend to scene features)
        attn_out, _ = self.cross_attn(
            query=queries, key=scene_features, value=scene_features
        )
        queries = self.norm2(queries + attn_out)

        # 3. FFN
        queries = self.norm3(queries + self.ffn(queries))

        return queries


class CurveQueryDecoder(nn.Module):
    """
    Full curve query decoder with iterative refinement.

    Architecture:
    - Q learnable query embeddings (one per potential DLO)
    - L decoder layers, each producing refined predictions
    - Auxiliary losses at each layer (deep supervision)
    """

    def __init__(
        self,
        d_model: int = 768,
        num_queries: int = 20,
        num_nodes: int = 64,
        num_decoder_layers: int = 6,
        nhead: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_bspline: bool = True,
        num_ctrl_points: int = 12,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_nodes = num_nodes
        self.num_decoder_layers = num_decoder_layers
        self.use_bspline = use_bspline
        self.num_ctrl_points = num_ctrl_points

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            CurveDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Prediction heads: decoupled coordinate head per layer for macro vs residual refinement
        self.coord_heads = nn.ModuleList([
            CoordinateHead(
                d_model, hidden_dim=256, num_nodes=num_nodes, num_layers=3,
                use_bspline=use_bspline, num_ctrl=num_ctrl_points
            )
            for _ in range(num_decoder_layers)
        ])
        self.vis_head = VisibilityHead(d_model, hidden_dim=128,
                                       num_nodes=num_nodes)
        self.conf_head = ConfidenceHead(d_model, hidden_dim=128)
        self.radius_head = RadiusHead(d_model, hidden_dim=64)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Handle legacy checkpoints that used a single shared coord_head."""
        old_prefix = prefix + "coord_head."
        has_old_head = any(k.startswith(old_prefix) for k in state_dict.keys())
        if has_old_head:
            for i in range(self.num_decoder_layers):
                for k in list(state_dict.keys()):
                    if k.startswith(old_prefix):
                        new_k = prefix + f"coord_heads.{i}." + k[len(old_prefix):]
                        if new_k not in state_dict:
                            state_dict[new_k] = state_dict[k].clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, scene_features: torch.Tensor) -> dict:
        """
        Args:
            scene_features: (B, N, d_model) — fused RGB+depth features

        Returns:
            dict with keys:
                'pred_coords': list[Tensor] — (B, Q, num_nodes, 3) per layer
                'pred_vis': list[Tensor] — (B, Q, num_nodes) per layer
                'pred_conf': list[Tensor] — (B, Q, 1) per layer
                'pred_radius': list[Tensor] — (B, Q, 1) per layer
        """
        B = scene_features.shape[0]

        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        # (B, Q, d_model)

        # Iterative refinement
        all_coords = []
        all_vis = []
        all_conf = []
        all_radius = []

        prev_coords = None

        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            # Refine queries via self-attention + cross-attention
            queries = decoder_layer(queries, scene_features)

            # Predict outputs from refined queries (layer-specific coord head)
            coords = self.coord_heads[layer_idx](queries, prev_coords)
            vis = self.vis_head(queries)
            conf = self.conf_head(queries)
            radius = self.radius_head(queries)

            all_coords.append(coords)
            all_vis.append(vis)
            all_conf.append(conf)
            all_radius.append(radius)

            # Use current coords as reference for next layer's residual
            prev_coords = coords.detach()  # Detach to prevent gradient explosion

        return {
            "pred_coords": all_coords,     # List of (B, Q, N, 3)
            "pred_vis": all_vis,            # List of (B, Q, N)
            "pred_conf": all_conf,          # List of (B, Q, 1)
            "pred_radius": all_radius,      # List of (B, Q, 1)
        }
