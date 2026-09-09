"""
Geometric Encoder: Lightweight network for depth + surface normals.

Processes 4-channel input (depth + XYZ normals) into geometric feature tokens
aligned with DINOv2's patch grid. Trained from scratch on synthetic data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: depthwise conv + pointwise FFN."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # → (B, C, H, W)
        return x + residual


class GeometricEncoder(nn.Module):
    """
    Encode depth map + surface normals into geometric feature tokens.

    Architecture:
    1. Patch embedding: Conv2d with stride=patch_size to match DINOv2 grid
    2. 4× ConvNeXt blocks for local geometric processing
    3. Output: (B, H_tokens * W_tokens, geo_embed_dim)
    """

    def __init__(
        self,
        in_channels: int = 4,        # depth(1) + normals(3)
        embed_dim: int = 256,         # Output feature dimension
        patch_size: int = 14,         # Must match DINOv2
        num_layers: int = 4,          # ConvNeXt blocks
        image_height: int = 720,
        image_width: int = 1280,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.h_tokens = image_height // patch_size
        self.w_tokens = image_width // patch_size

        # Patch embedding — project each 14×14 patch into embed_dim
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=7, stride=7, padding=3),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
        )
        # Total stride = 2 × 7 = 14 → matches DINOv2 patch grid

        # Local geometric processing
        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(embed_dim) for _ in range(num_layers)]
        )

        # Positional encoding (learnable, matches DINOv2 grid size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, self.h_tokens, self.w_tokens) * 0.02
        )

    def forward(self, depth: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, H, W) depth map in meters, 0 = invalid
            normals: (B, 3, H, W) surface normals (unit vectors)

        Returns:
            features: (B, H_tokens * W_tokens, embed_dim)
        """
        # Concatenate depth + normals
        x = torch.cat([depth, normals], dim=1)  # (B, 4, H, W)

        # Patch embedding (stride=14 total)
        x = self.patch_embed(x)  # (B, embed_dim, H/14, W/14)

        # Add positional encoding
        pos_embed = self.pos_embed
        if pos_embed.shape[2:] != x.shape[2:]:
            pos_embed = F.interpolate(
                pos_embed, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        x = x + pos_embed

        # Local geometric processing
        x = self.blocks(x)  # (B, embed_dim, H_t, W_t)

        # Flatten to token sequence
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        return x
