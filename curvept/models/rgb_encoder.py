"""
RGB Encoder: DINOv2-B/14 backbone with LoRA adapters.

Provides rich, domain-agnostic visual features. The self-supervised DINOv2
features transfer much better from synthetic to real domains than
ImageNet-supervised features.
"""

import torch
import torch.nn as nn

from .lora import inject_lora, get_lora_params


class RGBEncoder(nn.Module):
    """
    DINOv2 Vision Transformer backbone with LoRA fine-tuning.

    Output: (B, H_tokens, W_tokens, embed_dim) feature grid
    where H_tokens = H // patch_size, W_tokens = W // patch_size
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        embed_dim: int = 768,
        patch_size: int = 14,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_num_blocks: int = 4,
        freeze_backbone: bool = True,
        image_height: int = 720,
        image_width: int = 1280,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.h_tokens = image_height // patch_size
        self.w_tokens = image_width // patch_size

        # Load DINOv2 from torch.hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )

        # Freeze all backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        # Inject LoRA into the last N transformer blocks
        # DINOv2 uses "qkv" as a single fused linear layer in each attention block
        num_blocks = len(self.backbone.blocks)
        target_block_indices = list(range(num_blocks - lora_num_blocks, num_blocks))

        for idx in target_block_indices:
            block = self.backbone.blocks[idx]
            attn = block.attn
            # DINOv2's attention has a fused qkv projection
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                from .lora import LoRALinear
                attn.qkv = LoRALinear(attn.qkv, rank=lora_rank, alpha=lora_alpha)

        # ImageNet normalization (DINOv2 expects this)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) in [0, 1] range

        Returns:
            features: (B, H_tokens * W_tokens, embed_dim)
        """
        # Normalize
        x = (rgb - self.mean) / self.std

        # DINOv2 forward — get patch tokens (exclude CLS token)
        # Use get_intermediate_layers for clean patch token extraction
        features = self.backbone.forward_features(x)

        # Extract patch tokens (remove CLS token if present)
        patch_tokens = features["x_norm_patchtokens"]  # (B, N_patches, embed_dim)

        return patch_tokens

    def get_lora_params(self) -> list[nn.Parameter]:
        """Get only LoRA parameters for separate LR scheduling."""
        return get_lora_params(self.backbone)

    @property
    def num_tokens(self) -> int:
        return self.h_tokens * self.w_tokens
