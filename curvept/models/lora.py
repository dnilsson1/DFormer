"""
LoRA (Low-Rank Adaptation) layers for efficient fine-tuning of frozen transformers.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
Applied to DINOv2 attention Q,V projections for domain adaptation (synthetic → real).
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with low-rank adaptation.

    Computes: y = W_frozen @ x + (B @ A) @ x * (alpha / rank)
    where A ∈ R^{rank × in}, B ∈ R^{out × rank} are trainable,
    and W_frozen is the original frozen weight matrix.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        self.linear = original_linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Initialize A with Kaiming, B with zeros → initial LoRA output is zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen forward
        out = self.linear(x)
        # LoRA delta: (B @ A) @ x, scaled
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}")


# NOTE: This utility is not currently used — LoRA injection is done inline
# in RGBEncoder.__init__ for more precise control over which layers to adapt.
def inject_lora(model: nn.Module, target_modules: list[str],
                rank: int = 8, alpha: float = 16.0) -> nn.Module:
    """
    Inject LoRA adapters into specified linear layers of a model.

    Args:
        model: The model to modify (e.g., DINOv2 ViT)
        target_modules: List of attribute names to replace (e.g., ["qkv"] or ["q_proj", "v_proj"])
        rank: LoRA rank
        alpha: LoRA scaling factor

    Returns:
        Modified model with LoRA layers injected
    """
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                original = getattr(module, target)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
                    setattr(module, target, lora_layer)
    return model


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Extract only the LoRA parameters from a model (for optimizer)."""
    params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            params.append(param)
    return params


def count_lora_params(model: nn.Module) -> tuple[int, int]:
    """Count (trainable_lora_params, total_params)."""
    lora = sum(p.numel() for p in get_lora_params(model))
    total = sum(p.numel() for p in model.parameters())
    return lora, total
