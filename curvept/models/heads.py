"""
Prediction Heads for CurvePT.

Each curve query produces multiple outputs through specialized MLP heads:
- 3D coordinates: 64 ordered skeleton nodes (N×3)
- Visibility: per-node occlusion/FOV score (N)
- Confidence: is this query a real DLO or background? (1)
- Radius: estimated cross-section radius (1)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple multi-layer perceptron with GELU activation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/dropout on last layer
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_clamped_cubic_bspline_basis(num_nodes: int = 64, num_ctrl: int = 12,
                                        degree: int = 3) -> torch.Tensor:
    """
    Compute clamped cubic B-spline basis evaluation matrix B in R^(num_nodes x num_ctrl).
    Clamped knots ensure exact interpolation at endpoints: p(0) = c_0, p(1) = c_{K-1}.
    """
    import numpy as np
    import scipy.interpolate as si

    knots = [0.0] * degree + list(np.linspace(0.0, 1.0, num_ctrl - degree + 1)) + [1.0] * degree
    t = np.linspace(0.0, 1.0, num_nodes)
    basis = np.zeros((num_nodes, num_ctrl), dtype=np.float32)
    for i in range(num_ctrl):
        c = np.zeros(num_ctrl, dtype=np.float32)
        c[i] = 1.0
        spl = si.BSpline(knots, c, degree)
        basis[:, i] = spl(t)
    return torch.from_numpy(basis)


class CoordinateHead(nn.Module):
    """
    Predict 3D skeleton coordinates for a curve query.

    Can operate in two modes:
    1. B-Spline mode (default, use_bspline=True):
       Predicts K control points (K << num_nodes) and evaluates them along a fixed,
       differentiable clamped cubic B-spline basis matrix B in R^(num_nodes x K).
       Guarantees C^2 continuous smoothness by construction. Eliminates
       high-frequency sawtooth jitter.
    2. Direct node mode (use_bspline=False):
       Directly predicts num_nodes * 3 coordinates (legacy behavior).

    Output: (B, Q, num_nodes, 3) — ordered 3D points in camera frame.
    Uses residual refinement: each decoder layer predicts a DELTA
    that's added to the previous layer's prediction.
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 256,
                 num_nodes: int = 64, num_layers: int = 3,
                 use_bspline: bool = True, num_ctrl: int = 12):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_bspline = use_bspline
        self.num_ctrl = num_ctrl

        if self.use_bspline:
            basis = compute_clamped_cubic_bspline_basis(num_nodes=num_nodes, num_ctrl=num_ctrl)
            self.register_buffer("basis_matrix", basis, persistent=False)
            out_dim = num_ctrl * 3
        else:
            self.register_buffer("basis_matrix", None, persistent=False)
            out_dim = num_nodes * 3

        self.mlp = MLP(d_model, hidden_dim, out_dim, num_layers=num_layers)

    def forward(self, query_features: torch.Tensor,
                prev_coords: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query_features: (B, Q, d_model) — refined query embeddings
            prev_coords: (B, Q, num_nodes, 3) — previous layer's prediction (for residual)

        Returns:
            coords: (B, Q, num_nodes, 3) — predicted 3D coordinates
        """
        B, Q, _ = query_features.shape

        if self.use_bspline:
            ctrl_delta = self.mlp(query_features).view(B, Q, self.num_ctrl, 3)
            # Differentiable linear evaluation along B-spline basis: (B, Q, num_nodes, 3)
            delta = torch.matmul(self.basis_matrix, ctrl_delta)
        else:
            delta = self.mlp(query_features).view(B, Q, self.num_nodes, 3)

        if prev_coords is not None:
            return prev_coords + delta  # Residual refinement
        return delta

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Handle backward compatibility:
        If checkpoint was trained with raw num_nodes * 3 outputs, project weights
        to num_ctrl * 3 control points using the basis pseudo-inverse (B^dagger).
        """
        if self.use_bspline:
            last_layer_idx = len(self.mlp.net) - 1
            weight_key = prefix + f"mlp.net.{last_layer_idx}.weight"
            bias_key = prefix + f"mlp.net.{last_layer_idx}.bias"

            if weight_key in state_dict and state_dict[weight_key].shape[0] == self.num_nodes * 3:
                old_weight = state_dict[weight_key]
                old_bias = state_dict[bias_key]
                hidden_dim = old_weight.shape[1]

                old_w = old_weight.view(self.num_nodes, 3, hidden_dim)
                old_b = old_bias.view(self.num_nodes, 3)

                # Pseudo-inverse of basis matrix B: (num_ctrl, num_nodes)
                B_pinv = torch.linalg.pinv(self.basis_matrix.to(old_weight.device))
                new_w = torch.einsum("kn, nmd -> kmd", B_pinv, old_w).reshape(self.num_ctrl * 3, hidden_dim)
                new_b = torch.einsum("kn, nd -> kd", B_pinv, old_b).reshape(self.num_ctrl * 3)

                state_dict[weight_key] = new_w
                state_dict[bias_key] = new_b

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class VisibilityHead(nn.Module):
    """
    Predict per-node visibility for each curve query.

    Output: (B, Q, num_nodes) — sigmoid probability that each node is
    visible (in FOV and not occluded).
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 128,
                 num_nodes: int = 64):
        super().__init__()
        self.mlp = MLP(d_model, hidden_dim, num_nodes, num_layers=2)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """Returns: (B, Q, num_nodes) logits (apply sigmoid for probabilities)."""
        return self.mlp(query_features)


class ConfidenceHead(nn.Module):
    """
    Predict whether a curve query corresponds to a real DLO.

    Output: (B, Q, 1) — sigmoid probability that this is a real detection
    (vs. background/no-object).
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.mlp = MLP(d_model, hidden_dim, 1, num_layers=2)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """Returns: (B, Q, 1) logits."""
        return self.mlp(query_features)


class RadiusHead(nn.Module):
    """
    Predict the cross-section radius of each DLO.

    Output: (B, Q, 1) — radius in meters (softplus activation for positivity).
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 64):
        super().__init__()
        self.mlp = MLP(d_model, hidden_dim, 1, num_layers=2)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """Returns: (B, Q, 1) positive radius values."""
        return nn.functional.softplus(self.mlp(query_features))
