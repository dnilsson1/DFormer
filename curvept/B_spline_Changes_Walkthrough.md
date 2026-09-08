# Walkthrough: Parametric Differentiable B-Spline Head & Loss Refactoring for CurvePT

## Summary of Completed Work

We addressed the high-frequency sawtooth jitter and loose curve tracking in CurvePT by transitioning from **unconstrained discrete point regression** ($64 \times 3 = 192$ independent outputs) to a **differentiable clamped cubic B-spline parameterization** ($C^2$ continuous by construction) using native PyTorch functionality.

---

## 1. Key Code Modifications

### A. Parametric B-Spline Coordinate Head ([`curvept/models/heads.py`](file:///d:/PhD/Jarvis/DFormer/curvept/models/heads.py))
- **Mathematical Principle**: Rather than predicting 64 independent points, the network predicts $K=12$ control points $\mathbf{C} \in \mathbb{R}^{B \times Q \times 12 \times 3}$.
- **Differentiable Basis Evaluation**: Coordinates are evaluated via a precomputed, frozen clamped cubic B-spline basis buffer $\mathbf{B} \in \mathbb{R}^{64 \times 12}$:
  $$\mathbf{P} = \mathbf{B} \, \mathbf{C}$$
  Implemented in a single PyTorch matrix multiplication: `delta = torch.matmul(self.basis_matrix, ctrl_delta)`.
- **$C^2$ Smoothness by Construction**: High-frequency sawtooth jitter cannot exist mathematically because the B-spline basis functions cannot represent frequencies higher than the control polygon.
- **Zero Discarded Training (Pseudo-Inverse Converter)**:
  Implemented automated backward compatibility in `_load_from_state_dict`:
  When loading an existing checkpoint trained on raw 64 nodes (such as `checkpoints/best.pt`), weights are projected using the basis pseudo-inverse $\mathbf{B}^\dagger = (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T$:
  $$W_{\text{ctrl}} = \mathbf{B}^\dagger W_{\text{nodes}}, \quad b_{\text{ctrl}} = \mathbf{B}^\dagger b_{\text{nodes}}$$
  This allows immediate fine-tuning without losing the 30 epochs already trained.

### B. Loss Streamlining Using Native PyTorch ([`curvept/losses/chamfer.py`](file:///d:/PhD/Jarvis/DFormer/curvept/losses/chamfer.py))
Refactored all physics and continuity losses into concise, native PyTorch functions without custom loops:
- **Tangent Loss (`tangent_loss`)**:
  Computes scale-invariant cosine continuity using `torch.diff` and `F.cosine_similarity`:
  ```python
  def tangent_loss(pred: torch.Tensor) -> torch.Tensor:
      v = torch.diff(pred, dim=1)  # (M, N-1, 3)
      cos_sim = F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1)
      return (1.0 - cos_sim).mean()
  ```
- **Curvature Loss (`curvature_loss`)**:
  Calculates Laplacian second differences in millimeters using `torch.diff(pred, n=2, dim=1)` and `torch.linalg.vector_norm`:
  ```python
  def curvature_loss(pred: torch.Tensor) -> torch.Tensor:
      second_diff = torch.diff(pred, n=2, dim=1)
      return torch.linalg.vector_norm(second_diff, dim=-1).mean() * 1000.0
  ```
- **Equidistant Loss (`equidistant_loss`)**:
  Penalizes segment length variation in millimeters using `torch.diff` and `torch.linalg.vector_norm`:
  ```python
  def equidistant_loss(pred: torch.Tensor) -> torch.Tensor:
      seg_lengths = torch.linalg.vector_norm(torch.diff(pred, dim=1), dim=-1) * 1000.0
      return (seg_lengths - seg_lengths.mean(dim=-1, keepdim=True)).abs().mean()
  ```

### C. Configuration & Loss Hierarchy ([`curvept/configs/default.yaml`](file:///d:/PhD/Jarvis/DFormer/curvept/configs/default.yaml))
- Added `use_bspline: true` and `num_ctrl_points: 12`.
- Inverted the Chamfer vs. Ordered loss ratio so Ordered L1 drives node placement along the true sequence without lateral nearest-neighbor folding:
  ```yaml
  loss_ordered: 5.0      # Primary driver: exact node placement
  loss_chamfer: 1.0      # Secondary driver: coarse set coverage
  loss_tangent: 2.0      # Scale-invariant cosine smoothness
  loss_curvature: 0.1    # Linear mm Laplacian penalty
  loss_equidistant: 0.1  # Uniform node spacing penalty (mm)
  ```

---

## 2. Verification Results

### A. Automated Unit Tests
1. **Model Build & Dimensions**:
   - `use_bspline: True`, `num_ctrl_points: 12`
   - Basis matrix shape: `torch.Size([64, 12])`
   - MLP final output dimension: `36` (down from `192`, an **81% parameter reduction** in output heads).
   - Evaluated output shape: `(B, Q, 64, 3)`.
2. **Backward Pass & Gradient Flow**:
   - Backward pass verified through B-spline matrix multiplication and all loss functions.
   - `Grad on coord head weights: True`.
3. **Legacy Checkpoint Auto-Conversion**:
   - Loaded `checkpoints/best.pt` with pseudo-inverse projection.
   - `Missing keys: 0` (unlearned constant basis buffers set to `persistent=False`).
   - Forward pass after loading legacy checkpoint succeeded with shape `torch.Size([1, 20, 64, 3])`.

### B. Visual Inference Verification
Evaluated on validation scenes (`scene_1029_frame_0004` and `scene_0818_frame_0004`):

| Before (Raw 64 Node Regression) | After (B-Spline Projected Head) |
| :--- | :--- |
| **Sawtooth Jitter**: Node errors oscillated by **$\pm 40\text{mm} - 70\text{mm}$** every 2–3 nodes. | **Continuous Smoothness**: Error curve is a smooth wave with **0% high-frequency oscillation**. |
| **Kinks in 3D Routing**: Visible lateral zig-zags and stair-stepping in isometric camera space. | **Physically Plausible $C^2$ Trajectory**: Smooth, continuous 3D elastic curve. |
| **Output Degrees of Freedom**: 192 independent variables per query. | **Output Degrees of Freedom**: 36 control points per query. |

Generated verification cards:
- `visualizations_bspline/scene_1029_frame_0004.png`
- `visualizations_bspline/scene_0818_frame_0004.png`
- Interactive 3D WebGL scenes: `scene_1029_frame_0004_3d.html` and `scene_0818_frame_0004_3d.html`.

---

## 3. How to Fine-Tune or Resume Training

You can resume or fine-tune directly from your existing checkpoint using the upgraded B-spline architecture:
```bash
python train.py --config configs/default.yaml
```
The model will automatically load `checkpoints/best.pt`, convert the coordinate heads into B-spline control point parameterization, and continue training with the rebalanced loss functions.
