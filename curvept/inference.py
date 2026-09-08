"""
CurvePT Inference Script.

Load a trained model and run 3D curve estimation on RGBD input.
Visualizes results by projecting predicted 3D curves onto the RGB image.

Usage:
    python inference.py --checkpoint checkpoints/best.pt --image scene/frame.hdf5
    python inference.py --checkpoint checkpoints/best.pt --rgb image.png --depth depth.npy
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from models.curvept import build_curvept


def load_model(checkpoint_path: str, config_path: str = None,
               device: str = "cuda") -> tuple:
    """Load trained CurvePT model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try to find config
    if config_path is None:
        config_path = str(Path(checkpoint_path).parent.parent / "configs" / "default.yaml")
    if not os.path.exists(config_path):
        config_path = "configs/default.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = build_curvept(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config


def load_rgbd(hdf5_path: str = None, rgb_path: str = None,
              depth_path: str = None, max_depth: float = 3.0) -> tuple:
    """Load RGBD data from HDF5 or separate files."""
    if hdf5_path:
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            rgb = np.array(f["colors"], dtype=np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            depth = np.array(f["depth"], dtype=np.float32)
            normals = np.array(f.get("normals",
                                     np.zeros_like(rgb)), dtype=np.float32)
    else:
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB) / 255.0
        rgb = rgb.astype(np.float32)
        depth = np.load(depth_path).astype(np.float32)
        normals = np.zeros_like(rgb, dtype=np.float32)

    # Normalize depth
    depth = np.clip(depth, 0, max_depth) / max_depth

    # To tensors (B, C, H, W)
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    normals_t = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0)

    return rgb_t, depth_t, normals_t, rgb


def project_3d_to_2d(points_3d: np.ndarray,
                     fx: float = 605.2, fy: float = 605.1,
                     cx: float = 425.7, cy: float = 246.0) -> np.ndarray:
    """Project 3D points to 2D image coordinates."""
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z = np.maximum(z, 1e-6)
    u = fx * x / z + cx
    v = fy * y / z + cy
    return np.stack([u, v], axis=-1)


def visualize_predictions(rgb: np.ndarray, predictions: list[dict],
                          output_path: str = None, show: bool = True):
    """
    Visualize predicted 3D curves projected onto the RGB image.

    Each detected DLO is drawn with a different color, with transparency
    based on predicted visibility.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.imshow(rgb)

    # Color palette for different DLOs
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    curves = predictions[0]["curves"]  # First (and only) image in batch
    for i, curve in enumerate(curves):
        pts_3d = curve["points_3d"]
        vis = curve["visibility"]
        conf = curve["confidence"]
        radius = curve["radius"]

        # Project to 2D
        pts_2d = project_3d_to_2d(pts_3d)

        # Draw curve with visibility-based alpha
        color = colors[i % len(colors)]
        for j in range(len(pts_2d) - 1):
            alpha = min(vis[j], vis[j + 1]) * 0.8 + 0.2
            ax.plot(pts_2d[j:j+2, 0], pts_2d[j:j+2, 1],
                    color=color, linewidth=2, alpha=alpha)

        # Label
        mid_idx = len(pts_2d) // 2
        ax.annotate(
            f"DLO {i} ({conf:.2f}, r={radius*1000:.1f}mm)",
            pts_2d[mid_idx], color="white", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc=color[:3], alpha=0.7)
        )

    ax.set_title(f"CurvePT: {len(curves)} DLOs detected")
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="CurvePT Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML path")
    parser.add_argument("--image", type=str, default=None,
                        help="HDF5 frame path")
    parser.add_argument("--rgb", type=str, default=None,
                        help="RGB image path (if not using HDF5)")
    parser.add_argument("--depth", type=str, default=None,
                        help="Depth map path (if not using HDF5)")
    parser.add_argument("--output", type=str, default="prediction.png",
                        help="Output visualization path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for detection")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.config, device)

    # Load input
    print("Loading RGBD input...")
    rgb_t, depth_t, normals_t, rgb_np = load_rgbd(
        hdf5_path=args.image, rgb_path=args.rgb, depth_path=args.depth,
        max_depth=config["data"]["max_depth"]
    )
    rgb_t = rgb_t.to(device)
    depth_t = depth_t.to(device)
    normals_t = normals_t.to(device)

    # Inference
    print("Running inference...")
    t0 = __import__("time").time()
    predictions = model.predict(rgb_t, depth_t, normals_t,
                                confidence_threshold=args.threshold)
    dt = __import__("time").time() - t0

    n_curves = len(predictions[0]["curves"])
    print(f"  Detected {n_curves} DLOs in {dt*1000:.1f}ms")

    for i, curve in enumerate(predictions[0]["curves"]):
        print(f"    DLO {i}: conf={curve['confidence']:.3f}, "
              f"r={curve['radius']*1000:.1f}mm, "
              f"vis={curve['visibility'].mean():.2f}")

    # Visualize
    visualize_predictions(rgb_np, predictions, output_path=args.output)


if __name__ == "__main__":
    main()
