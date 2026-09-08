"""
CurvePT Comprehensive Inference & Visualization Suite.

Generates high-resolution multi-panel verification figures comparing
predicted 3D curves against ground truth on validation frames,
individual scenes, or raw RGB-D inputs.

Visualizations generated per sample:
  1. RGB View: High-contrast 2D curve overlay (GT dashed vs. Pred solid)
  2. Depth Map: Depth colormap (inferno) with projected 3D spatial alignment
  3. 3D Isometric View: True physical 3D spatial routing in camera coordinates (meters)
  4. Node Error Profile: Head-to-tail Euclidean error curve (mm) with PCK thresholds

Usage:
  # 1. Run on a specific scene/frame directory:
  python visualize_eval.py --checkpoint checkpoints/best.pt --frame_dir "F:/.../scene_0001/frame_0000"

  # 2. Run on 10 diverse validation samples and save gallery + report:
  python visualize_eval.py --checkpoint checkpoints/best.pt --num_samples 10 --output_dir vis_results

  # 3. Run on raw RGB-D image (with on-the-fly surface normal computation):
  python visualize_eval.py --checkpoint checkpoints/best.pt --rgb my_image.png --depth my_depth.npz
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/background rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent))

from models.curvept import build_curvept
from data.dlo_dataset import DLODataset
from losses.hungarian import hungarian_match
from losses.chamfer import chamfer_distance


def compute_surface_normals(depth: np.ndarray, fx: float = 605.2, fy: float = 605.1,
                            cx: float = 640.0, cy: float = 360.0) -> np.ndarray:
    """Compute 3-channel surface normals on-the-fly from depth using camera intrinsics."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 3D back-projection
    z = np.clip(depth, 0.05, 10.0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    # Gradients along X and Y
    dz_dx = np.gradient(points, axis=1)
    dz_dy = np.gradient(points, axis=0)

    # Normal vector via cross product
    normals = np.cross(dz_dx, dz_dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, np.maximum(norm, 1e-6))
    
    # Flip if pointing away from camera
    normals = np.where(normals[..., 2:3] > 0, -normals, normals)
    return normals.astype(np.float32)


def project_3d_to_2d(pts_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D camera-frame points (N, 3) to 2D image coordinates (N, 2)."""
    x, y, z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
    safe_z = np.maximum(z, 1e-4)
    u = K[0, 0] * (x / safe_z) + K[0, 2]
    v = K[1, 1] * (y / safe_z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def generate_sample_figure(
    rgb: np.ndarray,
    depth: np.ndarray,
    gt_curves: list[dict],
    pred_curves: list[dict],
    matches: list[tuple[int, int]],
    K: np.ndarray,
    output_path: str,
    title_suffix: str = "",
):
    """
    Render a 4-panel publication-quality verification figure:
      [Top-Left]:     RGB Image + 2D GT (dashed green) vs Pred (solid vibrant)
      [Top-Right]:    Depth Map (inferno colormap) + Projected 3D alignment
      [Bottom-Left]:  3D Physical Camera-Space View (meters)
      [Bottom-Right]: Node-by-Node Euclidean Error profile (mm) + Metrics Box
    """
    fig = plt.figure(figsize=(20, 14), dpi=150)
    plt.suptitle(f"CurvePT 3D Deformable Linear Object Estimation {title_suffix}",
                 fontsize=18, fontweight="bold", y=0.98)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # =========================================================================
    # Panel 1: RGB Image with 2D Projections
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(rgb)
    ax1.set_title(f"RGB Camera View ({len(pred_curves)} Pred vs {len(gt_curves)} GT DLOs)",
                  fontsize=13, fontweight="semibold")

    # Plot GT curves
    for g_idx, gt in enumerate(gt_curves):
        pts_2d = project_3d_to_2d(gt["coords"], K)
        vis = gt["vis"]
        for j in range(len(pts_2d) - 1):
            if vis[j] > 0.5 and vis[j + 1] > 0.5:
                ax1.plot(pts_2d[j:j+2, 0], pts_2d[j:j+2, 1],
                         color="lime", linestyle="--", linewidth=3, alpha=0.9,
                         label="Ground Truth" if (g_idx == 0 and j == 0) else "")
        # GT Endpoints
        if vis[0] > 0.5:
            ax1.scatter(pts_2d[0, 0], pts_2d[0, 1], color="lime", s=40, zorder=5)
        if vis[-1] > 0.5:
            ax1.scatter(pts_2d[-1, 0], pts_2d[-1, 1], color="lime", s=40, zorder=5)

    # Plot Predicted curves
    for p_idx, pred in enumerate(pred_curves):
        pts_2d = project_3d_to_2d(pred["coords"], K)
        vis = pred["vis"]
        conf = pred["conf"]
        c = colors[p_idx % len(colors)]
        for j in range(len(pts_2d) - 1):
            alpha = float(min(vis[j], vis[j + 1]) * 0.7 + 0.3)
            ax1.plot(pts_2d[j:j+2, 0], pts_2d[j:j+2, 1],
                     color=c, linewidth=2.5, alpha=alpha,
                     label=f"Pred DLO {p_idx} (conf={conf:.2f})" if j == 0 else "")
        # Pred Endpoints
        ax1.scatter(pts_2d[0, 0], pts_2d[0, 1], color=c, s=50, edgecolors="white", zorder=6)
        ax1.scatter(pts_2d[-1, 0], pts_2d[-1, 1], color=c, s=50, edgecolors="white", zorder=6)

        # Midpoint label
        mid = len(pts_2d) // 2
        ax1.annotate(f"Pred #{p_idx} ({conf:.2f})", pts_2d[mid],
                     color="white", fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc=c[:3], ec="white", lw=1, alpha=0.85))

    ax1.set_xlim([0, rgb.shape[1]])
    ax1.set_ylim([rgb.shape[0], 0])
    ax1.axis("off")
    ax1.legend(loc="upper right", framealpha=0.85, fontsize=10)

    # =========================================================================
    # Panel 2: Depth Map with Spatial Overlay
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)
    depth_vis = np.clip(depth, 0.2, 3.0)
    im2 = ax2.imshow(depth_vis, cmap="inferno")
    ax2.set_title("Aligned Metric Depth Map (m)", fontsize=13, fontweight="semibold")
    cbar = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04)
    cbar.set_label("Depth Z (meters)", fontsize=10)

    # Overlay curves on depth
    for p_idx, pred in enumerate(pred_curves):
        pts_2d = project_3d_to_2d(pred["coords"], K)
        c = colors[p_idx % len(colors)]
        ax2.plot(pts_2d[:, 0], pts_2d[:, 1], color="cyan", linewidth=2.0, alpha=0.85)

    ax2.set_xlim([0, rgb.shape[1]])
    ax2.set_ylim([rgb.shape[0], 0])
    ax2.axis("off")

    # =========================================================================
    # Panel 3: Physical 3D Camera Space View (Isometric)
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.set_title("True 3D Spatial Routing (Camera Frame)", fontsize=13, fontweight="semibold")

    # Draw Ground Truth 3D curves
    for g_idx, gt in enumerate(gt_curves):
        pts = gt["coords"]
        vis = gt["vis"] > 0.5
        if vis.sum() > 1:
            ax3.plot(pts[vis, 0], pts[vis, 2], -pts[vis, 1],
                     color="lime", linestyle="--", linewidth=3.0, alpha=0.85,
                     label="GT 3D" if g_idx == 0 else "")
            ax3.scatter(pts[vis, 0], pts[vis, 2], -pts[vis, 1], color="lime", s=20)

    # Draw Predicted 3D curves
    for p_idx, pred in enumerate(pred_curves):
        pts = pred["coords"]
        c = colors[p_idx % len(colors)]
        ax3.plot(pts[:, 0], pts[:, 2], -pts[:, 1],
                 color=c, linewidth=2.5,
                 label=f"Pred #{p_idx}" if len(gt_curves) == 0 or p_idx < 3 else "")
        ax3.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], color=c, s=25, edgecolors="black", linewidth=0.5)

    ax3.set_xlabel("X (Right, m)", labelpad=8)
    ax3.set_ylabel("Z (Depth, m)", labelpad=8)
    ax3.set_zlabel("-Y (Up, m)", labelpad=8)
    ax3.view_init(elev=25, azim=-60)
    ax3.legend(loc="upper right", framealpha=0.8, fontsize=9)

    # =========================================================================
    # Panel 4: Quantitative Node Error Profile & Metrics
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Node-Level Physical Error Profile (mm)", fontsize=13, fontweight="semibold")

    mne_list = []
    pck10_list, pck20_list, pck50_list = [], [], []

    if len(matches) > 0:
        for m_idx, (p_idx, g_idx) in enumerate(matches):
            p_pts = pred_curves[p_idx]["coords"] * 1000.0  # mm
            g_pts = gt_curves[g_idx]["coords"] * 1000.0    # mm
            v_gt = gt_curves[g_idx]["vis"]

            # Direction-invariant alignment
            err_fwd = np.linalg.norm(p_pts - g_pts, axis=-1)
            err_rev = np.linalg.norm(p_pts - g_pts[::-1], axis=-1)
            err_nodes = err_fwd if err_fwd.mean() < err_rev.mean() else err_rev

            nodes = np.arange(len(err_nodes))
            c = colors[p_idx % len(colors)]
            ax4.plot(nodes, err_nodes, color=c, linewidth=2.5, marker="o", markersize=4,
                     label=f"Pred #{p_idx} → GT #{g_idx} (Mean: {err_nodes.mean():.1f}mm)")

            # Accumulate stats
            mne_list.append(err_nodes.mean())
            pck10_list.append((err_nodes < 10.0).mean() * 100.0)
            pck20_list.append((err_nodes < 20.0).mean() * 100.0)
            pck50_list.append((err_nodes < 50.0).mean() * 100.0)

        # Threshold lines
        ax4.axhline(10.0, color="green", linestyle=":", alpha=0.7, label="10mm (1.0cm)")
        ax4.axhline(20.0, color="orange", linestyle=":", alpha=0.7, label="20mm (2.0cm)")
        ax4.axhline(50.0, color="red", linestyle=":", alpha=0.7, label="50mm (5.0cm)")

        ax4.set_xlabel("Node Index along Cable (Head → Tail)", fontsize=11)
        ax4.set_ylabel("3D Euclidean Error (mm)", fontsize=11)
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # Metrics Summary Text Box
        summary_text = (
            f"EVALUATION METRICS:\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Mean MNE:    {np.mean(mne_list):.1f} mm\n"
            f"• PCK @ 10mm:  {np.mean(pck10_list):.1f}%\n"
            f"• PCK @ 20mm:  {np.mean(pck20_list):.1f}%\n"
            f"• PCK @ 50mm:  {np.mean(pck50_list):.1f}%\n"
            f"• Radius Err:  {abs(pred_curves[0]['radius'] - gt_curves[0]['radius'])*1000:.2f} mm"
        )
        ax4.text(0.03, 0.95, summary_text, transform=ax4.transAxes,
                 fontsize=11, fontfamily="monospace", verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.6", fc="#f8f9fa", ec="#343a40", lw=1.5, alpha=0.95))
    else:
        ax4.text(0.5, 0.5, "No ground truth available for error profiling\n(Unsupervised Real-World Mode)",
                 ha="center", va="center", fontsize=12, color="gray")
        ax4.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED FIGURE] {output_path}")


def export_interactive_3d_html(
    gt_curves: list[dict],
    pred_curves: list[dict],
    output_path: str,
    title: str = "CurvePT 3D Viewer",
):
    """Export an interactive 3D WebGL viewer using Three.js as a standalone HTML file."""
    gt_data = [g["coords"].tolist() for g in gt_curves]
    pred_data = [{"coords": p["coords"].tolist(), "conf": p["conf"]} for p in pred_curves]

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; background: #0f172a; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    #ui {{ position: absolute; top: 16px; left: 16px; background: rgba(15,23,42,0.88); color: #f8fafc; padding: 14px 18px; border-radius: 10px; border: 1px solid #334155; box-shadow: 0 4px 20px rgba(0,0,0,0.5); pointer-events: none; }}
    h1 {{ margin: 0 0 6px 0; font-size: 16px; font-weight: 600; color: #38bdf8; }}
    p {{ margin: 2px 0; font-size: 13px; color: #94a3b8; }}
    .legend {{ display: flex; align-items: center; gap: 8px; margin-top: 6px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="ui">
    <h1>{title}</h1>
    <p>Left Click + Drag: Rotate | Right Click + Drag: Pan | Scroll: Zoom</p>
    <div class="legend"><span class="dot" style="background:#22c55e;"></span><span style="color:#22c55e;">Ground Truth 3D Curves ({len(gt_data)})</span></div>
    <div class="legend"><span class="dot" style="background:#06b6d4;"></span><span style="color:#06b6d4;">Predicted 3D Curves ({len(pred_data)})</span></div>
  </div>
  <script>
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(0, 0.8, 2.5);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const grid = new THREE.GridHelper(3, 15, 0x334155, 0x1e293b);
    grid.position.y = -0.5;
    scene.add(grid);
    const axes = new THREE.AxesHelper(0.4);
    scene.add(axes);

    const gtCurves = {json.dumps(gt_data)};
    const predCurves = {json.dumps(pred_data)};

    // Render GT Curves (Green)
    gtCurves.forEach(pts => {{
      const points = pts.map(p => new THREE.Vector3(p[0], -p[1], -p[2]));
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({{ color: 0x22c55e, linewidth: 3 }});
      scene.add(new THREE.Line(geom, mat));
      points.forEach(pt => {{
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.008, 8, 8), new THREE.MeshBasicMaterial({{ color: 0x4ade80 }}));
        dot.position.copy(pt);
        scene.add(dot);
      }});
    }});

    // Render Pred Curves (Cyan / Orange)
    const colors = [0x06b6d4, 0xf97316, 0xa855f7, 0xeab308];
    predCurves.forEach((item, idx) => {{
      const points = item.coords.map(p => new THREE.Vector3(p[0], -p[1], -p[2]));
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({{ color: colors[idx % colors.length], linewidth: 4 }});
      scene.add(new THREE.Line(geom, mat));
      points.forEach(pt => {{
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.01, 8, 8), new THREE.MeshBasicMaterial({{ color: colors[idx % colors.length] }}));
        dot.position.copy(pt);
        scene.add(dot);
      }});
    }});

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  [SAVED 3D HTML] {output_path}")


def load_single_frame(frame_dir: str, num_nodes: int = 64, max_depth: float = 3.0):
    """Load RGB, Depth, Normals, and GT from a dataset frame directory."""
    frame_dir = Path(frame_dir)
    rgb_path = frame_dir / "rgb_left.png"
    depth_path = frame_dir / "depth_clean.npz"
    if not depth_path.exists():
        depth_path = frame_dir / "depth_noisy.npz"
    normals_path = frame_dir / "normals.npz"
    gt_path = frame_dir / "curve_gt.json"

    # 1. RGB
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0

    # 2. Depth
    with np.load(depth_path) as df:
        depth = np.array(df["depth"], dtype=np.float32)

    # 3. Normals
    if normals_path.exists():
        with np.load(normals_path) as nf:
            normals = np.array(nf["normals"], dtype=np.float32)
    else:
        normals = compute_surface_normals(depth)

    # 4. Ground Truth
    gt_curves = []
    K = np.array([[605.2, 0.0, 640.0], [0.0, 605.1, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    if gt_path.exists():
        with open(gt_path, "r") as f:
            gt_data = json.load(f)

        if "camera_intrinsics" in gt_data:
            K = np.array(gt_data["camera_intrinsics"], dtype=np.float32)

        w2c = np.array(gt_data.get("world_to_cam", np.eye(4)), dtype=np.float32)
        for c in gt_data.get("curves", []):
            if "points_3d" in c:
                pts = np.array(c["points_3d"], dtype=np.float32)
            elif "points_cam" in c:
                pts = np.array(c["points_cam"], dtype=np.float32)
            else:
                pw = np.array(c["points_world"], dtype=np.float32)
                pwh = np.concatenate([pw, np.ones((len(pw), 1), dtype=np.float32)], axis=1)
                pts = (w2c @ pwh.T).T[:, :3]

            # Resample curve to model num_nodes (e.g. 64)
            pts = DLODataset._resample_curve(pts, num_nodes)

            zc = pts[:, 2]
            xc = pts[:, 0]
            yc = pts[:, 1]
            valid_z = (zc > 0.05) & (zc <= max_depth)
            safe_z = np.where(valid_z, zc, 1.0)
            u = K[0, 0] * (xc / safe_z) + K[0, 2]
            v = K[1, 1] * (yc / safe_z) + K[1, 2]
            in_f = valid_z & (u >= 0) & (u < rgb.shape[1]) & (v >= 0) & (v < rgb.shape[0])

            if in_f.sum() >= 2:
                r = float(c.get("radius", c.get("estimated_radius", 0.005)))
                gt_curves.append({
                    "coords": pts,
                    "vis": in_f.astype(np.float32),
                    "radius": r,
                })

    return rgb, depth, normals, gt_curves, K


def run_evaluation(
    model,
    device,
    samples: list[dict],
    output_dir: str,
    conf_threshold: float = 0.25,
    num_nodes: int = 64,
    max_depth: float = 3.0,
    export_html: bool = False,
):
    """Run model inference across samples and generate visual verification cards."""
    os.makedirs(output_dir, exist_ok=True)
    all_mne = []

    print(f"\n{'='*75}")
    print(f"  CurvePT Verification & Visual Inference: {len(samples)} samples")
    print(f"{'='*75}")

    for idx, sample in enumerate(samples):
        rgb, depth, normals, gt_curves, K = load_single_frame(
            sample["frame_dir"], num_nodes=num_nodes, max_depth=max_depth
        )

        # Prepare tensors (B, C, H, W)
        norm_depth = np.clip(depth, 0, max_depth) / max_depth
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        depth_t = torch.from_numpy(norm_depth).unsqueeze(0).unsqueeze(0).to(device)
        normals_t = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(rgb_t, depth_t, normals_t)
            p_coords = preds["pred_coords"][-1][0]  # (Q, N, 3) in meters
            p_conf = torch.sigmoid(preds["pred_conf"][-1][0, :, 0])  # (Q,)
            p_vis = torch.sigmoid(preds["pred_vis"][-1][0])          # (Q, N)
            p_radius = preds["pred_radius"][-1][0, :, 0]             # (Q,)

        # 1. Match predictions with GT across all Q queries using Hungarian matching
        raw_matches = []
        matched_pred_indices = set()
        if len(gt_curves) > 0:
            gt_coords_t = torch.from_numpy(np.stack([g["coords"] for g in gt_curves])).unsqueeze(0).to(device)
            gt_vis_t = torch.from_numpy(np.stack([g["vis"] for g in gt_curves])).unsqueeze(0).to(device)
            p_coords_all = p_coords.unsqueeze(0)
            p_conf_all = preds["pred_conf"][-1][:, :, 0:1]
            p_vis_all = preds["pred_vis"][-1]
            num_gt = torch.tensor([len(gt_curves)], device=device)

            m = hungarian_match(p_coords_all, p_conf_all, p_vis_all,
                                gt_coords_t, gt_vis_t, num_curves=num_gt)[0]
            p_m, g_m = m
            for pi, gi in zip(p_m.cpu().numpy(), g_m.cpu().numpy()):
                if gi < len(gt_curves):
                    raw_matches.append((int(pi), int(gi)))
                    matched_pred_indices.add(int(pi))

        # 2. Active predictions: all matched queries + any extra queries passing conf_threshold
        pred_curves = []
        curve_idx_map = {}
        for q in range(p_coords.shape[0]):
            conf_val = float(p_conf[q].item())
            if q in matched_pred_indices or conf_val >= conf_threshold:
                curve_idx_map[q] = len(pred_curves)
                pred_curves.append({
                    "orig_q": q,
                    "coords": p_coords[q].cpu().numpy(),
                    "conf": conf_val,
                    "vis": p_vis[q].cpu().numpy(),
                    "radius": float(p_radius[q].item()),
                })

        # Remap match indices to pred_curves indices
        matches = []
        for pi, gi in raw_matches:
            if pi in curve_idx_map:
                matches.append((curve_idx_map[pi], gi))

        # Render Figure
        sample_name = sample.get("name", f"sample_{idx:03d}")
        fig_path = os.path.join(output_dir, f"{sample_name}.png")
        generate_sample_figure(
            rgb=rgb, depth=depth, gt_curves=gt_curves,
            pred_curves=pred_curves, matches=matches, K=K,
            output_path=fig_path, title_suffix=f"[{sample_name}]"
        )

        if export_html:
            html_path = os.path.join(output_dir, f"{sample_name}_3d.html")
            export_interactive_3d_html(
                gt_curves=gt_curves, pred_curves=pred_curves,
                output_path=html_path, title=f"CurvePT 3D: {sample_name}"
            )

        # Print per-sample metrics
        if len(matches) > 0:
            sample_mne = []
            for p_i, g_i in matches:
                p_pts = pred_curves[p_i]["coords"] * 1000.0
                g_pts = gt_curves[g_i]["coords"] * 1000.0
                err_f = np.linalg.norm(p_pts - g_pts, axis=-1)
                err_r = np.linalg.norm(p_pts - g_pts[::-1], axis=-1)
                err = np.minimum(err_f.mean(), err_r.mean())
                sample_mne.append(err)
                all_mne.append(err)
            print(f"  [{idx+1:02d}/{len(samples):02d}] {sample_name:<25} | "
                  f"GT: {len(gt_curves)} | Pred: {len(pred_curves)} | "
                  f"MNE: {np.mean(sample_mne):.1f} mm")
        else:
            print(f"  [{idx+1:02d}/{len(samples):02d}] {sample_name:<25} | "
                  f"Pred DLOs: {len(pred_curves)} (no GT matches)")

    if len(all_mne) > 0:
        print(f"\n{'='*75}")
        print(f"  OVERALL EVALUATION SUMMARY across {len(all_mne)} evaluated curves:")
        print(f"  • Mean Node Error (MNE):   {np.mean(all_mne):.1f} mm")
        print(f"  • Median MNE:              {np.median(all_mne):.1f} mm")
        print(f"  • Best Single Curve MNE:   {np.min(all_mne):.1f} mm")
        print(f"  • Visual Cards Saved To:   {output_dir}")
        print(f"{'='*75}\n")


def main():
    parser = argparse.ArgumentParser(description="CurvePT Inference & Visual Verification Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--frame_dir", type=str, default=None,
                        help="Evaluate a specific frame directory (e.g. scene_0001/frame_0000)")
    parser.add_argument("--split", type=str, default="configs/splits/val.json",
                        help="Path to dataset split file")
    parser.add_argument("--data_root", type=str,
                        default="F:/Jarvis/1k_Curve_Dataset/Curve_Data_Compact_RealSenseD435_v3",
                        help="Dataset root directory")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples to visualize from split")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Confidence threshold for DLO queries")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Output directory for visual verification figures")
    parser.add_argument("--export_html", action="store_true",
                        help="Export interactive 3D WebGL HTML viewer for each sample")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Config and Model
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading checkpoint from: {args.checkpoint}")
    model = build_curvept(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("  [LOADED] Model ready for inference.")

    # 2. Collect Samples to evaluate
    samples = []
    if args.frame_dir:
        samples.append({
            "name": Path(args.frame_dir).name,
            "frame_dir": args.frame_dir,
        })
    else:
        # Load from validation split
        if os.path.exists(args.split):
            with open(args.split, "r") as f:
                split_scenes = json.load(f).get("scenes", [])
            import glob
            for s_name in split_scenes[:args.num_samples]:
                s_dir = os.path.join(args.data_root, s_name)
                f_dirs = sorted(glob.glob(os.path.join(s_dir, "frame_*")))
                if f_dirs:
                    chosen_f = f_dirs[len(f_dirs) // 2]
                    samples.append({
                        "name": f"{s_name}_{Path(chosen_f).name}",
                        "frame_dir": chosen_f,
                    })
        else:
            print(f"Split file {args.split} not found!")

    if not samples:
        print("No valid frame directories found to evaluate.")
        return

    # 3. Run Evaluation & Rendering
    run_evaluation(
        model=model,
        device=device,
        samples=samples,
        output_dir=args.output_dir,
        conf_threshold=args.threshold,
        num_nodes=config["model"]["num_nodes"],
        max_depth=config["data"]["max_depth"],
        export_html=args.export_html,
    )


if __name__ == "__main__":
    main()
