"""
render_dlo_dataset.py — Procedural Synthetic Dataset Generator for DLO 3D Segmentation

Generates diverse training data for state-of-the-art DLO (Deformable Linear Object)
3D segmentation and tracking models (TrackDLO, DLO-Splatting, SAM2 fine-tuning, GNNs).

ALL DLOs are generated procedurally — no pre-made .blend file required.
Each scene creates a unique set of cables with random shapes, lengths, radii,
materials, and colors, placed in a random room with varied lighting.

Shape strategies:
  - random_bezier: Smooth curves from random control points
  - catenary: Hanging cables between anchor points
  - wave: Sinusoidal cables along surfaces
  - helix: Coiled/spiral cables
  - floor_route: Cables routed along floors with turns
  - draped: Cables draped with gravity-like sag
  - loop: Self-crossing cables (figure-8, loops)

Features:
  - Stereo camera rendering (ZED2, RealSense D435, OAK-D presets)
  - Ray-cast occlusion visibility for skeleton nodes
  - Domain randomization (materials, lighting, colors, distractor objects)
  - Depth noise simulation (missing data on thin objects, quantization, flying pixels)
  - Normal map rendering
  - GT centerline directly from generated paths (exact, no re-extraction)

Output:
  output/
  ├── scene_XXXX/
  │   ├── metadata.json
  │   └── frame_XXXX/
  │       ├── rgb_left.png / rgb_right.png
  │       ├── depth_clean.npz / depth_noisy.npz
  │       ├── instance_segmap.png / semantic_segmap.png
  │       ├── normals.npz
  │       ├── curve_gt.json
  │       └── camera.json
  └── dataset_manifest.json

Usage:
  blenderproc run render_dlo_dataset.py [output_dir] [cctextures_path] [options]

  # 50k images for GNN training (ZED2 stereo):
  blenderproc run render_dlo_dataset.py dlo_dataset ./cctextures \\
      --num_scenes 5000 --poses_per_scene 10 --stereo_camera zed2

  # Quick test (50 images, monocular):
  blenderproc run render_dlo_dataset.py test_output ./cctextures \\
      --num_scenes 5 --poses_per_scene 10 --stereo_camera none
"""

import blenderproc as bproc
import argparse
import numpy as np
import os
import bpy
import random
import json
import re
import math
from mathutils import Vector, Matrix


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & CLI
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="Procedural DLO Dataset Generator for 3D Segmentation Training"
)
parser.add_argument('output_dir', nargs='?', default="dlo_dataset",
                    help="Path to where the dataset will be saved.")
parser.add_argument('cc_material_path', nargs='?',
                    default="./BlenderProc_Project/Segmentation/cctextures",
                    help="Path to CCTextures folder for room materials.")

# Scene generation
parser.add_argument('--num_scenes', type=int, default=5000,
                    help="Number of unique scenes (5000×10 poses = 50k images).")
parser.add_argument('--poses_per_scene', type=int, default=10,
                    help="Number of camera poses per scene.")

# Procedural DLO settings
parser.add_argument('--min_dlos', type=int, default=3,
                    help="Minimum DLOs per scene.")
parser.add_argument('--max_dlos', type=int, default=15,
                    help="Maximum DLOs per scene.")
parser.add_argument('--min_dlo_length', type=float, default=0.3,
                    help="Minimum DLO arc length in meters.")
parser.add_argument('--max_dlo_length', type=float, default=3.0,
                    help="Maximum DLO arc length in meters.")
parser.add_argument('--min_dlo_radius', type=float, default=0.001,
                    help="Minimum DLO cross-section radius in meters (1mm).")
parser.add_argument('--max_dlo_radius', type=float, default=0.020,
                    help="Maximum DLO cross-section radius in meters (20mm).")
parser.add_argument('--min_complexity', type=int, default=3,
                    help="Min control points for curve shape.")
parser.add_argument('--max_complexity', type=int, default=20,
                    help="Max control points for curve shape.")
parser.add_argument('--curve_gt_points', type=int, default=64,
                    help="Number of equidistant GT skeleton nodes per curve.")

# Stereo camera settings
parser.add_argument('--stereo_camera', default="zed2",
                    choices=["zed2", "realsense_d435", "realsense_d435i", "oak_d", "custom", "none"],
                    help="Stereo camera preset. 'none' for monocular only.")
parser.add_argument('--stereo_baseline', type=float, default=None,
                    help="Custom stereo baseline in meters (overrides preset).")
parser.add_argument('--image_width', type=int, default=1280,
                    help="Render width in pixels.")
parser.add_argument('--image_height', type=int, default=720,
                    help="Render height in pixels.")

# Domain randomization
parser.add_argument('--num_distractor_objects', type=int, default=5,
                    help="Max distractor objects per scene (0 = disable).")
parser.add_argument('--randomize_lighting', action='store_true', default=True,
                    help="Randomize lighting beyond ceiling emission.")

# Depth noise
parser.add_argument('--add_depth_noise', action='store_true', default=False,
                    help="Generate noisy depth alongside clean depth (prefer training-time augmentation).")
parser.add_argument('--depth_noise_sigma', type=float, default=0.002,
                    help="Std-dev of Gaussian depth noise in meters.")

# Rendering quality
parser.add_argument('--render_samples', type=int, default=32,
                    help="Number of render samples (higher = less noise, slower).")
parser.add_argument('--render_noise_threshold', type=float, default=0.005,
                    help="Adaptive sampling noise threshold.")
parser.add_argument('--write_coco', action='store_true', default=False,
                    help="Write COCO annotations (optional, for 2D detection baselines).")

args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STEREO CAMERA PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

STEREO_PRESETS = {
    "zed2": {
        "name": "ZED 2",
        "baseline_m": 0.120,
        "fx": 527.0, "fy": 527.0, "cx": 640.0, "cy": 360.0,
        "width": 1280, "height": 720,
    },
    "realsense_d435": {
        "name": "Intel RealSense D435",
        "baseline_m": 0.050,
        "fx": 605.2, "fy": 605.1, "cx": 425.7, "cy": 246.0,
        "width": 1280, "height": 720,
    },
    "realsense_d435i": {
        "name": "Intel RealSense D435i",
        "baseline_m": 0.050,
        "fx": 605.2, "fy": 605.1, "cx": 425.7, "cy": 246.0,
        "width": 1280, "height": 720,
    },
    "oak_d": {
        "name": "Luxonis OAK-D",
        "baseline_m": 0.075,
        "fx": 570.0, "fy": 570.0, "cx": 640.0, "cy": 360.0,
        "width": 1280, "height": 720,
    },
}


def get_camera_config(args):
    """Build camera configuration from CLI args and presets."""
    if args.stereo_camera == "none":
        return {
            "stereo": False, "baseline_m": 0.0,
            "K": np.array([[605.2, 0.0, args.image_width / 2.0],
                           [0.0, 605.1, args.image_height / 2.0],
                           [0.0, 0.0, 1.0]], dtype=np.float32),
            "width": args.image_width, "height": args.image_height,
            "preset_name": "monocular",
        }
    if args.stereo_camera == "custom":
        baseline = args.stereo_baseline if args.stereo_baseline else 0.120
        return {
            "stereo": True, "baseline_m": baseline,
            "K": np.array([[605.2, 0.0, args.image_width / 2.0],
                           [0.0, 605.1, args.image_height / 2.0],
                           [0.0, 0.0, 1.0]], dtype=np.float32),
            "width": args.image_width, "height": args.image_height,
            "preset_name": "custom",
        }

    preset = STEREO_PRESETS[args.stereo_camera]
    baseline = args.stereo_baseline if args.stereo_baseline else preset["baseline_m"]
    return {
        "stereo": True, "baseline_m": baseline,
        "K": np.array([[preset["fx"], 0.0, preset["cx"]],
                       [0.0, preset["fy"], preset["cy"]],
                       [0.0, 0.0, 1.0]], dtype=np.float32),
        "width": preset["width"], "height": preset["height"],
        "preset_name": preset["name"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MATH & INTERPOLATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_key(name):
    """Sanitize a string to be a valid dict/filename key."""
    sanitized = re.sub(r'[^0-9a-zA-Z_]+', '_', name).strip('_')
    return sanitized or "curve"


def resample_polyline(points, n_points):
    """Resample a polyline to n_points equidistant samples."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float32)
    if len(pts) == 0:
        return pts
    if n_points <= 0 or len(pts) == n_points:
        return pts
    if len(pts) == 1:
        return np.repeat(pts, max(1, n_points), axis=0)
    if n_points < 2:
        return pts[:1]

    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = cum_len[-1]
    if total_len < 1e-8:
        return np.repeat(pts[:1], n_points, axis=0)

    target = np.linspace(0.0, total_len, n_points)
    resampled = np.zeros((n_points, 3), dtype=np.float32)
    for dim in range(3):
        resampled[:, dim] = np.interp(target, cum_len, pts[:, dim])
    return resampled


def catmull_rom_chain(control_points, samples_per_segment=50):
    """
    Catmull-Rom spline interpolation through ordered control points.
    Returns a dense, smooth path that passes through every control point.
    """
    pts = np.asarray(control_points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return pts.astype(np.float32)
    if n == 2:
        return np.linspace(pts[0], pts[1], samples_per_segment).astype(np.float32)

    # Extend with phantom points (reflection) so spline reaches first/last point
    pts_ext = np.vstack([
        2.0 * pts[0] - pts[1],
        pts,
        2.0 * pts[-1] - pts[-2],
    ])

    all_points = []
    num_segments = n - 1

    for seg in range(num_segments):
        p0 = pts_ext[seg]
        p1 = pts_ext[seg + 1]
        p2 = pts_ext[seg + 2]
        p3 = pts_ext[seg + 3]

        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t

            point = 0.5 * (
                (2.0 * p1) +
                (-p0 + p2) * t +
                (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            all_points.append(point)

    # Add the final point
    all_points.append(pts[-1].copy())
    return np.array(all_points, dtype=np.float32)


def arc_length(points):
    """Compute total arc length of a polyline."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def laplacian_smooth(points, iterations=5, alpha=0.5):
    """Laplacian smoothing preserving endpoints."""
    pts = points.copy()
    for _ in range(iterations):
        new = pts.copy()
        for i in range(1, len(pts) - 1):
            new[i] = pts[i] + alpha * ((pts[i - 1] + pts[i + 1]) / 2.0 - pts[i])
        pts = new
    return pts


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PROCEDURAL DLO PATH GENERATION — 7 SHAPE STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

# --- Strategy 1: Random Bézier ---
def gen_random_bezier(center, spread, num_ctrl, floor_z):
    """
    Generate a smooth curve from random control points.
    Most versatile — produces everything from gentle arcs to complex tangles
    depending on num_ctrl and spread.
    """
    controls = []
    # Generate control points with a directional bias (not purely random cloud)
    # Pick a random primary direction
    main_dir = np.random.randn(3)
    main_dir[2] *= 0.3  # flatten vertical bias
    main_dir = main_dir / (np.linalg.norm(main_dir) + 1e-8)

    for i in range(num_ctrl):
        t = i / max(1, num_ctrl - 1)
        # Walk along main direction with random perpendicular offsets
        base = center + main_dir * spread * (t - 0.5) * 2.0
        offset = np.random.uniform(-spread * 0.6, spread * 0.6, 3)
        pt = base + offset
        pt[2] = max(pt[2], floor_z + 0.005)  # keep above floor
        controls.append(pt)

    dense = catmull_rom_chain(np.array(controls), samples_per_segment=60)
    return dense


# --- Strategy 2: Catenary (hanging cable) ---
def gen_catenary(anchor_a, anchor_b, sag_factor=0.3):
    """
    Hanging cable between two anchor points.
    Uses catenary equation: z(t) = -a * cosh((t - 0.5) / a) in the vertical plane.
    """
    num_samples = 200
    t = np.linspace(0, 1, num_samples)

    # Linear interpolation between anchors
    path = np.outer(1 - t, anchor_a) + np.outer(t, anchor_b)

    # Compute sag in the vertical direction
    span = np.linalg.norm(anchor_b[:2] - anchor_a[:2])  # horizontal span
    a = max(0.1, sag_factor * span)  # catenary parameter

    # Catenary sag (max at midpoint, zero at endpoints)
    sag = a * (np.cosh((t - 0.5) * span / a) - np.cosh(0.5 * span / a))
    # Normalize so endpoints have zero sag
    sag = sag - sag[0]

    path[:, 2] += sag  # apply vertical sag

    return path.astype(np.float32)


# --- Strategy 3: Wave / S-curve ---
def gen_wave(start, end, amplitude, frequency, floor_z):
    """
    Sinusoidal cable between two points — waves in a plane perpendicular
    to the start-end direction.
    """
    num_samples = 200
    t = np.linspace(0, 1, num_samples)

    # Main direction
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.repeat(start[np.newaxis, :], num_samples, axis=0).astype(np.float32)

    fwd = direction / length
    # Perpendicular direction (for wave displacement)
    up = np.array([0, 0, 1.0])
    side = np.cross(fwd, up)
    side_len = np.linalg.norm(side)
    if side_len < 1e-6:
        side = np.array([1, 0, 0.0])
    else:
        side = side / side_len

    # Linear base path
    path = np.outer(1 - t, start) + np.outer(t, end)

    # Add sinusoidal displacement
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    path += np.outer(wave, side)

    # Keep above floor
    path[:, 2] = np.maximum(path[:, 2], floor_z + 0.005)

    return path.astype(np.float32)


# --- Strategy 4: Helix / Coil ---
def gen_helix(center, radius, height, num_turns, floor_z):
    """
    Helical/spiral cable — like a coiled cord or spring.
    """
    num_samples = int(num_turns * 80)  # 80 samples per turn for smoothness
    num_samples = max(50, min(num_samples, 500))
    t = np.linspace(0, num_turns * 2 * np.pi, num_samples)

    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    z = center[2] + np.linspace(0, height, num_samples)
    z = np.maximum(z, floor_z + 0.005)

    # Random rotation of the helix axis
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_rot = center[0] + (x - center[0]) * cos_a - (y - center[1]) * sin_a
    y_rot = center[1] + (x - center[0]) * sin_a + (y - center[1]) * cos_a

    return np.column_stack([x_rot, y_rot, z]).astype(np.float32)


# --- Strategy 5: Floor route (cable on surface with turns) ---
def gen_floor_route(floor_z, bounds_min, bounds_max, num_waypoints):
    """
    Cable lying on a surface with random waypoints and smooth turns.
    Simulates cables routed along floors, tables, or shelves.
    """
    height = floor_z + np.random.uniform(0.003, 0.015)  # just above surface
    margin = 0.3

    waypoints = []
    for _ in range(num_waypoints):
        x = np.random.uniform(bounds_min[0] + margin, bounds_max[0] - margin)
        y = np.random.uniform(bounds_min[1] + margin, bounds_max[1] - margin)
        # Slight Z variation (cable isn't perfectly flat)
        z = height + np.random.uniform(-0.005, 0.01)
        waypoints.append([x, y, max(z, floor_z + 0.002)])

    waypoints = np.array(waypoints)
    dense = catmull_rom_chain(waypoints, samples_per_segment=60)
    return dense


# --- Strategy 6: Draped (gravity sag between supports) ---
def gen_draped(start, end, mid_height, floor_z):
    """
    Cable draped between two elevated points, sagging toward the floor
    in the middle — like a cable hanging between two supports.
    """
    num_samples = 200
    t = np.linspace(0, 1, num_samples)

    # Base linear path (elevated)
    path = np.outer(1 - t, start) + np.outer(t, end)

    # Parabolic sag: max at midpoint
    sag = 4 * mid_height * t * (1 - t)
    path[:, 2] -= sag
    path[:, 2] = np.maximum(path[:, 2], floor_z + 0.005)

    return path.astype(np.float32)


# --- Strategy 7: Loop / Self-crossing ---
def gen_loop(center, size, floor_z, loop_type="figure8"):
    """
    Self-crossing cable — figure-8, trefoil, or simple loop.
    Critical for training occlusion handling.
    """
    num_samples = 300

    if loop_type == "figure8":
        t = np.linspace(0, 2 * np.pi, num_samples)
        x = center[0] + size * np.sin(t)
        y = center[1] + size * np.sin(t) * np.cos(t)
        z_base = center[2] + size * 0.3 * np.sin(2 * t)

    elif loop_type == "trefoil":
        t = np.linspace(0, 2 * np.pi, num_samples)
        x = center[0] + size * (np.sin(t) + 2 * np.sin(2 * t))
        y = center[1] + size * (np.cos(t) - 2 * np.cos(2 * t))
        z_base = center[2] + size * 0.5 * (-np.sin(3 * t))

    else:  # simple loop
        t = np.linspace(0, 2 * np.pi * 1.3, num_samples)  # slightly > full circle
        x = center[0] + size * np.cos(t)
        y = center[1] + size * np.sin(t)
        z_base = center[2] + size * 0.1 * np.sin(3 * t)

    z = np.maximum(z_base, floor_z + 0.005)
    path = np.column_stack([x, y, z]).astype(np.float32)

    # Randomly rotate around Z
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_r = center[0] + (path[:, 0] - center[0]) * cos_a - (path[:, 1] - center[1]) * sin_a
    y_r = center[1] + (path[:, 0] - center[0]) * sin_a + (path[:, 1] - center[1]) * cos_a
    path[:, 0] = x_r
    path[:, 1] = y_r

    return path


# --- Master generator ---
SHAPE_STRATEGIES = [
    "random_bezier", "catenary", "wave", "helix",
    "floor_route", "draped", "loop",
]

# Probability weights — more emphasis on common cable shapes
STRATEGY_WEIGHTS = [
    0.25,  # random_bezier — most diverse
    0.15,  # catenary — very common (hanging cables)
    0.15,  # wave — common (routed cables)
    0.08,  # helix — less common (coiled cords)
    0.15,  # floor_route — very common (floor cables)
    0.12,  # draped — common (cable between fixtures)
    0.10,  # loop — important for occlusion training
]


def generate_dlo_path(bounds_min, bounds_max, floor_z, target_length,
                      complexity, strategy=None):
    """
    Master function: generate a DLO centerline path within the given bounds.

    Args:
        bounds_min/max: 3D room bounds (np arrays)
        floor_z: floor Z coordinate
        target_length: approximate desired arc length in meters
        complexity: number of control points (3-20)
        strategy: specific strategy name, or None for random

    Returns:
        path: Nx3 float32 array of dense centerline points
        strategy_name: which strategy was used
    """
    if strategy is None:
        strategy = random.choices(SHAPE_STRATEGIES, weights=STRATEGY_WEIGHTS, k=1)[0]

    center = (bounds_min + bounds_max) / 2.0
    room_size = bounds_max - bounds_min
    spread = min(room_size[0], room_size[1]) * 0.3  # cable spread relative to room

    # Scale spread by target length
    length_factor = min(target_length / 1.0, 2.0)  # normalize around 1m
    spread *= length_factor * 0.5

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if strategy == "random_bezier":
                # Random center point within room
                c = np.array([
                    np.random.uniform(bounds_min[0] + 0.5, bounds_max[0] - 0.5),
                    np.random.uniform(bounds_min[1] + 0.5, bounds_max[1] - 0.5),
                    floor_z + np.random.uniform(0.01, min(1.5, room_size[2] * 0.5)),
                ])
                path = gen_random_bezier(c, spread, complexity, floor_z)

            elif strategy == "catenary":
                # Two anchor points at different heights
                h1 = floor_z + np.random.uniform(0.3, min(2.0, room_size[2] * 0.6))
                h2 = floor_z + np.random.uniform(0.3, min(2.0, room_size[2] * 0.6))
                a = np.array([
                    np.random.uniform(bounds_min[0] + 0.5, bounds_max[0] - 0.5),
                    np.random.uniform(bounds_min[1] + 0.5, bounds_max[1] - 0.5),
                    h1
                ])
                b = np.array([
                    np.random.uniform(bounds_min[0] + 0.5, bounds_max[0] - 0.5),
                    np.random.uniform(bounds_min[1] + 0.5, bounds_max[1] - 0.5),
                    h2
                ])
                sag = np.random.uniform(0.2, 0.8)
                path = gen_catenary(a, b, sag_factor=sag)

            elif strategy == "wave":
                start = np.array([
                    np.random.uniform(bounds_min[0] + 0.3, bounds_max[0] - 0.3),
                    np.random.uniform(bounds_min[1] + 0.3, bounds_max[1] - 0.3),
                    floor_z + np.random.uniform(0.01, 0.5),
                ])
                direction = np.random.randn(3)
                direction[2] *= 0.2
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                end = start + direction * target_length
                # Clamp end to room bounds
                end[0] = np.clip(end[0], bounds_min[0] + 0.2, bounds_max[0] - 0.2)
                end[1] = np.clip(end[1], bounds_min[1] + 0.2, bounds_max[1] - 0.2)
                end[2] = max(end[2], floor_z + 0.01)
                amplitude = np.random.uniform(0.02, 0.15) * target_length
                frequency = np.random.uniform(1, max(2, complexity / 3))
                path = gen_wave(start, end, amplitude, frequency, floor_z)

            elif strategy == "helix":
                c = np.array([
                    np.random.uniform(bounds_min[0] + 1.0, bounds_max[0] - 1.0),
                    np.random.uniform(bounds_min[1] + 1.0, bounds_max[1] - 1.0),
                    floor_z + 0.01,
                ])
                radius = np.random.uniform(0.03, 0.2)
                num_turns = np.random.uniform(1.5, max(2, target_length / (2 * np.pi * radius)))
                height = np.random.uniform(0.1, min(1.0, room_size[2] * 0.3))
                path = gen_helix(c, radius, height, num_turns, floor_z)

            elif strategy == "floor_route":
                num_wp = max(3, complexity)
                path = gen_floor_route(floor_z, bounds_min, bounds_max, num_wp)

            elif strategy == "draped":
                h = np.random.uniform(0.5, min(2.0, room_size[2] * 0.5))
                start = np.array([
                    np.random.uniform(bounds_min[0] + 0.5, bounds_max[0] - 0.5),
                    np.random.uniform(bounds_min[1] + 0.5, bounds_max[1] - 0.5),
                    floor_z + h,
                ])
                end = np.array([
                    np.random.uniform(bounds_min[0] + 0.5, bounds_max[0] - 0.5),
                    np.random.uniform(bounds_min[1] + 0.5, bounds_max[1] - 0.5),
                    floor_z + h * np.random.uniform(0.5, 1.5),
                ])
                sag_depth = np.random.uniform(h * 0.3, h * 0.9)
                path = gen_draped(start, end, sag_depth, floor_z)

            elif strategy == "loop":
                c = np.array([
                    np.random.uniform(bounds_min[0] + 1.0, bounds_max[0] - 1.0),
                    np.random.uniform(bounds_min[1] + 1.0, bounds_max[1] - 1.0),
                    floor_z + np.random.uniform(0.01, 0.3),
                ])
                loop_size = np.random.uniform(0.1, min(0.5, spread * 0.5))
                loop_type = random.choice(["figure8", "trefoil", "simple_loop"])
                path = gen_loop(c, loop_size, floor_z, loop_type)

            else:
                path = gen_random_bezier(center, spread, complexity, floor_z)

            # Validate: path should have reasonable length and be within bounds
            length = arc_length(path)
            if length < 0.05:
                continue  # too short, retry

            # Clamp to room bounds (with small margin)
            margin = 0.1
            path[:, 0] = np.clip(path[:, 0], bounds_min[0] + margin, bounds_max[0] - margin)
            path[:, 1] = np.clip(path[:, 1], bounds_min[1] + margin, bounds_max[1] - margin)
            path[:, 2] = np.maximum(path[:, 2], floor_z + 0.002)

            # Scale path to approximately match target_length
            current_length = arc_length(path)
            if current_length > 1e-3:
                scale = target_length / current_length
                path_center = path.mean(axis=0)
                path = path_center + (path - path_center) * scale
                # Re-clamp after scaling
                path[:, 0] = np.clip(path[:, 0], bounds_min[0] + margin, bounds_max[0] - margin)
                path[:, 1] = np.clip(path[:, 1], bounds_min[1] + margin, bounds_max[1] - margin)
                path[:, 2] = np.maximum(path[:, 2], floor_z + 0.002)

            # Apply light smoothing
            path = laplacian_smooth(path, iterations=3, alpha=0.3)

            return path.astype(np.float32), strategy

        except Exception as e:
            print(f"    Path generation attempt {attempt+1} failed ({strategy}): {e}")
            continue

    # Fallback: simple straight line
    start = center.copy()
    start[2] = floor_z + 0.05
    end = start + np.array([target_length * 0.8, 0, 0])
    path = np.linspace(start, end, 100).astype(np.float32)
    return path, "fallback_line"


# ═══════════════════════════════════════════════════════════════════════════════
# 4b. CABLE BUNDLING — PARALLEL DLO PATH GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_bundle_paths(backbone_path, num_cables, base_radius):
    """
    Generate parallel cable paths offset from a backbone centerline.
    Simulates cables bundled through conduits, brackets, or cable trays.

    Each cable in the bundle runs roughly parallel to the backbone at a
    perpendicular offset of ~1.5–4× the cable radius, with small natural
    perturbations so they're not perfectly uniform.

    Args:
        backbone_path: Nx3 float32 array — the central route
        num_cables: number of parallel cables to generate
        base_radius: cable radius (used to compute offsets)

    Returns:
        list of Nx3 float32 arrays — one per cable
    """
    paths = []
    n_pts = len(backbone_path)
    if n_pts < 3 or num_cables < 1:
        return paths

    # Compute tangent vectors along the backbone
    tangents = np.zeros_like(backbone_path, dtype=np.float64)
    tangents[1:-1] = (backbone_path[2:] - backbone_path[:-2]) / 2.0
    tangents[0] = backbone_path[1] - backbone_path[0]
    tangents[-1] = backbone_path[-1] - backbone_path[-2]
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(tangent_norms, 1e-8)

    # Compute stable normal/binormal frame via cross product with up
    up = np.array([0.0, 0.0, 1.0])
    normals = np.cross(tangents, up)
    # Handle degenerate case (tangent nearly vertical)
    degenerate = np.linalg.norm(normals, axis=1) < 0.1
    normals[degenerate] = np.cross(tangents[degenerate], np.array([1.0, 0.0, 0.0]))
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    binormals = np.cross(tangents, normals)
    binormals = binormals / np.maximum(np.linalg.norm(binormals, axis=1, keepdims=True), 1e-8)

    for _ in range(num_cables):
        # Consistent offset angle for this cable (roughly constant along path)
        base_angle = np.random.uniform(0, 2 * np.pi)
        # Small angular drift along path for natural variation
        angle_drift = np.cumsum(np.random.normal(0, 0.01, n_pts))
        angles = base_angle + angle_drift

        # Offset distance: 1.5–4× the cable radius (tight bundle)
        offset_dist = np.random.uniform(1.5, 4.0) * base_radius

        # Compute perpendicular offset at each point
        cos_a = np.cos(angles)[:, None]
        sin_a = np.sin(angles)[:, None]
        offset = offset_dist * (cos_a * normals + sin_a * binormals)

        # Add small noise for natural variation (30% of radius)
        noise = np.random.normal(0, base_radius * 0.3, backbone_path.shape)
        noise = laplacian_smooth(noise.astype(np.float32), iterations=5, alpha=0.5)

        offset_path = (backbone_path + offset + noise).astype(np.float32)
        paths.append(offset_path)

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DLO MESH CREATION & MATERIAL ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

# Realistic cable colors: (name, RGB in linear color space)
DLO_COLORS = [
    ("black",      [0.010, 0.010, 0.010]),
    ("dark_grey",  [0.050, 0.050, 0.050]),
    ("grey",       [0.150, 0.150, 0.150]),
    ("white",      [0.700, 0.700, 0.700]),
    ("red",        [0.500, 0.020, 0.020]),
    ("blue",       [0.020, 0.050, 0.500]),
    ("yellow",     [0.600, 0.550, 0.020]),
    ("green",      [0.020, 0.350, 0.050]),
    ("orange",     [0.600, 0.200, 0.020]),
    ("brown",      [0.200, 0.100, 0.040]),
    ("purple",     [0.250, 0.020, 0.400]),
    ("cyan",       [0.020, 0.400, 0.450]),
    ("pink",       [0.500, 0.150, 0.250]),
]

# DLO material types with PBR parameters
DLO_MATERIAL_TYPES = [
    {"name": "rubber_smooth",  "roughness": (0.6, 0.9),  "metallic": 0.0, "specular": 0.3},
    {"name": "rubber_matte",   "roughness": (0.8, 1.0),  "metallic": 0.0, "specular": 0.1},
    {"name": "pvc_shiny",      "roughness": (0.2, 0.5),  "metallic": 0.0, "specular": 0.6},
    {"name": "braided_fabric", "roughness": (0.7, 0.95), "metallic": 0.0, "specular": 0.15},
    {"name": "metal_conduit",  "roughness": (0.3, 0.6),  "metallic": 0.8, "specular": 0.5},
    {"name": "silicone",       "roughness": (0.5, 0.8),  "metallic": 0.0, "specular": 0.4},
    {"name": "nylon",          "roughness": (0.4, 0.7),  "metallic": 0.0, "specular": 0.35},
    # Automotive-specific materials
    {"name": "corrugated_pp",  "roughness": (0.65, 0.85), "metallic": 0.0, "specular": 0.2},   # Black corrugated conduit
    {"name": "epdm_hose",      "roughness": (0.55, 0.75), "metallic": 0.0, "specular": 0.25},  # Automotive coolant/vacuum hose
    {"name": "steel_brake",    "roughness": (0.2, 0.4),   "metallic": 0.9, "specular": 0.6},   # Steel brake/fuel lines
    {"name": "aluminum_tube",  "roughness": (0.15, 0.35), "metallic": 0.85, "specular": 0.55}, # AC lines
]


def create_dlo_material(color_name, color_rgb, mat_type):
    """Create a PBR cable material in Blender."""
    roughness = np.random.uniform(*mat_type["roughness"])
    mat_name = f"DLO_{color_name}_{mat_type['name']}_{random.randint(0, 9999)}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (*color_rgb, 1.0)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = mat_type["metallic"]
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = mat_type["specular"]
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = mat_type["specular"]

    # Subtle noise for realism (cables aren't perfectly uniform)
    noise_tex = nodes.new('ShaderNodeTexNoise')
    noise_tex.inputs['Scale'].default_value = np.random.uniform(50, 200)
    noise_tex.inputs['Detail'].default_value = 5.0

    color_mix = nodes.new('ShaderNodeMixRGB')
    color_mix.blend_type = 'OVERLAY'
    color_mix.inputs['Fac'].default_value = np.random.uniform(0.02, 0.08)
    color_mix.inputs['Color1'].default_value = (*color_rgb, 1.0)
    links.new(noise_tex.outputs['Fac'], color_mix.inputs['Color2'])
    links.new(color_mix.outputs['Color'], bsdf.inputs['Base Color'])

    output_node = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    return mat, {"color": color_name, "material_type": mat_type["name"], "roughness": roughness}


def create_dlo_in_scene(path_points, radius, dlo_name, category_id, instance_id,
                        gt_sample_count):
    """
    Create a renderable DLO mesh from a centerline path.

    Process:
      1. Create a NURBS curve from path_points with bevel_depth (makes it a tube)
      2. Convert to mesh for BlenderProc compatibility
      3. Assign material, category_id, instance_id
      4. Build GT entry from the original path (exact centerline)

    Returns:
        mesh_obj: bpy.types.Object (mesh, for rendering)
        gt_entry: dict with centerline GT data
        material_info: dict describing the assigned material
    """
    # --- 1. Create Blender curve ---
    curve_data = bpy.data.curves.new(name=f"{dlo_name}_curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4   # cross-section smoothness
    curve_data.resolution_u = 12      # along-curve smoothness
    curve_data.fill_mode = 'FULL'

    # Use a NURBS spline — smoother than polyline
    spline = curve_data.splines.new('NURBS')
    # Subsample path for curve control points (too many = slow conversion)
    max_ctrl_points = 200
    if len(path_points) > max_ctrl_points:
        ctrl_points = resample_polyline(path_points, max_ctrl_points)
    else:
        ctrl_points = path_points

    spline.points.add(len(ctrl_points) - 1)
    for i, pt in enumerate(ctrl_points):
        spline.points[i].co = (float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
    spline.use_endpoint_u = True
    spline.order_u = min(4, len(ctrl_points))

    curve_obj = bpy.data.objects.new(f"{dlo_name}_curve", curve_data)
    bpy.context.scene.collection.objects.link(curve_obj)

    # --- 1b. Optional corrugated/ribbed tube profile ---
    # ~30% of DLOs get corrugated appearance (matching automotive conduits)
    taper_obj = None
    is_corrugated = np.random.random() < 0.3
    if is_corrugated:
        taper_curve = bpy.data.curves.new(name=f"{dlo_name}_taper", type='CURVE')
        taper_curve.dimensions = '2D'
        taper_spline = taper_curve.splines.new('POLY')
        path_length = arc_length(path_points)
        n_ribs = max(10, int(path_length / 0.005))  # ~5mm per rib period
        n_ribs = min(n_ribs, 200)
        taper_spline.points.add(n_ribs - 1)
        rib_amplitude = np.random.uniform(0.15, 0.3)  # 15-30% radius variation
        for ti in range(n_ribs):
            t_norm = ti / max(1, n_ribs - 1)
            r_val = 1.0 + rib_amplitude * math.sin(ti * math.pi)
            taper_spline.points[ti].co = (t_norm, r_val, 0.0, 1.0)
        taper_obj = bpy.data.objects.new(f"{dlo_name}_taper", taper_curve)
        bpy.context.scene.collection.objects.link(taper_obj)
        curve_data.taper_object = taper_obj

    # --- 2. Convert curve → mesh ---
    bpy.ops.object.select_all(action='DESELECT')
    curve_obj.select_set(True)
    bpy.context.view_layer.objects.active = curve_obj
    bpy.ops.object.convert(target='MESH')
    # curve_obj is now a mesh object (bpy renamed it, but reference is valid)
    mesh_obj = curve_obj

    # Clean up taper object (no longer needed after mesh conversion)
    if taper_obj is not None and taper_obj.name in bpy.data.objects:
        taper_data = taper_obj.data
        bpy.data.objects.remove(taper_obj, do_unlink=True)
        if taper_data and taper_data.users == 0:
            bpy.data.curves.remove(taper_data)

    if len(mesh_obj.data.vertices) < 3:
        # Conversion failed — remove and report
        bpy.data.objects.remove(mesh_obj, do_unlink=True)
        return None, None, None

    # --- 3. Assign material ---
    color_name, color_rgb = random.choice(DLO_COLORS)
    mat_type = random.choice(DLO_MATERIAL_TYPES)
    mat, material_info = create_dlo_material(color_name, color_rgb, mat_type)
    mesh_obj.data.materials.append(mat)

    # --- 4. Set BlenderProc properties ---
    mesh_obj.pass_index = category_id
    mesh_obj["category_id"] = category_id
    mesh_obj["instance"] = instance_id

    # --- 5. Build GT entry from the ORIGINAL generated path ---
    gt_points = resample_polyline(path_points, gt_sample_count)
    gt_entry = {
        "curve_id": sanitize_key(dlo_name),
        "curve_name": dlo_name,
        "spline_index": 0,
        "source_mesh": "procedural",
        "estimated_radius": float(radius),
        "points_world": gt_points.astype(np.float32),
    }

    return mesh_obj, gt_entry, material_info


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GROUND TRUTH, PROJECTION & RAY-CAST OCCLUSION
# ═══════════════════════════════════════════════════════════════════════════════

def project_points_to_image(points_cam, k_matrix, image_width, image_height):
    """Project 3D camera-space points to 2D image coordinates."""
    points_uv = np.full((points_cam.shape[0], 2), -1.0, dtype=np.float32)
    visibility = np.zeros(points_cam.shape[0], dtype=np.uint8)

    valid_depth = points_cam[:, 2] > 1e-6
    if not np.any(valid_depth):
        return points_uv, visibility

    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    x = points_cam[valid_depth, 0]
    y = points_cam[valid_depth, 1]
    z = points_cam[valid_depth, 2]
    points_uv[valid_depth, 0] = fx * x / z + cx
    points_uv[valid_depth, 1] = fy * y / z + cy

    in_frame = (
        (points_uv[:, 0] >= 0.0)
        & (points_uv[:, 0] < float(image_width))
        & (points_uv[:, 1] >= 0.0)
        & (points_uv[:, 1] < float(image_height))
    )
    visibility = np.logical_and(valid_depth, in_frame).astype(np.uint8)
    return points_uv, visibility


def raycast_visibility(bpy_scene, cam_to_world, points_world, depsgraph):
    """
    True occlusion check via ray-casting from camera to each skeleton node.
    Returns: uint8 array — 1 = visible, 0 = occluded
    """
    camera_origin = Vector(cam_to_world[:3, 3].tolist())
    occlusion_flags = np.zeros(len(points_world), dtype=np.uint8)
    tolerance = 0.015  # 15mm tolerance (accounts for cable radius)

    for i, point in enumerate(points_world):
        target = Vector(point.tolist())
        direction = target - camera_origin
        dist = direction.length
        if dist < 1e-6:
            occlusion_flags[i] = 1
            continue

        result, location, normal, face_idx, obj, matrix = bpy_scene.ray_cast(
            depsgraph, camera_origin, direction.normalized(),
            distance=dist + tolerance
        )

        if not result:
            occlusion_flags[i] = 1
        else:
            hit_dist = (location - camera_origin).length
            if hit_dist >= dist - tolerance:
                occlusion_flags[i] = 1
            else:
                occlusion_flags[i] = 0

    return occlusion_flags


def build_frame_ground_truth(curve_entries, frame_idx, cam_config, scene_index,
                             bpy_scene=None, depsgraph=None):
    """Build comprehensive per-frame GT for all DLOs."""
    k = cam_config["K"]
    w, h = cam_config["width"], cam_config["height"]

    world_to_cam = np.array(
        bproc.camera.get_world_to_cam_view_matrix(frame_idx), dtype=np.float32
    )
    cam_to_world = np.linalg.inv(world_to_cam).astype(np.float32)

    frame_gt = {
        "frame_index": int(frame_idx),
        "scene_index": int(scene_index),
        "image_size": {"width": int(w), "height": int(h)},
        "camera_intrinsics": k.tolist(),
        "world_to_cam": world_to_cam.tolist(),
        "cam_to_world": cam_to_world.tolist(),
        "stereo": cam_config["stereo"],
        "stereo_baseline_m": float(cam_config["baseline_m"]),
        "stereo_preset": cam_config["preset_name"],
        "curves": [],
    }

    for entry in curve_entries:
        pw = entry["points_world"]
        pw_h = np.concatenate([pw, np.ones((pw.shape[0], 1), dtype=np.float32)], axis=1)
        pc = (world_to_cam @ pw_h.T).T[:, :3].astype(np.float32)
        puv, vis_proj = project_points_to_image(pc, k, w, h)

        if bpy_scene is not None and depsgraph is not None:
            vis_ray = raycast_visibility(bpy_scene, cam_to_world, pw, depsgraph)
            vis_combined = (vis_proj.astype(bool) & vis_ray.astype(bool)).astype(np.uint8)
        else:
            vis_ray = vis_proj.copy()
            vis_combined = vis_proj.copy()

        node_depths = pc[:, 2].astype(np.float32)
        if len(pw) > 1:
            inter_dists = np.linalg.norm(np.diff(pw, axis=0), axis=1).tolist()
        else:
            inter_dists = []

        frame_gt["curves"].append({
            "curve_id": entry["curve_id"],
            "curve_name": entry["curve_name"],
            "spline_index": int(entry["spline_index"]),
            "source_mesh": entry["source_mesh"],
            "estimated_radius": float(entry["estimated_radius"]),
            "num_nodes": int(len(pw)),
            "points_world": pw.tolist(),
            "points_cam": pc.tolist(),
            "points_uv": puv.tolist(),
            "node_depths": node_depths.tolist(),
            "visibility_projection": vis_proj.tolist(),
            "visibility_raycast": vis_ray.tolist(),
            "visibility_combined": vis_combined.tolist(),
            "inter_node_distances": inter_dists,
            "total_arc_length": float(sum(inter_dists)),
        })

    return frame_gt


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEPTH NOISE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_depth_noise(depth_clean, sigma=0.002, thin_object_mask=None):
    """
    Simulate realistic depth sensor noise:
    1. Depth-proportional Gaussian noise
    2. Missing depth on thin object edges
    3. 12-bit quantization
    4. Flying pixels at discontinuities
    """
    depth_noisy = depth_clean.copy().astype(np.float32)
    valid = depth_noisy > 0
    if not np.any(valid):
        return depth_noisy

    # 1. Depth-proportional Gaussian noise: σ(z) ≈ σ_base × z²
    noise = np.random.normal(0, 1, depth_noisy.shape).astype(np.float32)
    depth_sigma = sigma * (depth_noisy ** 2)
    depth_sigma[~valid] = 0
    depth_noisy[valid] += (noise * depth_sigma)[valid]

    # 2. Missing depth on thin objects (DLO edges)
    if thin_object_mask is not None:
        try:
            from scipy import ndimage
            eroded = ndimage.binary_erosion(thin_object_mask, iterations=1)
            edge_px = thin_object_mask & ~eroded
            drop_prob = np.random.uniform(0.4, 0.7)
            drop = edge_px & (np.random.random(depth_noisy.shape) < drop_prob)
            depth_noisy[drop] = 0.0
        except ImportError:
            pass

    # 3. Depth quantization (12-bit structured light)
    quant = 0.001
    depth_noisy[valid] = np.round(depth_noisy[valid] / quant) * quant

    # 4. Flying pixels at depth discontinuities
    if np.any(valid):
        gx = np.abs(np.diff(depth_noisy, axis=1, prepend=0))
        gy = np.abs(np.diff(depth_noisy, axis=0, prepend=0))
        disc = (gx > 0.05) | (gy > 0.05)
        flying = disc & (np.random.random(depth_noisy.shape) < 0.3)
        n_flying = np.sum(flying & valid)
        if n_flying > 0:
            depth_noisy[flying & valid] *= np.random.uniform(0.95, 1.05, size=n_flying)

    return np.maximum(depth_noisy, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SCENE RANDOMIZATION (LIGHTING, DISTRACTORS)
# ═══════════════════════════════════════════════════════════════════════════════

def create_distractor_objects(floor, num_objects):
    """Spawn random geometric distractors on the floor for partial occlusions."""
    distractors = []
    if num_objects <= 0:
        return distractors

    actual = np.random.randint(0, num_objects + 1)
    if actual == 0:
        return distractors

    floor_bbox = floor.get_bound_box()
    fmin = np.min(floor_bbox, axis=0)
    fmax = np.max(floor_bbox, axis=0)

    for _ in range(actual):
        shape = random.choice(['cube', 'cylinder', 'sphere'])
        if shape == 'cube':
            s = np.random.uniform(0.05, 0.3)
            obj = bproc.object.create_primitive("CUBE")
            obj.set_scale([s, s, s])
        elif shape == 'cylinder':
            obj = bproc.object.create_primitive("CYLINDER")
            r = np.random.uniform(0.03, 0.15)
            h = np.random.uniform(0.05, 0.4)
            obj.set_scale([r, r, h])
        else:
            obj = bproc.object.create_primitive("SPHERE")
            r = np.random.uniform(0.03, 0.2)
            obj.set_scale([r, r, r])

        obj.set_location([
            np.random.uniform(fmin[0] + 0.5, fmax[0] - 0.5),
            np.random.uniform(fmin[1] + 0.5, fmax[1] - 0.5),
            fmin[2] + 0.1,
        ])
        obj.set_rotation_euler([
            np.random.uniform(0, np.pi),
            np.random.uniform(0, np.pi),
            np.random.uniform(0, 2 * np.pi),
        ])

        mat = bpy.data.materials.new(name=f"distractor_{random.randint(0, 9999)}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            g = np.random.uniform(0.05, 0.6)
            bsdf.inputs['Base Color'].default_value = (
                g + np.random.uniform(-0.05, 0.05),
                g + np.random.uniform(-0.05, 0.05),
                g + np.random.uniform(-0.05, 0.05), 1.0)
            bsdf.inputs['Roughness'].default_value = np.random.uniform(0.3, 0.9)
        obj.blender_obj.data.materials.append(mat)
        obj.set_cp("category_id", 0)
        distractors.append(obj)

    return distractors


def randomize_scene_lighting(room_objects):
    """Apply diverse lighting: ceiling emission + extra point/area/spot lights."""
    ceiling = bproc.filter.one_by_attr(room_objects, "name", "Ceiling")
    if ceiling:
        bproc.lighting.light_surface(
            [ceiling],
            emission_strength=np.random.uniform(3.0, 15.0),
            emission_color=[
                np.random.uniform(0.6, 1.0),
                np.random.uniform(0.6, 1.0),
                np.random.uniform(0.6, 1.0), 1.0
            ]
        )

    lights = []
    floor = bproc.filter.one_by_attr(room_objects, "name", "Floor")
    if floor:
        fbbox = floor.get_bound_box()
        fmin, fmax = np.min(fbbox, axis=0), np.max(fbbox, axis=0)

        for _ in range(np.random.randint(1, 4)):
            lt = random.choice(["POINT", "AREA", "SPOT"])
            ld = bpy.data.lights.new(name=f"rlight_{random.randint(0,9999)}", type=lt)
            temp = random.choice(["warm", "neutral", "cool"])
            ld.color = {"warm": (1,0.85,0.65), "cool": (0.75,0.85,1), "neutral": (1,1,1)}[temp]
            ld.energy = np.random.uniform(10, 200)
            if lt == "AREA":
                ld.size = np.random.uniform(0.3, 1.5)
            elif lt == "SPOT":
                ld.spot_size = np.random.uniform(0.5, 1.5)

            lo = bpy.data.objects.new(name=ld.name, object_data=ld)
            bpy.context.scene.collection.objects.link(lo)
            lo.location = (
                np.random.uniform(fmin[0], fmax[0]),
                np.random.uniform(fmin[1], fmax[1]),
                np.random.uniform(2.0, 4.0),
            )
            lights.append(lo)

    return lights


def cleanup_lights(lights):
    """Remove dynamically created lights."""
    for lo in lights:
        if lo and lo.name in bpy.data.objects:
            bpy.data.objects.remove(lo, do_unlink=True)


def cleanup_dlo_meshes(mesh_objects):
    """Remove procedurally generated DLO mesh objects from the scene."""
    for obj in mesh_objects:
        if obj and obj.name in bpy.data.objects:
            mesh_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. STEREO RENDERING & OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stereo_cam2world(left_cam2world, baseline_m):
    """Right camera = left camera offset along local X by baseline."""
    right = left_cam2world.copy()
    right[:3, 3] += left_cam2world[:3, 0] * baseline_m
    return right


def save_frame_data(frame_dir, frame_idx, render_data, frame_gt, cam_config,
                    depth_noisy=None, normals=None, instance_segmap=None,
                    semantic_segmap=None, rgb_right=None, depth_right=None):
    """Save all data for a single frame into a structured directory."""
    os.makedirs(frame_dir, exist_ok=True)

    # RGB left
    if "colors" in render_data and frame_idx < len(render_data["colors"]):
        from PIL import Image
        Image.fromarray(render_data["colors"][frame_idx]).save(
            os.path.join(frame_dir, "rgb_left.png"))

    # RGB right
    if rgb_right is not None:
        from PIL import Image
        Image.fromarray(rgb_right).save(os.path.join(frame_dir, "rgb_right.png"))

    # Depth clean
    if "depth" in render_data and frame_idx < len(render_data["depth"]):
        np.savez_compressed(os.path.join(frame_dir, "depth_clean.npz"),
                            depth=render_data["depth"][frame_idx].astype(np.float32))

    # Depth noisy
    if depth_noisy is not None:
        np.savez_compressed(os.path.join(frame_dir, "depth_noisy.npz"),
                            depth=depth_noisy.astype(np.float32))

    # Depth right
    if depth_right is not None:
        np.savez_compressed(os.path.join(frame_dir, "depth_right_clean.npz"),
                            depth=depth_right.astype(np.float32))

    # Normals
    if normals is not None:
        np.savez_compressed(os.path.join(frame_dir, "normals.npz"),
                            normals=normals.astype(np.float32))

    # Instance segmap
    if instance_segmap is not None:
        from PIL import Image
        Image.fromarray(instance_segmap.astype(np.uint16)).save(
            os.path.join(frame_dir, "instance_segmap.png"))

    # Semantic segmap (DLO vs background)
    if semantic_segmap is not None:
        from PIL import Image
        Image.fromarray(semantic_segmap.astype(np.uint8)).save(
            os.path.join(frame_dir, "semantic_segmap.png"))

    # Curve GT JSON
    if frame_gt is not None:
        with open(os.path.join(frame_dir, "curve_gt.json"), "w") as f:
            json.dump(frame_gt, f, indent=2)

    # Camera JSON
    cam = {
        "camera_intrinsics_K": cam_config["K"].tolist(),
        "image_width": cam_config["width"],
        "image_height": cam_config["height"],
        "stereo": cam_config["stereo"],
        "stereo_baseline_m": cam_config["baseline_m"],
        "stereo_preset": cam_config["preset_name"],
    }
    if frame_gt:
        cam["world_to_cam_left"] = frame_gt.get("world_to_cam")
        cam["cam_to_world_left"] = frame_gt.get("cam_to_world")
    with open(os.path.join(frame_dir, "camera.json"), "w") as f:
        json.dump(cam, f, indent=2)


def write_dataset_manifest(output_dir, scene_records):
    """Write a global manifest indexing all scenes and frames."""
    manifest = {
        "dataset_name": "DLO_3D_Segmentation_Synthetic",
        "generator": "render_dlo_dataset.py",
        "generator_mode": "procedural",
        "schema_version": 3,
        "total_scenes": len(scene_records),
        "total_frames": sum(r["num_frames"] for r in scene_records),
        "stereo": scene_records[0]["stereo"] if scene_records else False,
        "scenes": scene_records,
    }
    path = os.path.join(output_dir, "dataset_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE — INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

bproc.init()

cam_config = get_camera_config(args)
IMAGE_WIDTH = cam_config["width"]
IMAGE_HEIGHT = cam_config["height"]
K_MATRIX = cam_config["K"]
STEREO = cam_config["stereo"]
BASELINE = cam_config["baseline_m"]

print(f"\n{'='*60}")
print(f"  DLO DATASET GENERATOR — PROCEDURAL MODE")
print(f"{'='*60}")
print(f"  Camera: {cam_config['preset_name']}")
print(f"  Resolution: {IMAGE_WIDTH}×{IMAGE_HEIGHT}")
print(f"  Stereo: {STEREO} (baseline={BASELINE*1000:.1f}mm)")
print(f"  Target: {args.num_scenes} scenes × {args.poses_per_scene} poses "
      f"= {args.num_scenes * args.poses_per_scene} frames")
print(f"  DLOs/scene: {args.min_dlos}–{args.max_dlos}")
print(f"  DLO length: {args.min_dlo_length}–{args.max_dlo_length}m")
print(f"  DLO radius: {args.min_dlo_radius*1000:.1f}–{args.max_dlo_radius*1000:.1f}mm")
print(f"  Complexity: {args.min_complexity}–{args.max_complexity} control points")
print(f"{'='*60}\n")

# Render settings
bproc.renderer.set_render_devices(desired_gpu_device_type=["OPTIX", "CUDA"])
bproc.renderer.set_max_amount_of_samples(args.render_samples)
bproc.renderer.set_noise_threshold(args.render_noise_threshold)
bproc.renderer.set_denoiser("OPTIX")
bproc.renderer.set_light_bounces(max_bounces=36)
bproc.renderer.enable_depth_output(activate_antialiasing=True, antialiasing_distance_max=4)
bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(
    map_by=["category_id", "instance", "name"],
    default_values={'category_id': 0, 'instance': 0, 'name': ""}
)

# Load room materials
print("Loading CCTextures...")
try:
    available_materials = [
        n for n in os.listdir(args.cc_material_path)
        if os.path.isdir(os.path.join(args.cc_material_path, n))
    ]
except FileNotFoundError:
    raise RuntimeError(f"CCTextures not found: {args.cc_material_path}")
if not available_materials:
    raise RuntimeError(f"No materials in {args.cc_material_path}")
print(f"Found {len(available_materials)} room materials.")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. MAIN GENERATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

scene_records = []
total_dlos_generated = 0
strategy_counts = {s: 0 for s in SHAPE_STRATEGIES + ["fallback_line"]}

for scene_i in range(args.num_scenes):
    print(f"\n{'─'*60}")
    print(f"  SCENE {scene_i + 1} / {args.num_scenes}")
    print(f"{'─'*60}")

    bproc.utility.reset_keyframes()

    # --- Camera ---
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    bproc.camera.set_intrinsics_from_K_matrix(K_MATRIX.tolist(), IMAGE_WIDTH, IMAGE_HEIGHT)
    cam_obj.data.clip_start = 0.05
    cam_obj.data.clip_end = 100.0

    # --- Room ---
    num_mats = np.random.randint(2, 4)
    mat_names = random.sample(available_materials, min(num_mats, len(available_materials)))
    room_mats = bproc.loader.load_ccmaterials(args.cc_material_path, used_assets=mat_names)

    room_objects = bproc.constructor.construct_random_room(
        used_floor_area=np.random.uniform(25, 35),
        interior_objects=[],
        materials=room_mats,
        fac_from_square_room=1.5
    )
    floor = bproc.filter.one_by_attr(room_objects, "name", "Floor")
    if not floor:
        print("  ⚠ No floor — skipping.")
        bproc.object.delete_multiple(room_objects)
        continue

    floor_bbox = floor.get_bound_box()
    floor_min = np.min(floor_bbox, axis=0)
    floor_max = np.max(floor_bbox, axis=0)
    floor_z = float(floor_min[2])

    # ─── PROCEDURAL DLO GENERATION ───────────────────────────────────────────
    # Use lognormal distribution: biased toward denser scenes (peak ~8 DLOs)
    num_dlos = int(np.clip(
        np.random.lognormal(mean=np.log(8), sigma=0.5),
        args.min_dlos, args.max_dlos
    ))
    print(f"  Generating {num_dlos} procedural DLOs...")

    dlo_mesh_objects = []   # bpy objects (for cleanup)
    dlo_bproc_objects = []  # BlenderProc wrappers (for BVH, render)
    curve_entries = []      # GT entries
    material_assignments = []
    dlo_strategies = []

    # Helper: create a single DLO from a path and register it
    def _create_and_register_dlo(path, radius, dlo_idx, strategy_label):
        dlo_name = f"DLO_{scene_i:04d}_{dlo_idx:02d}"
        category_id = dlo_idx + 1
        instance_id = dlo_idx + 1
        mesh_obj, gt_entry, mat_info = create_dlo_in_scene(
            path, radius, dlo_name, category_id, instance_id,
            args.curve_gt_points
        )
        if mesh_obj is None:
            print(f"    ⚠ DLO {dlo_idx} creation failed — skipping.")
            return False
        dlo_mesh_objects.append(mesh_obj)
        curve_entries.append(gt_entry)
        try:
            bproc_obj = bproc.types.MeshObject(mesh_obj)
            bproc_obj.set_cp("category_id", category_id)
            bproc_obj.set_cp("instance", instance_id)
            bproc_obj.set_cp("name", dlo_name)
            dlo_bproc_objects.append(bproc_obj)
        except Exception as e:
            print(f"    ⚠ BlenderProc wrap failed for DLO {dlo_idx}: {e}")
            return False
        if mat_info:
            mat_info["dlo_name"] = dlo_name
            mat_info["strategy"] = strategy_label
            mat_info["length"] = float(arc_length(path))
            mat_info["radius"] = float(radius)
            material_assignments.append(mat_info)
        actual_len = arc_length(path)
        print(f"    ✓ {dlo_name}: {strategy_label}, L={actual_len:.2f}m, "
              f"r={radius*1000:.1f}mm")
        return True

    dlo_i = 0
    remaining = num_dlos
    while remaining > 0 and dlo_i < num_dlos + 5:  # +5 slack for failed attempts
        # ~40% chance of generating a cable bundle (if enough DLOs remaining)
        use_bundle = np.random.random() < 0.4 and remaining >= 2

        if use_bundle:
            bundle_size = min(np.random.randint(2, 6), remaining)  # 2-5 cables
            target_length = np.random.uniform(args.min_dlo_length, args.max_dlo_length)
            radius = np.random.uniform(args.min_dlo_radius, args.max_dlo_radius)
            complexity = np.random.randint(args.min_complexity, args.max_complexity + 1)

            # Generate backbone path
            backbone, strategy_used = generate_dlo_path(
                floor_min, floor_max, floor_z, target_length, complexity
            )
            strategy_counts[strategy_used] = strategy_counts.get(strategy_used, 0) + 1

            # Generate parallel offset paths
            bundle_paths = generate_bundle_paths(backbone, bundle_size, radius)

            print(f"    ── Bundle ({bundle_size} cables, {strategy_used} backbone) ──")
            created = 0
            for bp in bundle_paths:
                # Each cable in bundle gets slightly different radius
                cable_radius = radius * np.random.uniform(0.7, 1.3)
                strategy_label = f"bundle_{strategy_used}"
                if _create_and_register_dlo(bp, cable_radius, dlo_i, strategy_label):
                    dlo_strategies.append(strategy_label)
                    dlo_i += 1
                    created += 1
            remaining -= created

        else:
            # Independent DLO (original behavior)
            target_length = np.random.uniform(args.min_dlo_length, args.max_dlo_length)
            radius = np.random.uniform(args.min_dlo_radius, args.max_dlo_radius)
            complexity = np.random.randint(args.min_complexity, args.max_complexity + 1)

            path, strategy_used = generate_dlo_path(
                floor_min, floor_max, floor_z, target_length, complexity
            )
            strategy_counts[strategy_used] = strategy_counts.get(strategy_used, 0) + 1

            if _create_and_register_dlo(path, radius, dlo_i, strategy_used):
                dlo_strategies.append(strategy_used)
                dlo_i += 1
                remaining -= 1
            else:
                remaining -= 1  # still decrement to avoid infinite loop

    total_dlos_generated += len(dlo_bproc_objects)

    if not dlo_bproc_objects:
        print("  ⚠ No valid DLOs created — skipping scene.")
        cleanup_dlo_meshes(dlo_mesh_objects)
        bproc.object.delete_multiple(room_objects)
        continue

    # ─── DISTRACTOR OBJECTS ──────────────────────────────────────────────────
    distractor_objects = create_distractor_objects(floor, args.num_distractor_objects)
    if distractor_objects:
        print(f"  + {len(distractor_objects)} distractor objects.")

    # ─── LIGHTING ────────────────────────────────────────────────────────────
    extra_lights = []
    if args.randomize_lighting:
        extra_lights = randomize_scene_lighting(room_objects)
    else:
        ceiling = bproc.filter.one_by_attr(room_objects, "name", "Ceiling")
        if ceiling:
            bproc.lighting.light_surface(
                [ceiling],
                emission_strength=np.random.uniform(5, 10),
                emission_color=[np.random.uniform(0.7, 1) for _ in range(3)] + [1.0]
            )

    # ─── CAMERA POSES ────────────────────────────────────────────────────────
    print(f"  Sampling {args.poses_per_scene} camera poses...")

    all_scene_objs = dlo_bproc_objects + room_objects + distractor_objects
    # Point of interest = center of all DLO bounding boxes
    all_dlo_bboxes = np.concatenate([o.get_bound_box() for o in dlo_bproc_objects])
    poi = np.mean(all_dlo_bboxes, axis=0)

    bvh_tree = bproc.object.create_bvh_tree_multi_objects(all_scene_objs)

    poses_added = 0
    for _ in range(args.poses_per_scene):
        for attempt in range(200):
            # 30% close-up mode: target a single DLO at very close range
            close_up = np.random.random() < 0.3

            if close_up:
                # Pick a random DLO and look at its center
                target_dlo = random.choice(dlo_bproc_objects)
                target_bbox = target_dlo.get_bound_box()
                local_poi = np.mean(target_bbox, axis=0)
                # Small random offset from DLO center
                local_poi += np.random.uniform(-0.05, 0.05, 3)
                loc = bproc.sampler.shell(
                    center=local_poi, radius_min=0.2, radius_max=0.6,
                    elevation_min=5, elevation_max=85
                )
                look_at = local_poi
            else:
                # Standard view: close-to-medium range, random offset from POI
                look_at = poi + np.random.uniform(-0.2, 0.2, 3)
                loc = bproc.sampler.shell(
                    center=look_at, radius_min=0.3, radius_max=1.5,
                    elevation_min=5, elevation_max=85
                )

            # Check camera is inside the room
            if not (floor_min[0] < loc[0] < floor_max[0] and
                    floor_min[1] < loc[1] < floor_max[1]):
                continue

            rot = bproc.camera.rotation_from_forward_vec(look_at - loc)
            c2w = bproc.math.build_transformation_mat(loc, rot)

            if not bproc.camera.perform_obstacle_in_view_check(c2w, {"min": 0.15}, bvh_tree):
                bproc.camera.add_camera_pose(c2w)
                poses_added += 1
                break

    print(f"  Added {poses_added} poses.")

    # ─── RENDER ──────────────────────────────────────────────────────────────
    if poses_added > 0:
        print(f"  Rendering {poses_added} frames (left)...")
        data = bproc.renderer.render()

        # Stereo right eye
        right_data = None
        if STEREO:
            print(f"  Rendering {poses_added} frames (right)...")
            # Save left poses, add right poses
            left_w2c_list = []
            for fi in range(poses_added):
                left_w2c_list.append(np.array(
                    bproc.camera.get_world_to_cam_view_matrix(fi), dtype=np.float32
                ))

            bproc.utility.reset_keyframes()
            for w2c in left_w2c_list:
                left_c2w = np.linalg.inv(w2c)
                right_c2w = compute_stereo_cam2world(left_c2w, BASELINE)
                bproc.camera.add_camera_pose(right_c2w)
            right_data = bproc.renderer.render()

        # --- Per-frame output ---
        scene_dir = os.path.join(args.output_dir, f"scene_{scene_i:04d}")
        os.makedirs(scene_dir, exist_ok=True)

        bpy_scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()

        frame_records = []
        for fi in range(poses_added):
            frame_dir = os.path.join(scene_dir, f"frame_{fi:04d}")

            # GT
            frame_gt = build_frame_ground_truth(
                curve_entries, fi, cam_config, scene_i,
                bpy_scene=bpy_scene, depsgraph=depsgraph
            ) if curve_entries else None

            # Depth noise
            depth_noisy = None
            if args.add_depth_noise and "depth" in data and fi < len(data["depth"]):
                thin_mask = None
                if "category_id_segmaps" in data and fi < len(data["category_id_segmaps"]):
                    thin_mask = data["category_id_segmaps"][fi] > 0
                depth_noisy = simulate_depth_noise(
                    data["depth"][fi], sigma=args.depth_noise_sigma,
                    thin_object_mask=thin_mask
                )

            # Normals
            normals = data["normals"][fi] if "normals" in data and fi < len(data["normals"]) else None

            # Segmaps
            inst_seg = data["instance_segmaps"][fi] if "instance_segmaps" in data and fi < len(data["instance_segmaps"]) else None
            sem_seg = None
            if "category_id_segmaps" in data and fi < len(data["category_id_segmaps"]):
                sem_seg = (data["category_id_segmaps"][fi] > 0).astype(np.uint8) * 255

            # Right eye
            rgb_r = right_data["colors"][fi] if right_data and fi < len(right_data.get("colors", [])) else None
            depth_r = right_data["depth"][fi] if right_data and fi < len(right_data.get("depth", [])) else None

            save_frame_data(
                frame_dir, fi, data, frame_gt, cam_config,
                depth_noisy=depth_noisy, normals=normals,
                instance_segmap=inst_seg, semantic_segmap=sem_seg,
                rgb_right=rgb_r, depth_right=depth_r,
            )

            frame_records.append({
                "frame_index": fi,
                "path": f"frame_{fi:04d}",
                "has_stereo": rgb_r is not None,
                "has_depth_noisy": depth_noisy is not None,
                "has_curve_gt": frame_gt is not None,
                "num_curves": len(frame_gt["curves"]) if frame_gt else 0,
            })

        # COCO annotations (optional — for 2D detection baselines)
        if args.write_coco:
            try:
                bproc.writer.write_coco_annotations(
                    os.path.join(args.output_dir, 'coco_data'),
                    instance_segmaps=data["instance_segmaps"],
                    instance_attribute_maps=data["instance_attribute_maps"],
                    colors=data["colors"],
                    color_file_format="JPEG"
                )
            except Exception as e:
                print(f"  ⚠ COCO write failed: {e}")

        # Scene metadata
        scene_meta = {
            "scene_index": scene_i,
            "num_frames": poses_added,
            "stereo": STEREO,
            "baseline_m": BASELINE,
            "num_dlos": len(dlo_bproc_objects),
            "num_distractors": len(distractor_objects),
            "dlo_configs": material_assignments,
            "dlo_strategies": dlo_strategies,
            "camera_preset": cam_config["preset_name"],
        }
        with open(os.path.join(scene_dir, "metadata.json"), "w") as f:
            json.dump(scene_meta, f, indent=2)

        scene_records.append({
            "scene_index": scene_i,
            "path": f"scene_{scene_i:04d}",
            "num_frames": poses_added,
            "stereo": STEREO,
            "num_curves": len(curve_entries),
            "strategies": dlo_strategies,
            "frames": frame_records,
        })

        print(f"  ✓ Scene {scene_i+1} saved ({poses_added} frames, "
              f"{len(dlo_bproc_objects)} DLOs)")
    else:
        print(f"  ⚠ No camera poses — nothing rendered.")

    # ─── CLEANUP ─────────────────────────────────────────────────────────────
    if distractor_objects:
        bproc.object.delete_multiple(distractor_objects)
    cleanup_lights(extra_lights)
    cleanup_dlo_meshes(dlo_mesh_objects)
    bproc.object.delete_multiple(room_objects)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

if scene_records:
    write_dataset_manifest(args.output_dir, scene_records)

total_frames = sum(r["num_frames"] for r in scene_records)

print(f"\n{'═'*60}")
print(f"  DATASET GENERATION COMPLETE")
print(f"{'═'*60}")
print(f"  Scenes:       {len(scene_records)} / {args.num_scenes}")
print(f"  Total frames: {total_frames}")
print(f"  Total DLOs:   {total_dlos_generated}")
print(f"  Stereo:       {'Yes' if STEREO else 'No'}")
print(f"  Depth noise:  {'Yes' if args.add_depth_noise else 'No'}")
print(f"  Camera:       {cam_config['preset_name']}")
print(f"  Output:       {args.output_dir}")
print(f"\n  Shape strategy distribution:")
for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
    if count > 0:
        print(f"    {strat:20s}: {count:5d} ({100*count/max(1,total_dlos_generated):.1f}%)")
print(f"{'═'*60}")
