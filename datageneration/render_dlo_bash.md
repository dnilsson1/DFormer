# DLO Dataset Rendering Guide

Two complementary scripts for generating synthetic 3D curve extraction training data.

| Script | Purpose | DLO Source |
|---|---|---|
| `render_dlo_dataset.py` | Procedural DLO generation at scale | Generated (7 shape strategies) |
| `render_scene_nurbs.py` | Assembly rendering with GT curves | Loaded from `.blend` file |

---

## 1. `render_dlo_dataset.py` — Procedural DLO Dataset

Generates diverse training data with procedural cables in random rooms. No `.blend` file needed.

### Quick Start

```bash
# Test run (15 images, monocular, fast):
blenderproc run render_dlo_dataset.py test_output ./cctextures \
    --num_scenes 3 --poses_per_scene 5 --stereo_camera none

# Full 50k dataset for GNN training (ZED2 stereo):
blenderproc run render_dlo_dataset.py dlo_dataset ./cctextures \
    --num_scenes 5000 --poses_per_scene 10 --stereo_camera zed2

# D435i stereo, 10k images:
blenderproc run render_dlo_dataset.py dlo_d435i ./cctextures \
    --num_scenes 1000 --poses_per_scene 10 --stereo_camera realsense_d435i

# High quality render (more samples, less render noise):
blenderproc run render_dlo_dataset.py dlo_hq ./cctextures \
    --num_scenes 500 --poses_per_scene 10 --stereo_camera realsense_d435i \
    --render_samples 64 --render_noise_threshold 0.002

# Dense cable scenes (automotive-like density):
blenderproc run render_dlo_dataset.py dlo_dense ./cctextures \
    --num_scenes 2000 --poses_per_scene 10 --stereo_camera realsense_d435i \
    --min_dlos 5 --max_dlos 15

# Include optional COCO annotations for 2D detection baselines:
blenderproc run render_dlo_dataset.py dlo_dataset ./cctextures \
    --num_scenes 500 --poses_per_scene 10 --stereo_camera zed2 --write_coco

# With depth noise (if NOT doing noise as training-time augmentation):
blenderproc run render_dlo_dataset.py dlo_noisy ./cctextures \
    --num_scenes 500 --poses_per_scene 10 --add_depth_noise --depth_noise_sigma 0.003
```

### All CLI Arguments

```
Positional:
  output_dir                Output directory (default: dlo_dataset)
  cc_material_path          Path to CCTextures folder

Scene generation:
  --num_scenes N            Number of unique scenes (default: 5000)
  --poses_per_scene N       Camera poses per scene (default: 10)

Procedural DLO settings:
  --min_dlos N              Min DLOs per scene (default: 1)
  --max_dlos N              Max DLOs per scene (default: 8)
  --min_dlo_length F        Min arc length in meters (default: 0.3)
  --max_dlo_length F        Max arc length in meters (default: 3.0)
  --min_dlo_radius F        Min cross-section radius in meters (default: 0.001 = 1mm)
  --max_dlo_radius F        Max cross-section radius in meters (default: 0.020 = 20mm)
  --min_complexity N        Min control points for shape (default: 3)
  --max_complexity N        Max control points for shape (default: 20)
  --curve_gt_points N       GT skeleton nodes per curve (default: 64)

Stereo camera:
  --stereo_camera PRESET    zed2 | realsense_d435 | realsense_d435i | oak_d | custom | none
  --stereo_baseline F       Override preset baseline (meters)
  --image_width N           Render width (default: from preset)
  --image_height N          Render height (default: from preset)

Depth noise (off by default):
  --add_depth_noise         Enable depth noise simulation
  --depth_noise_sigma F     Noise std-dev in meters (default: 0.002)

Rendering quality:
  --render_samples N        Cycles samples (default: 32, higher = cleaner)
  --render_noise_threshold F  Adaptive sampling threshold (default: 0.005)

Domain randomization:
  --num_distractor_objects N  Max distractors per scene (default: 5, 0 = disable)
  --randomize_lighting      Randomize beyond ceiling emission (default: on)

Output:
  --write_coco              Write COCO annotations (off by default)
```

### Output Structure

```
output/
├── dataset_manifest.json          # Global index of all scenes/frames
├── scene_0000/
│   ├── metadata.json              # Scene config (DLO params, strategies, materials)
│   ├── frame_0000/
│   │   ├── rgb_left.png           # Left RGB image
│   │   ├── rgb_right.png          # Right RGB image (stereo only)
│   │   ├── depth_clean.npz        # Clean depth map (float32, meters)
│   │   ├── depth_right_clean.npz  # Right depth (stereo only)
│   │   ├── depth_noisy.npz        # Noisy depth (if --add_depth_noise)
│   │   ├── normals.npz            # Surface normal map (float32, XYZ)
│   │   ├── instance_segmap.png    # Per-pixel DLO instance IDs (uint16)
│   │   ├── semantic_segmap.png    # Binary DLO vs background (uint8)
│   │   ├── curve_gt.json          # 3D curve ground truth
│   │   └── camera.json            # Camera intrinsics + extrinsics
│   ├── frame_0001/
│   │   └── ...
│   └── ...
└── coco_data/                     # Only if --write_coco
    └── ...
```

### Shape Strategies

7 procedural curve generation strategies with weighted sampling:

| Strategy | Weight | Description |
|---|---|---|
| `random_bezier` | 25% | Smooth curves from random control points |
| `catenary` | 15% | Hanging cables between anchor points |
| `wave` | 15% | Sinusoidal cables along surfaces |
| `floor_route` | 15% | Cables routed along floors with turns |
| `draped` | 12% | Cables sagging between elevated supports |
| `loop` | 10% | Self-crossing (figure-8, trefoil) |
| `helix` | 8% | Coiled/spiral cables |

~30% of DLOs get a corrugated/ribbed tube profile (matching automotive conduits).

### DLO Material Types

| Material | Description |
|---|---|
| `rubber_smooth` | Standard smooth rubber insulation |
| `rubber_matte` | Matte rubber (low specular) |
| `pvc_shiny` | Shiny PVC cable jacket |
| `braided_fabric` | Fabric-braided cable sheath |
| `metal_conduit` | Metallic flexible conduit |
| `silicone` | Silicone tubing |
| `nylon` | Nylon cable sleeve |
| `corrugated_pp` | Black corrugated polypropylene conduit |
| `epdm_hose` | Automotive EPDM coolant/vacuum hose |
| `steel_brake` | Steel brake/fuel line |
| `aluminum_tube` | Aluminum AC line |

---

## 2. `render_scene_nurbs.py` — Assembly Scene Renderer

Renders a pre-made `.blend` assembly (e.g., Tofas) in random rooms with GT curve extraction from NURBS centerlines.

### Quick Start

```bash
# Test run (5 scenes, D435i stereo):
blenderproc run render_scene_nurbs.py \
    ./BlenderProc_Project/Segmentation/Tofas_2025-06-22.blend \
    tofas_test ./cctextures \
    --num_scenes 5 --poses_per_scene 5 --stereo_camera realsense_d435i

# Full dataset (500 scenes, D435i stereo):
blenderproc run render_scene_nurbs.py \
    ./BlenderProc_Project/Segmentation/Tofas_2025-06-22.blend \
    tofas_dataset ./cctextures \
    --num_scenes 500 --poses_per_scene 10 --stereo_camera realsense_d435i

# ZED2 stereo:
blenderproc run render_scene_nurbs.py \
    ./BlenderProc_Project/Segmentation/Tofas_2025-06-22.blend \
    tofas_zed2 ./cctextures \
    --num_scenes 500 --poses_per_scene 10 --stereo_camera zed2

# Monocular (no stereo):
blenderproc run render_scene_nurbs.py \
    ./BlenderProc_Project/Segmentation/Tofas_2025-06-22.blend \
    tofas_mono ./cctextures \
    --num_scenes 100 --poses_per_scene 10 --stereo_camera none

# Disable curve GT (RGB/depth/segmentation only):
blenderproc run render_scene_nurbs.py \
    ./BlenderProc_Project/Segmentation/Tofas_2025-06-22.blend \
    tofas_no_gt ./cctextures \
    --num_scenes 100 --poses_per_scene 10 --disable_curve_gt
```

### All CLI Arguments

```
Positional:
  scene                     Path to .blend file with interior objects
  output_dir                Output directory (default: tofas_output_COCO)
  cc_material_path          Path to CCTextures folder

Curve GT:
  --curve_collection NAME   Blender collection with GT curves (default: Extracted_Centerlines)
  --curve_gt_points N       Points sampled per curve (default: 64)
  --disable_curve_gt        Skip curve GT export

Scene generation:
  --num_scenes N            Number of unique scenes (default: 50)
  --poses_per_scene N       Camera poses per scene (default: 10)

Stereo camera:
  --stereo_camera PRESET    zed2 | realsense_d435 | realsense_d435i | oak_d | custom | none
                            (default: realsense_d435i)
  --stereo_baseline F       Override preset baseline (meters)
  --image_width N           Render width (default: from preset)
  --image_height N          Render height (default: from preset)

Depth noise (off by default):
  --add_depth_noise         Enable depth noise simulation
  --depth_noise_sigma F     Noise std-dev in meters (default: 0.002)

Rendering quality:
  --render_samples N        Cycles samples (default: 24)
  --render_noise_threshold F  Adaptive sampling threshold (default: 0.005)

Output:
  --write_coco              Write COCO annotations (off by default)
```

### Output Structure

```
output/
├── 0/                             # Scene index
│   ├── curve_gt.json              # 3D curve ground truth (all frames)
│   ├── 0.hdf5                     # Frame 0: RGB, depth, segmaps, normals, curve GT
│   ├── 1.hdf5                     # Frame 1
│   └── ...
├── 1/
│   └── ...
└── coco_data/                     # Only if --write_coco
    └── ...
```

HDF5 keys (per frame):
- `colors` — Left RGB (H×W×3 uint8)
- `colors_right` — Right RGB (stereo only)
- `depth` — Left depth (H×W float32, meters)
- `depth_right` — Right depth (stereo only)
- `depth_noisy` — Noisy depth (if `--add_depth_noise`)
- `normals` — Surface normals (H×W×3 float32)
- `normals_right` — Right normals (stereo only)
- `instance_segmaps` — Instance segmentation (H×W int32)
- `category_id_segmaps` — Category segmentation (H×W int32)
- `curve_gt_points_world__<ID>` — 3D world coords per curve (N×3 float32)
- `curve_gt_points_cam__<ID>` — 3D camera coords per curve (F×N×3 float32)
- `curve_gt_points_uv__<ID>` — 2D projections per curve (F×N×2 float32)
- `curve_gt_visibility__<ID>` — Per-node visibility (F×N uint8)
- `curve_gt_world_to_cam` — Camera extrinsics (F×4×4 float32)

---

## 3. Camera Presets

Both scripts share the same stereo camera presets:

| Preset | Baseline | fx | fy | cx | cy | Resolution |
|---|---|---|---|---|---|---|
| `zed2` | 120mm | 527.0 | 527.0 | 640.0 | 360.0 | 1280×720 |
| `realsense_d435` | 50mm | 605.2 | 605.1 | 425.7 | 246.0 | 1280×720 |
| `realsense_d435i` | 50mm | 605.2 | 605.1 | 425.7 | 246.0 | 1280×720 |
| `oak_d` | 75mm | 570.0 | 570.0 | 640.0 | 360.0 | 1280×720 |

Use `--stereo_camera none` for monocular rendering.

---

## 4. `curve_gt.json` Format

The primary ground truth format for 3D curve extraction training.

```json
{
  "frame_index": 0,
  "scene_index": 0,
  "image_size": {"width": 1280, "height": 720},
  "camera_intrinsics": [[605.2, 0, 425.7], [0, 605.1, 246.0], [0, 0, 1]],
  "world_to_cam": [[...], [...], [...], [...]],
  "cam_to_world": [[...], [...], [...], [...]],
  "stereo": true,
  "stereo_baseline_m": 0.05,
  "curves": [
    {
      "curve_id": "DLO_0000_00",
      "estimated_radius": 0.008,
      "num_nodes": 64,
      "points_world": [[x,y,z], ...],
      "points_cam": [[x,y,z], ...],
      "points_uv": [[u,v], ...],
      "node_depths": [z0, z1, ...],
      "visibility_projection": [1, 1, 0, ...],
      "visibility_raycast": [1, 0, 0, ...],
      "visibility_combined": [1, 0, 0, ...],
      "inter_node_distances": [d01, d12, ...],
      "total_arc_length": 1.23
    }
  ]
}
```

| Field | Description |
|---|---|
| `points_world` | 3D skeleton in world frame (N×3) |
| `points_cam` | 3D skeleton in camera frame (N×3) |
| `points_uv` | 2D projection onto image (N×2, pixels) |
| `node_depths` | Depth of each node (N, meters) |
| `visibility_projection` | 1 = in frame, 0 = outside FOV |
| `visibility_raycast` | 1 = unoccluded, 0 = occluded by geometry |
| `visibility_combined` | 1 = visible (in frame AND unoccluded) |
| `inter_node_distances` | Distances between consecutive nodes |
| `total_arc_length` | Total curve length in meters |

---

## 5. Recommended Training Pipeline

```
┌─────────────────────────────────┐
│  render_dlo_dataset.py          │  ← Large-scale procedural diversity
│  50k+ images, varied shapes    │
└──────────────┬──────────────────┘
               │ Pre-train
               ▼
      ┌────────────────┐
      │  3D Curve Model │  (GNN / Transformer / etc.)
      └────────┬───────┘
               │ Fine-tune
               ▼
┌─────────────────────────────────┐
│  render_scene_nurbs.py          │  ← Domain-specific assembly data
│  5k images, real assembly GT   │
└─────────────────────────────────┘
```

> **Note on noise**: Sensor noise (depth noise, RGB noise, blur) is better applied
> as training-time augmentation in your PyTorch/TF DataLoader rather than baked
> into the rendered data. Both scripts have `--add_depth_noise` available but
> disabled by default for this reason.
