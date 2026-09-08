"""
Dataset for loading procedural DLO renders from render_dlo_dataset.py.

Expected directory structure:
    dlo_dataset/
    ├── scene_0000/
    │   ├── frame_000.hdf5      (RGB, depth, normals, segmentation)
    │   └── curve_gt.json       (3D skeleton ground truth)
    ├── scene_0001/
    │   ├── frame_000.hdf5
    │   └── curve_gt.json
    └── ...

Also supports rendered frame directories:
    dlo_dataset/
    ├── scene_0000/frame_0000/
    │   ├── rgb_left.png
    │   ├── depth_clean.npz
    │   ├── normals.npz
    │   └── curve_gt.json
    └── ...
"""

import os
import json
import glob
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class DLODataset(Dataset):
    """
    PyTorch Dataset for procedural DLO renders.

    Loads RGBD data + 3D curve ground truth from render_dlo_dataset.py output.
    """

    def __init__(
        self,
        root_dir: str,
        num_nodes: int = 64,
        max_curves: int = 20,
        image_height: int = 720,
        image_width: int = 1280,
        max_depth: float = 3.0,
        transform=None,
        split_file: str | None = None,
        use_noisy_depth: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.num_nodes = num_nodes
        self.max_curves = max_curves
        self.image_height = image_height
        self.image_width = image_width
        self.max_depth = max_depth
        self.transform = transform
        self.split_file = split_file
        self.use_noisy_depth = use_noisy_depth

        # Find all scene directories
        self.samples = self._discover_samples()
        print(f"DLODataset: found {len(self.samples)} samples in {root_dir}")

    def _discover_samples(self) -> list[dict]:
        """Discover legacy HDF5 samples and rendered frame-directory samples."""
        samples = []
        scene_dirs = sorted(glob.glob(os.path.join(self.root_dir, "scene_*")))
        if self.split_file:
            with open(self.split_file, "r") as f:
                split_scenes = set(json.load(f)["scenes"])
            scene_dirs = [
                scene_dir for scene_dir in scene_dirs
                if Path(scene_dir).name in split_scenes
            ]

        for scene_dir in scene_dirs:
            gt_path = os.path.join(scene_dir, "curve_gt.json")
            if os.path.exists(gt_path):
                hdf5_files = sorted(glob.glob(os.path.join(scene_dir, "frame_*.hdf5")))
                if not hdf5_files:
                    hdf5_files = sorted(glob.glob(os.path.join(scene_dir, "*.hdf5")))

                for hdf5_path in hdf5_files:
                    samples.append({
                        "format": "hdf5",
                        "hdf5_path": hdf5_path,
                        "gt_path": gt_path,
                        "scene_dir": scene_dir,
                        "frame_idx": self._extract_frame_idx(hdf5_path),
                    })

            frame_dirs = sorted(glob.glob(os.path.join(scene_dir, "frame_*")))
            for frame_dir in frame_dirs:
                frame_gt_path = os.path.join(frame_dir, "curve_gt.json")
                rgb_path = os.path.join(frame_dir, "rgb_left.png")
                depth_path = os.path.join(frame_dir, "depth_clean.npz")
                depth_noisy_path = os.path.join(frame_dir, "depth_noisy.npz")
                normals_path = os.path.join(frame_dir, "normals.npz")
                if all(os.path.exists(path) for path in
                       (frame_gt_path, rgb_path, depth_path, normals_path)):
                    samples.append({
                        "format": "frame_dir",
                        "gt_path": frame_gt_path,
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "depth_noisy_path": depth_noisy_path if os.path.exists(depth_noisy_path) else None,
                        "normals_path": normals_path,
                        "scene_dir": scene_dir,
                        "frame_idx": self._extract_frame_idx(frame_dir),
                    })

        return samples

    @staticmethod
    def _extract_frame_idx(path: str) -> int:
        """Extract frame index from filename like 'frame_000.hdf5'."""
        basename = os.path.splitext(os.path.basename(path))[0]
        parts = basename.split("_")
        for p in reversed(parts):
            try:
                return int(p)
            except ValueError:
                continue
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        if sample["format"] == "hdf5":
            rgb, depth, normals = self._load_hdf5(sample["hdf5_path"])
        else:
            use_noisy = self.use_noisy_depth and (random.random() > 0.5)
            rgb, depth, normals = self._load_frame_dir(sample, use_noisy=use_noisy)

        # Load curve ground truth
        with open(sample["gt_path"], "r") as f:
            gt_data = json.load(f)

        curves = gt_data.get("curves", [])

        # Camera intrinsics matrix
        intrinsics = gt_data.get("camera_intrinsics")
        if intrinsics is not None:
            K_matrix = np.array(intrinsics, dtype=np.float32)
        else:
            K_matrix = np.array([
                [605.2, 0.0, float(self.image_width) / 2.0],
                [0.0, 605.1, float(self.image_height) / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)

        # Parse curve data
        gt_coords = []
        gt_visibility = []
        gt_radius = []

        for curve in curves:
            pts_3d = self._camera_points(curve, gt_data)
            r = float(curve.get("radius", curve.get("estimated_radius", 0.005)))

            # Resample to fixed number of nodes
            pts_3d = self._resample_curve(pts_3d, self.num_nodes)

            # Node-level frustum visibility
            frustum_vis = self._compute_frustum_visibility(pts_3d, K_matrix)

            # Combine with any existing visibility annotation (e.g. occlusion)
            if "visibility_combined" in curve or "visibility" in curve:
                raw_vis = np.array(
                    curve.get("visibility_combined", curve.get("visibility")),
                    dtype=np.float32,
                )
                vis = self._resample_1d(raw_vis, self.num_nodes) * frustum_vis
            else:
                vis = frustum_vis

            # Discard curves that are completely or mostly outside the camera frustum
            # (requires at least 2 visible nodes and at least 10% visible length)
            if vis.sum() < 2.0 or (vis.mean() < 0.1):
                continue

            gt_coords.append(pts_3d)
            gt_visibility.append(vis)
            gt_radius.append(r)

        # Pad to max_curves
        num_curves = min(len(gt_coords), self.max_curves)
        coords_padded = np.zeros((self.max_curves, self.num_nodes, 3),
                                 dtype=np.float32)
        vis_padded = np.zeros((self.max_curves, self.num_nodes),
                              dtype=np.float32)
        radius_padded = np.zeros(self.max_curves, dtype=np.float32)

        for i in range(num_curves):
            coords_padded[i] = gt_coords[i]
            vis_padded[i] = gt_visibility[i]
            radius_padded[i] = gt_radius[i]

        # Normalize depth
        depth = np.clip(depth, 0, self.max_depth) / self.max_depth

        # Convert to tensors
        # RGB: (H, W, 3) → (3, H, W)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        # Normals: (H, W, 3) → (3, H, W)
        normals_tensor = torch.from_numpy(normals).permute(2, 0, 1).contiguous()

        result = {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "normals": normals_tensor,
            "coords": torch.from_numpy(coords_padded),
            "visibility": torch.from_numpy(vis_padded),
            "radius": torch.from_numpy(radius_padded),
            "num_curves": torch.tensor(num_curves, dtype=torch.long),
            "scene_dir": sample["scene_dir"],
            "frame_idx": sample["frame_idx"],
        }

        if self.transform is not None:
            result = self.transform(result)

        return result

    def _load_hdf5(self, hdf5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load RGB, depth, and normals from a legacy HDF5 sample."""
        with h5py.File(hdf5_path, "r") as f:
            rgb = np.array(f["colors"], dtype=np.float32) if "colors" in f else np.zeros(
                (self.image_height, self.image_width, 3), dtype=np.float32)
            depth = np.array(f["depth"], dtype=np.float32) if "depth" in f else np.zeros(
                (self.image_height, self.image_width), dtype=np.float32)
            normals = np.array(f["normals"], dtype=np.float32) if "normals" in f else np.zeros(
                (self.image_height, self.image_width, 3), dtype=np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0
        return rgb, depth, normals

    @staticmethod
    def _load_frame_dir(sample: dict, use_noisy: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load RGB, depth, and normals from a rendered frame directory."""
        rgb = np.asarray(Image.open(sample["rgb_path"]).convert("RGB"), dtype=np.float32) / 255.0
        depth_target = sample["depth_path"]
        if use_noisy and sample.get("depth_noisy_path"):
            depth_target = sample["depth_noisy_path"]
        with np.load(depth_target) as depth_file:
            depth = np.array(depth_file["depth"], dtype=np.float32)
        with np.load(sample["normals_path"]) as normals_file:
            normals = np.array(normals_file["normals"], dtype=np.float32)
        return rgb, depth, normals

    def _compute_frustum_visibility(self, pts_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Compute node-level camera frustum visibility (N,) in [0.0, 1.0]."""
        zc = pts_3d[:, 2]
        xc = pts_3d[:, 0]
        yc = pts_3d[:, 1]

        valid_z = (zc > 0.05) & (zc <= self.max_depth)
        safe_z = np.where(valid_z, zc, 1.0)

        u = K[0, 0] * (xc / safe_z) + K[0, 2]
        v = K[1, 1] * (yc / safe_z) + K[1, 2]

        in_frustum = (
            valid_z
            & (u >= 0)
            & (u < self.image_width)
            & (v >= 0)
            & (v < self.image_height)
        )
        return in_frustum.astype(np.float32)

    @staticmethod
    def _camera_points(curve: dict, frame_gt: dict) -> np.ndarray:
        """Return curve points in the camera coordinate system."""
        if "points_3d" in curve:
            return np.array(curve["points_3d"], dtype=np.float32)
        if "points_cam" in curve:
            return np.array(curve["points_cam"], dtype=np.float32)
        points_world = np.array(curve["points_world"], dtype=np.float32)
        world_to_cam = np.array(frame_gt["world_to_cam"], dtype=np.float32)
        points_world_h = np.concatenate(
            [points_world, np.ones((len(points_world), 1), dtype=np.float32)], axis=1
        )
        return (world_to_cam @ points_world_h.T).T[:, :3]

    @staticmethod
    def _resample_curve(points: np.ndarray, target_n: int) -> np.ndarray:
        """Resample a 3D curve to exactly target_n evenly-spaced points."""
        if len(points) == target_n:
            return points
        if len(points) < 2:
            return np.tile(points[0] if len(points) == 1
                          else np.zeros(3), (target_n, 1))

        # Compute cumulative arc length
        diffs = np.diff(points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
        total_length = cum_length[-1]

        if total_length < 1e-8:
            return np.tile(points[0], (target_n, 1))

        # Interpolate at evenly-spaced arc lengths
        target_lengths = np.linspace(0, total_length, target_n)
        resampled = np.zeros((target_n, 3), dtype=np.float32)
        for dim in range(3):
            resampled[:, dim] = np.interp(target_lengths, cum_length, points[:, dim])

        return resampled

    @staticmethod
    def _resample_1d(values: np.ndarray, target_n: int) -> np.ndarray:
        """Resample a 1D array to target_n points via linear interpolation."""
        if len(values) == target_n:
            return values
        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, target_n)
        return np.interp(x_new, x_old, values).astype(np.float32)
