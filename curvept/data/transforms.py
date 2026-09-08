"""
Data augmentation transforms for CurvePT training.

All transforms operate on the full sample dict and modify both images
AND ground truth coordinates consistently.
"""

import torch
import numpy as np
import random
import torchvision.transforms.functional as TF


class ColorJitter:
    """Random color jitter on RGB image (does not affect depth/normals/GT)."""

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05):
        self.jitter = torch.nn.Identity()  # Placeholder
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample: dict) -> dict:
        rgb = sample["rgb"]
        rgb = TF.adjust_brightness(rgb, 1.0 + random.uniform(-self.brightness,
                                                                   self.brightness))
        rgb = TF.adjust_contrast(rgb, 1.0 + random.uniform(-self.contrast,
                                                                 self.contrast))
        rgb = TF.adjust_saturation(rgb, 1.0 + random.uniform(-self.saturation,
                                                                   self.saturation))
        rgb = TF.adjust_hue(rgb, random.uniform(-self.hue, self.hue))
        sample["rgb"] = rgb.clamp(0, 1)
        return sample


class DepthNoise:
    """Add Gaussian noise to depth map (simulates sensor noise)."""

    def __init__(self, sigma: float = 0.002):
        self.sigma = sigma

    def __call__(self, sample: dict) -> dict:
        depth = sample["depth"]
        valid = depth > 0
        noise = torch.randn_like(depth) * self.sigma
        depth = depth + noise * valid.float()
        sample["depth"] = depth.clamp(0)
        return sample


class RandomDepthDropout:
    """
    Randomly drops out depth patches (setting depth to 0).
    Simulates real-world IR stereo artifacts and specular dropouts on thin wires.
    """

    def __init__(self, p: float = 0.5, max_patches: int = 5, max_size: int = 80):
        self.p = p
        self.max_patches = max_patches
        self.max_size = max_size

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        depth = sample["depth"]
        _, H, W = depth.shape
        num_patches = random.randint(1, self.max_patches)
        for _ in range(num_patches):
            h = random.randint(20, min(self.max_size, H) - 1)
            w = random.randint(20, min(self.max_size, W) - 1)
            y = random.randint(0, max(1, H - h) - 1)
            x = random.randint(0, max(1, W - w) - 1)
            depth[:, y:y+h, x:x+w] = 0.0
        sample["depth"] = depth
        return sample


class RandomCableCutout:
    """
    Randomly masks small patches of RGB and Depth.
    Simulates partial cable occlusions by hands, tools, or other obstacles.
    """

    def __init__(self, p: float = 0.3, max_size: int = 60):
        self.p = p
        self.max_size = max_size

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        rgb = sample["rgb"]
        depth = sample["depth"]
        _, H, W = rgb.shape
        h = random.randint(20, min(self.max_size, H) - 1)
        w = random.randint(20, min(self.max_size, W) - 1)
        y = random.randint(0, max(1, H - h) - 1)
        x = random.randint(0, max(1, W - w) - 1)

        rgb[:, y:y+h, x:x+w] = 0.0
        depth[:, y:y+h, x:x+w] = 0.0
        sample["rgb"] = rgb
        sample["depth"] = depth
        return sample


class RandomHorizontalFlip:
    """
    Random horizontal flip with consistent transformation of GT coordinates.

    Flips: RGB, depth, normals images AND mirrors 3D X-coordinates.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        # Flip images
        sample["rgb"] = TF.hflip(sample["rgb"])
        sample["depth"] = TF.hflip(sample["depth"])
        sample["normals"] = TF.hflip(sample["normals"])
        # Flip normal X component
        sample["normals"][0] = -sample["normals"][0]

        # Mirror 3D coordinates (flip X axis)
        # This requires knowing the camera principal point, but for centered
        # cameras, flipping x-coord of 3D points is approximately correct
        # For production, use proper projection → flip → unproject
        sample["coords"][:, :, 0] = -sample["coords"][:, :, 0]

        return sample


class NormalizeRGB:
    """Normalize RGB to [-1, 1] range (some backbones prefer this)."""

    def __call__(self, sample: dict) -> dict:
        sample["rgb"] = sample["rgb"] * 2.0 - 1.0
        return sample


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_train_transforms(config: dict) -> Compose:
    """Build training augmentation pipeline from config."""
    dc = config["data"]
    transforms = [
        ColorJitter(brightness=dc.get("color_jitter", 0.3),
                    contrast=dc.get("color_jitter", 0.3),
                    saturation=dc.get("color_jitter", 0.3) * 0.5),
    ]
    if dc.get("depth_noise_sigma", 0) > 0:
        transforms.append(DepthNoise(sigma=dc["depth_noise_sigma"]))
    if dc.get("depth_dropout", True):
        transforms.append(RandomDepthDropout(p=0.5))
    if dc.get("cable_cutout", False):
        transforms.append(RandomCableCutout(p=0.3))
    if dc.get("horizontal_flip", True):
        transforms.append(RandomHorizontalFlip(p=0.5))

    return Compose(transforms)


def build_val_transforms(config: dict) -> Compose:
    """Build validation transforms (minimal, no augmentation)."""
    return Compose([])
