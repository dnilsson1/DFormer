import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import sys
from importlib import import_module

# Add root dir to path to allow imports from models and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.builder import EncoderDecoder as segmodel
from utils.transforms import normalize

def get_class_colors():
    def uint82bin(n, count=8):
        return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
    N = 41
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap.tolist()

def preprocess_image(rgb_path, depth_path, config):
    # Load RGB (BGR mode as per training)
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise FileNotFoundError(f"RGB image not found at {rgb_path}")
    
    # Load Depth
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found at {depth_path}")

    # Resize to config dimensions
    h, w = config.image_height, config.image_width
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    # Handle depth channels (replicate if single channel)
    # Since we loaded as grayscale, it is 2D (H, W)
    if len(depth.shape) == 2:
        depth = cv2.merge([depth, depth, depth])
    
    # Normalize RGB (using config mean/std)
    rgb = normalize(rgb, config.norm_mean, config.norm_std)
    
    # Normalize Depth (using hardcoded values from ValPre)
    depth = normalize(depth, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])
    
    # Transpose to CHW
    rgb = rgb.transpose(2, 0, 1)
    depth = depth.transpose(2, 0, 1)
    
    return torch.from_numpy(rgb).float().unsqueeze(0), torch.from_numpy(depth).float().unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Single Image Inference")
    parser.add_argument("-cfg", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("-r", "--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("-d", "--depth", type=str, required=True, help="Path to Depth image")
    parser.add_argument("-o", "--output", type=str, default="output.png", help="Path to save output image")
    args = parser.parse_args()

    # Load config
    sys.path.insert(0, os.path.dirname(args.config))
    config_name = os.path.basename(args.config).split('.')[0]
    config_module = import_module(config_name)
    config = config_module.config

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    network = nn.DataParallel(network)
    network.to(device)

    # Load checkpoint in a robust way to handle saved dicts with/without 'module.' prefix
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Get the state dict whether it's saved as 'model' or 'state_dict' or directly as dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Normalize keys so they match the DataParallel wrapper of the model
    first_key = next(iter(state_dict.keys()))
    if isinstance(network, nn.DataParallel):
        # Model expects keys prefixed with 'module.'
        if not first_key.startswith("module."):
            state_dict = {"module." + k: v for k, v in state_dict.items()}
    else:
        if first_key.startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load with strict=False to allow compatible mismatches (like missing metadata)
    network.load_state_dict(state_dict, strict=False)
    network.eval()

    # Preprocess
    print(f"Processing {args.rgb} and {args.depth}")
    rgb, depth = preprocess_image(args.rgb, args.depth, config)
    rgb = rgb.to(device)
    depth = depth.to(device)

    # Inference
    with torch.no_grad():
        output = network(rgb, depth)
        # Output is usually (B, C, H, W)
        # We need argmax
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Load the original depth to create a foreground mask
    # Read as UNCHANGED to preserve original bit depth
    raw_depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    raw_depth = cv2.resize(raw_depth, (config.image_width, config.image_height), interpolation=cv2.INTER_NEAREST)
    
    # Determine depth format and create foreground mask
    # 16-bit depth (RealSense raw): values in mm, 0 = no reading
    # 8-bit depth (normalized): 0 = close, 255 = far, use threshold
    if raw_depth.dtype == np.uint16:
        # 16-bit: mask out pixels with no depth reading (depth == 0)
        foreground_mask = raw_depth > 0
        print(f"16-bit depth: Valid pixels: {foreground_mask.sum()}, No-depth pixels: {(~foreground_mask).sum()}")
    else:
        # 8-bit normalized: foreground is typically darker (closer), background is lighter (farther)
        # Use a threshold - pixels with depth > threshold are considered background
        bg_threshold = 180  # Adjust based on your depth normalization
        foreground_mask = raw_depth < bg_threshold
        print(f"8-bit depth: Foreground pixels (<{bg_threshold}): {foreground_mask.sum()}, Background pixels: {(~foreground_mask).sum()}")

    # Visualize
    colors = get_class_colors()
    # Create colored mask
    h, w = pred.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(len(colors)):
        color_mask[pred == i] = colors[i]
    
    # Load original RGB for overlay
    orig_rgb = cv2.imread(args.rgb)
    orig_rgb = cv2.resize(orig_rgb, (config.image_width, config.image_height))
    
    # Create overlay: blend only on foreground, show original RGB on background
    overlay = orig_rgb.copy()
    # Apply colored segmentation only where foreground_mask is True
    overlay[foreground_mask] = cv2.addWeighted(
        orig_rgb[foreground_mask].reshape(-1, 3), 0.5,
        color_mask[foreground_mask].reshape(-1, 3), 0.5,
        0
    ).reshape(-1, 3)
    
    # Save result
    cv2.imwrite(args.output, color_mask)
    overlay_path = args.output.replace(".png", "_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"Saved result to {args.output} and overlay to {overlay_path}")

if __name__ == "__main__":
    main()
