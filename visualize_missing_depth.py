import cv2
import numpy as np
import os
import argparse

def visualize_missing_depth(rgb_path, depth_path, output_path):
    # Load RGB
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f"Error loading RGB: {rgb_path}")
        return

    # Load Depth (unchanged to preserve 16-bit values if applicable)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"Error loading Depth: {depth_path}")
        return

    # Resize depth to match RGB if necessary (though they seem to be from the same sensor)
    if depth.shape[:2] != rgb.shape[:2]:
        print(f"Resizing depth {depth.shape} to match RGB {rgb.shape}")
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Allow processing if depth is 3-channel (sometimes saved as such, though usually 1-channel)
    if len(depth.shape) == 3:
        depth = depth[:, :, 0] # Assume grayscale information in first channel

    # Identify missing data (0 values)
    # RealSense uses 0 for invalid depth
    missing_mask = (depth == 0)

    # Create overlay
    # We will make missing pixels Red (0, 0, 255)
    overlay = rgb.copy()
    
    # Define overlay color: Red in BGR
    color = np.array([0, 0, 255], dtype=np.uint8)
    
    # Apply color to masked area
    # We can do a blend, or a hard replacement.
    # Hard replacement makes it very visible.
    # overlay[missing_mask] = color 
    
    # Let's do a blend to maintain some context if needed, but for "missing" data, 
    # there is no depth context. But there is RGB context.
    # So a semi-transparent red is good.
    
    alpha = 0.5
    
    # Create a colored layer
    color_layer = np.zeros_like(rgb)
    color_layer[missing_mask] = color
    
    # Where mask is true, blend. Where false, keep original.
    # But we only want to blend where the mask is True.
    
    # Easiest way using cv2.addWeighted for the whole image masked
    # or just manual calculation
    
    # Pixels where mask is true: alpha * color + (1-alpha) * rgb
    roi = rgb[missing_mask]
    blended = (color * alpha + roi * (1 - alpha)).astype(np.uint8)
    overlay[missing_mask] = blended

    # Save
    cv2.imwrite(output_path, overlay)
    print(f"Saved missing depth overlay to: {output_path}")
    
    # Also calculate percentage
    total_pixels = depth.size
    missing_pixels = np.sum(missing_mask)
    print(f"Missing depth pixels: {missing_pixels} / {total_pixels} ({missing_pixels/total_pixels:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to Depth image")
    parser.add_argument("--output", required=True, help="Path to Output image")
    args = parser.parse_args()
    
    visualize_missing_depth(args.rgb, args.depth, args.output)
