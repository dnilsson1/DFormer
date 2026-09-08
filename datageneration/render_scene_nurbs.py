import blenderproc as bproc
import argparse
import numpy as np
import os
import bpy
import random
import json
import re
from mathutils import Vector

# 1. SETUP AND CONFIGURATION
parser = argparse.ArgumentParser(
    description="Assembly Scene Renderer with NURBS GT for DLO Dataset"
)
parser.add_argument('scene', nargs='?',
                    default="./BlenderProc_Project/Segmentation/Tofas_2025-06-22.blend",
                    help="Path to the blend file with interior objects.")
parser.add_argument('output_dir', nargs='?', default="tofas_output_COCO",
                    help="Path to where the final files will be saved")
parser.add_argument('cc_material_path', nargs='?',
                    default="./BlenderProc_Project/Segmentation/cctextures",
                    help="Path to CCTextures folder.")
parser.add_argument('--curve_collection', default="Extracted_Centerlines",
                    help="Collection name that contains curve GT objects.")
parser.add_argument('--curve_gt_points', type=int, default=64,
                    help="Number of points sampled per curve for GT export.")
parser.add_argument('--disable_curve_gt', action='store_true',
                    help="Disable curve GT export if you only want RGB/depth/segmentation.")

# Scene generation
parser.add_argument('--num_scenes', type=int, default=50,
                    help="Number of unique scenes to generate.")
parser.add_argument('--poses_per_scene', type=int, default=10,
                    help="Number of camera poses per scene.")

# Stereo camera settings
parser.add_argument('--stereo_camera', default="realsense_d435i",
                    choices=["zed2", "realsense_d435", "realsense_d435i", "oak_d", "custom", "none"],
                    help="Stereo camera preset. 'none' for monocular only.")
parser.add_argument('--stereo_baseline', type=float, default=None,
                    help="Custom stereo baseline in meters (overrides preset).")
parser.add_argument('--image_width', type=int, default=1280,
                    help="Render width in pixels.")
parser.add_argument('--image_height', type=int, default=720,
                    help="Render height in pixels.")

# Depth noise
parser.add_argument('--add_depth_noise', action='store_true', default=False,
                    help="Generate noisy depth alongside clean depth (prefer training-time augmentation).")
parser.add_argument('--depth_noise_sigma', type=float, default=0.002,
                    help="Std-dev of Gaussian depth noise in meters.")

# Rendering quality
parser.add_argument('--render_samples', type=int, default=24,
                    help="Number of render samples (higher = less noise, slower).")
parser.add_argument('--render_noise_threshold', type=float, default=0.005,
                    help="Adaptive sampling noise threshold.")
parser.add_argument('--write_coco', action='store_true', default=False,
                    help="Write COCO annotations (optional, for 2D detection baselines).")

args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# STEREO CAMERA PRESETS
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


cam_config = get_camera_config(args)
NUM_UNIQUE_SCENES = args.num_scenes
POSES_PER_SCENE = args.poses_per_scene
IMAGE_WIDTH = cam_config["width"]
IMAGE_HEIGHT = cam_config["height"]
K_MATRIX = cam_config["K"]
STEREO = cam_config["stereo"]
BASELINE = cam_config["baseline_m"]


def sanitize_key(name):
    sanitized = re.sub(r'[^0-9a-zA-Z_]+', '_', name).strip('_')
    return sanitized or "curve"


def resample_polyline(points, n_points):
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


def find_curve_objects(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection is not None:
        curves = [obj for obj in collection.objects if obj.type == 'CURVE']
    else:
        curves = [
            obj for obj in bpy.data.objects
            if obj.type == 'CURVE' and ("source_mesh" in obj or obj.name.startswith("CL_"))
        ]
    curves.sort(key=lambda obj: obj.name)
    return curves


def extract_curve_entries(curve_objects, sample_count):
    sample_count = max(2, int(sample_count))
    entries = []
    id_counts = {}

    for curve_obj in curve_objects:
        source_mesh = str(curve_obj.get("source_mesh", ""))
        estimated_radius = curve_obj.get("estimated_radius", 0.0)
        try:
            estimated_radius = float(estimated_radius)
        except (TypeError, ValueError):
            estimated_radius = 0.0

        for spline_idx, spline in enumerate(curve_obj.data.splines):
            if spline.type == 'BEZIER':
                local_points = np.array(
                    [[p.co.x, p.co.y, p.co.z] for p in spline.bezier_points],
                    dtype=np.float32
                )
            else:
                local_points = np.array(
                    [[p.co.x, p.co.y, p.co.z] for p in spline.points],
                    dtype=np.float32
                )

            if local_points.shape[0] < 2:
                continue

            sampled_local = resample_polyline(local_points, sample_count)
            world_points = np.zeros_like(sampled_local)
            for idx, point in enumerate(sampled_local):
                world_vec = curve_obj.matrix_world @ Vector((float(point[0]), float(point[1]), float(point[2])))
                world_points[idx] = [world_vec.x, world_vec.y, world_vec.z]

            base_id = sanitize_key(f"{curve_obj.name}_s{spline_idx}")
            count = id_counts.get(base_id, 0)
            id_counts[base_id] = count + 1
            curve_id = base_id if count == 0 else f"{base_id}_{count}"

            entries.append({
                "curve_id": curve_id,
                "curve_name": curve_obj.name,
                "spline_index": int(spline_idx),
                "source_mesh": source_mesh,
                "estimated_radius": estimated_radius,
                "points_world": world_points.astype(np.float32),
            })

    return entries


def project_points_to_image(points_cam, k_matrix, image_width, image_height):
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


def build_curve_ground_truth(curve_entries, poses_added, k_matrix, image_width, image_height, scene_index):
    if poses_added <= 0 or not curve_entries:
        return None, {}

    gt_payload = {
        "schema_version": 1,
        "scene_index": int(scene_index),
        "image_size": {"width": int(image_width), "height": int(image_height)},
        "camera_intrinsics": k_matrix.tolist(),
        "curves": [],
        "frames": []
    }

    hdf5_payload = {
        "curve_gt_camera_K": k_matrix.astype(np.float32)
    }

    for entry in curve_entries:
        gt_payload["curves"].append({
            "curve_id": entry["curve_id"],
            "curve_name": entry["curve_name"],
            "spline_index": int(entry["spline_index"]),
            "source_mesh": entry["source_mesh"],
            "estimated_radius": float(entry["estimated_radius"]),
            "points_world": entry["points_world"].tolist()
        })
        hdf5_payload[f"curve_gt_points_world__{entry['curve_id']}"] = entry["points_world"].astype(np.float32)

    world_to_cam_frames = []
    cam_to_world_frames = []
    per_curve_series = {
        entry["curve_id"]: {"cam": [], "uv": [], "visibility": []}
        for entry in curve_entries
    }

    for frame_idx in range(poses_added):
        world_to_cam = np.array(bproc.camera.get_world_to_cam_view_matrix(frame_idx), dtype=np.float32)
        cam_to_world = np.linalg.inv(world_to_cam).astype(np.float32)

        world_to_cam_frames.append(world_to_cam)
        cam_to_world_frames.append(cam_to_world)

        frame_data = {
            "frame_index": int(frame_idx),
            "world_to_cam": world_to_cam.tolist(),
            "cam_to_world": cam_to_world.tolist(),
            "curve_observations": []
        }

        for entry in curve_entries:
            points_world = entry["points_world"]
            points_world_h = np.concatenate(
                [points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)],
                axis=1
            )
            points_cam_h = (world_to_cam @ points_world_h.T).T
            points_cam = points_cam_h[:, :3].astype(np.float32)

            points_uv, visibility = project_points_to_image(points_cam, k_matrix, image_width, image_height)

            per_curve_series[entry["curve_id"]]["cam"].append(points_cam)
            per_curve_series[entry["curve_id"]]["uv"].append(points_uv)
            per_curve_series[entry["curve_id"]]["visibility"].append(visibility)

            frame_data["curve_observations"].append({
                "curve_id": entry["curve_id"],
                "points_cam": points_cam.tolist(),
                "points_uv": points_uv.tolist(),
                "visibility": visibility.tolist()
            })

        gt_payload["frames"].append(frame_data)

    hdf5_payload["curve_gt_world_to_cam"] = np.stack(world_to_cam_frames, axis=0).astype(np.float32)
    hdf5_payload["curve_gt_cam_to_world"] = np.stack(cam_to_world_frames, axis=0).astype(np.float32)

    for curve_id, series in per_curve_series.items():
        hdf5_payload[f"curve_gt_points_cam__{curve_id}"] = np.stack(series["cam"], axis=0).astype(np.float32)
        hdf5_payload[f"curve_gt_points_uv__{curve_id}"] = np.stack(series["uv"], axis=0).astype(np.float32)
        hdf5_payload[f"curve_gt_visibility__{curve_id}"] = np.stack(series["visibility"], axis=0).astype(np.uint8)

    return gt_payload, hdf5_payload


def write_curve_ground_truth_json(scene_output_dir, gt_payload):
    if gt_payload is None:
        return None

    out_path = os.path.join(scene_output_dir, "curve_gt.json")
    with open(out_path, "w", encoding="utf-8") as gt_file:
        json.dump(gt_payload, gt_file, indent=2)
    return out_path

def compute_stereo_cam2world(left_cam2world, baseline_m):
    """Right camera = left camera offset along local X by baseline."""
    right = left_cam2world.copy()
    right[:3, 3] += left_cam2world[:3, 0] * baseline_m
    return right


def simulate_depth_noise(depth_clean, sigma=0.002):
    """
    Simulate realistic depth sensor noise (disabled by default).
    Gaussian noise proportional to depth² + 12-bit quantization.
    """
    depth_noisy = depth_clean.copy().astype(np.float32)
    valid = depth_noisy > 0
    if not np.any(valid):
        return depth_noisy

    noise = np.random.normal(0, 1, depth_noisy.shape).astype(np.float32)
    depth_sigma = sigma * (depth_noisy ** 2)
    depth_sigma[~valid] = 0
    depth_noisy[valid] += (noise * depth_sigma)[valid]

    quant = 0.001
    depth_noisy[valid] = np.round(depth_noisy[valid] / quant) * quant

    return np.maximum(depth_noisy, 0.0)


bproc.init()

# 2. ONE-TIME SETUP
# These actions are performed only once at the start of the script.

# 2.1. Configure render settings
print(f"\n{'='*60}")
print(f"  ASSEMBLY SCENE RENDERER")
print(f"{'='*60}")
print(f"  Camera: {cam_config['preset_name']}")
print(f"  Resolution: {IMAGE_WIDTH}\u00d7{IMAGE_HEIGHT}")
print(f"  Stereo: {STEREO} (baseline={BASELINE*1000:.1f}mm)")
print(f"  Scenes: {NUM_UNIQUE_SCENES} \u00d7 {POSES_PER_SCENE} poses")
print(f"  Depth noise: {args.add_depth_noise}")
print(f"{'='*60}\n")

print("Configuring render settings...")
bproc.renderer.set_render_devices(desired_gpu_device_type=["OPTIX","CUDA"])
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
print("Depth, normals, and segmentation outputs enabled.")

# 2.2. Load materials for the rooms (can be done once)
print("Scanning for available material names...")
try:
    available_material_names = [name for name in os.listdir(args.cc_material_path) if os.path.isdir(os.path.join(args.cc_material_path, name))]
except FileNotFoundError:
    raise RuntimeError(f"The CCTextures path was not found: {args.cc_material_path}")
if not available_material_names:
    raise RuntimeError(f"No material sub-directories found in {args.cc_material_path}")
print(f"Found {len(available_material_names)} available materials to choose from.")

# 2.3. Load and process the main interior objects ONCE
print("Loading and processing interior objects...")
reloaded_interior_objects = bproc.loader.load_blend(args.scene)
interior_mesh_objects = bproc.filter.all_with_type(reloaded_interior_objects, bproc.types.MeshObject)
if not interior_mesh_objects:
    raise RuntimeError(f"No mesh objects found in {args.scene}. Aborting.")

curve_objects = []
if not args.disable_curve_gt:
    curve_objects = find_curve_objects(args.curve_collection)
    for curve_obj in curve_objects:
        curve_obj.hide_render = True
    if curve_objects:
        print(f"Found {len(curve_objects)} curve objects for GT export.")
    else:
        print("Warning: No curve objects found. Curve GT export will be skipped.")

for obj in interior_mesh_objects:
    obj.clear_parent()
    category_id = obj.get_cp('category_id', 0)
    obj.blender_obj.pass_index = int(category_id)
    obj.blender_obj.hide_render = False
print(f"Successfully loaded and processed {len(interior_mesh_objects)} objects.")


# 3. MAIN GENERATION LOOP
# This loop creates a new, unique scene in each iteration by creating a new room
# and re-placing the already-loaded interior objects.
for i in range(NUM_UNIQUE_SCENES):
    print(f"\n--- Generating Scene {i + 1} / {NUM_UNIQUE_SCENES} ---")

    # 3.1. Reset keyframes to clear camera poses from the previous run
    bproc.utility.reset_keyframes()

    # 3.2. Create and configure a new camera for this fresh scene.
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    bproc.camera.set_intrinsics_from_K_matrix(K_MATRIX.tolist(), IMAGE_WIDTH, IMAGE_HEIGHT)
    cam_obj.data.clip_start = 0.05
    cam_obj.data.clip_end = 100.0

    # 3.3. Load a new set of random materials for this specific room.
    num_materials_for_this_room = np.random.randint(2, 4)
    material_names_for_this_room = random.sample(available_material_names, num_materials_for_this_room)
    materials_for_this_room = bproc.loader.load_ccmaterials(args.cc_material_path, used_assets=material_names_for_this_room)

    # 3.4. Construct a new random room.
    print("Constructing a new random room...")
    room_objects = bproc.constructor.construct_random_room(
        used_floor_area=np.random.uniform(25, 35),
        interior_objects=[],
        materials=materials_for_this_room,
        fac_from_square_room=1.5
    )
    floor = bproc.filter.one_by_attr(room_objects, "name", "Floor")
    if not floor:
        print(f"Warning: No floor in new room. Skipping scene {i+1}.")
        bproc.object.delete_multiple(room_objects) # Clean up the failed room
        continue

    # 3.5. Place the pre-loaded interior objects into the new room with intersection checks.
    print("Finding a valid placement for objects in the new room...")
    
    assembly_bbox_points = np.concatenate([obj.get_bound_box() for obj in interior_mesh_objects])
    
    placement_successful = False
    placement_attempts = 0
    translation_vector = None

    while not placement_successful and placement_attempts < 50:
        placement_attempts += 1

        # Sample a potential location on the floor
        target_location_on_floor = bproc.sampler.upper_region(objects_to_sample_on=[floor], min_height=0.0, max_height=0.0)
        
        # FIX: Calculate the point to move based on the lowest point of the assembly
        current_assembly_center = np.mean(assembly_bbox_points, axis=0)
        lowest_z_point = np.min(assembly_bbox_points, axis=0)[2]
        point_on_object_to_move = np.array([current_assembly_center[0], current_assembly_center[1], lowest_z_point])
        
        temp_translation_vector = target_location_on_floor - point_on_object_to_move

        # Calculate the potential new bounding box of the assembly
        potential_bbox = assembly_bbox_points + temp_translation_vector
        
        # Get the floor's bounding box
        floor_bbox = floor.get_bound_box()
        
        # Check if the potential bbox is fully inside the floor's bbox (2D check is sufficient)
        if (np.min(potential_bbox, axis=0)[0] > np.min(floor_bbox, axis=0)[0] and
            np.max(potential_bbox, axis=0)[0] < np.max(floor_bbox, axis=0)[0] and
            np.min(potential_bbox, axis=0)[1] > np.min(floor_bbox, axis=0)[1] and
            np.max(potential_bbox, axis=0)[1] < np.max(floor_bbox, axis=0)[1]):
            
            print(f"Valid placement found on attempt {placement_attempts}.")
            translation_vector = temp_translation_vector
            placement_successful = True
        
    if placement_successful:
        # Apply the successful translation to all objects in the assembly
        for obj in interior_mesh_objects:
            obj.set_location(obj.get_location() + translation_vector)
        if curve_objects:
            translation_vec = Vector((
                float(translation_vector[0]),
                float(translation_vector[1]),
                float(translation_vector[2])
            ))
            for curve_obj in curve_objects:
                if curve_obj.parent is None:
                    curve_obj.location = curve_obj.location + translation_vec
    else:
        print(f"Warning: Failed to find a valid placement after {placement_attempts} attempts. Skipping scene.")
        bproc.object.delete_multiple(room_objects)
        continue

    # 3.6. Light the new scene.
    all_scene_objects = interior_mesh_objects + room_objects
    ceiling = bproc.filter.one_by_attr(room_objects, "name", "Ceiling")
    if ceiling:
        # Colors are specified as rgba where a is alpha
        # cyan_color = [0.0, 1.0, 1.0, 1.0]
        # bproc.lighting.light_surface([ceiling], emission_strength=np.random.uniform(5.0, 10.0), emission_color= cyan_color)
        
        # Define the range for with the uniform distribution centered around natural wihte light, A range 0.7-1.0 allows for noticeable tints in the lignhing.
        min_brightness = 0.7
        max_brightness = 1.0

        # Sample R, G, and B independently from this uniform range
        r = np.random.uniform(min_brightness, max_brightness)
        g = np.random.uniform(min_brightness, max_brightness)
        b = np.random.uniform(min_brightness, max_brightness)

        # Create the final RGBA color (Alpha is 1.0 for full opacity)
        sampled_white_color = [r, g, b, 1.0]

        # Use it in your bproc function
        bproc.lighting.light_surface(
            [ceiling],
            emission_strength=np.random.uniform(5.0, 10.0),
            emission_color=sampled_white_color
        )

    # 3.7. Sample multiple camera poses for the current scene configuration.
    print(f"Sampling {POSES_PER_SCENE} camera poses...")
    # The point of interest is the new center of the assembly
    point_of_interest = np.mean(np.concatenate([obj.get_bound_box() for obj in interior_mesh_objects]), axis=0)
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(all_scene_objects)

    # Get the floor's bounding box once before the loop for efficiency
    floor_bbox = floor.get_bound_box()
    floor_min_corner = np.min(floor_bbox, axis=0)
    floor_max_corner = np.max(floor_bbox, axis=0)

    poses_added = 0
    for j in range(POSES_PER_SCENE):
        # Try to find a valid pose, with a limit on attempts
        for _ in range(200): 
            camera_location = bproc.sampler.shell(
                center=point_of_interest, 
                radius_min=0.8, 
                radius_max=2.5, 
                elevation_min=10, 
                elevation_max=85
            )

            # Check if the sampled camera location is inside the room's X/Y boundaries
            is_inside_x = floor_min_corner[0] < camera_location[0] < floor_max_corner[0]
            is_inside_y = floor_min_corner[1] < camera_location[1] < floor_max_corner[1]

            if is_inside_x and is_inside_y:
                # This location is inside the room, NOW check for obstacles
                rotation_matrix = bproc.camera.rotation_from_forward_vec(point_of_interest - camera_location)
                cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)
                
                # Check for obstacles between camera and the point of interest
                if not bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 1.0}, bvh_tree):
                    bproc.camera.add_camera_pose(cam2world_matrix)
                    poses_added += 1
                    break # Success, break from the attempt loop and go to the next pose

    print(f"Successfully added {poses_added} poses for this scene.")

    # 4. RENDER & WRITE DATA (Per-Scene)
    if poses_added > 0:
        print(f"Rendering {poses_added} frames for scene {i+1} (left eye)...")
        data = bproc.renderer.render()

        # Build curve GT while left-eye keyframes are still active
        curve_gt_payload = None
        if curve_objects and not args.disable_curve_gt:
            bpy.context.view_layer.update()
            curve_entries = extract_curve_entries(curve_objects, args.curve_gt_points)
            if curve_entries:
                curve_gt_payload, curve_hdf5_payload = build_curve_ground_truth(
                    curve_entries=curve_entries,
                    poses_added=poses_added,
                    k_matrix=K_MATRIX,
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                    scene_index=i
                )
                data.update(curve_hdf5_payload)
            else:
                print("Warning: Curve objects were found, but no valid spline data was extracted.")

        # Stereo right eye rendering
        right_data = None
        if STEREO:
            print(f"Rendering {poses_added} frames for scene {i+1} (right eye)...")
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

        # Depth noise (disabled by default — prefer training-time augmentation)
        if args.add_depth_noise and "depth" in data:
            noisy_depths = []
            for fi in range(len(data["depth"])):
                noisy_depths.append(simulate_depth_noise(
                    data["depth"][fi], sigma=args.depth_noise_sigma
                ))
            data["depth_noisy"] = noisy_depths

        # Add stereo data to output
        if right_data is not None:
            if "colors" in right_data:
                data["colors_right"] = right_data["colors"]
            if "depth" in right_data:
                data["depth_right"] = right_data["depth"]
            if "normals" in right_data:
                data["normals_right"] = right_data["normals"]

        print(f"Writing data for scene {i+1}...")
        scene_output_dir = os.path.join(args.output_dir, str(i))
        os.makedirs(scene_output_dir, exist_ok=True)

        if curve_gt_payload is not None:
            curve_json_path = write_curve_ground_truth_json(scene_output_dir, curve_gt_payload)
            print(f"Curve GT saved to {curve_json_path}")

        bproc.writer.write_hdf5(scene_output_dir, data)
        print(f"Data for scene {i+1} saved to {scene_output_dir}")

        # COCO annotations (optional — for 2D detection baselines)
        if args.write_coco:
            try:
                bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                                    instance_segmaps=data["instance_segmaps"],
                                                    instance_attribute_maps=data["instance_attribute_maps"],
                                                    colors=data["colors"],
                                                    color_file_format="JPEG")
            except Exception as e:
                print(f"Warning: COCO write failed: {e}")
    else:
        print(f"Warning: No valid camera poses found for scene {i+1}. Nothing was rendered.")


    # 5. Clean up the room objects before the next iteration
    print("Cleaning up room for next scene...")
    bproc.object.delete_multiple(room_objects)


# 6. FINAL REPORT
print(f"\n--- SCRIPT FINISHED ---")
print(f"Generated {NUM_UNIQUE_SCENES} scenes.")
print(f"Output data saved in base directory: {args.output_dir}")
