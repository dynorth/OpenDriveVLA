#!/usr/bin/env python3
"""
Create visualizations with trajectory points plotted on the road image.
Fixed coordinate transformation for NuScenes camera projection.
"""

import json
import os
import pickle
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pyquaternion import Quaternion

def parse_trajectory(trajectory_str):
    """
    Parse trajectory string like '[(0.08,4.25),(0.31,8.37),...]' into list of (x,y) tuples
    """
    matches = re.findall(r'\(([-\d.]+),([-\d.]+)\)', trajectory_str)
    trajectory = [(float(x), float(y)) for x, y in matches]
    return trajectory

def project_ego_to_image(points_ego, cam_info):
    """
    Project 3D points from ego vehicle frame to camera image

    Ego frame: x=forward, y=left, z=up
    But trajectory uses: x=lateral(right+), y=forward
    Camera frame: x=right, y=down, z=forward

    Args:
        points_ego: Nx2 or Nx3 array [(x_lateral, y_forward, z_height), ...]
        cam_info: camera info dict with intrinsics and extrinsics
    Returns:
        points_img: Nx2 image coordinates, valid_mask: boolean mask
    """
    # Convert 2D trajectory to 3D (assume z=0, ground level)
    if points_ego.shape[1] == 2:
        # Trajectory format: (x_lateral, y_forward) need to convert to ego frame
        # Ego frame convention: x=forward, y=left, z=up
        # Trajectory: x=lateral_right, y=forward
        # So: ego_x = traj_y, ego_y = -traj_x, ego_z = 0
        points_ego_3d = np.zeros((len(points_ego), 3))
        points_ego_3d[:, 0] = points_ego[:, 1]  # forward = y from trajectory
        points_ego_3d[:, 1] = -points_ego[:, 0]  # left = -x from trajectory
        points_ego_3d[:, 2] = 0.0  # ground level
        points_ego = points_ego_3d

    # Get camera extrinsics
    sensor2ego_rot = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
    sensor2ego_trans = np.array(cam_info['sensor2ego_translation'])

    # Transform from ego to camera: cam = R^T @ (ego - t)
    ego2sensor_rot = sensor2ego_rot.T
    ego2sensor_trans = -ego2sensor_rot @ sensor2ego_trans

    points_cam = points_ego @ ego2sensor_rot.T + ego2sensor_trans

    # Filter points behind camera (z > 0 means in front)
    valid_mask = points_cam[:, 2] > 0.5

    # Project to image using intrinsics
    cam_intrinsic = np.array(cam_info['cam_intrinsic'])
    points_2d_hom = points_cam @ cam_intrinsic.T
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]

    return points_2d, valid_mask

def create_visualization(sample_info, result, output_path, debug=False):
    """
    Create a visualization of the camera image with predicted trajectory overlay
    """
    # Get the front camera image path and intrinsics
    cam_front = sample_info['cams']['CAM_FRONT']
    img_path = os.path.join('data/nuscenes', cam_front['data_path'])

    if not os.path.exists(img_path):
        if debug:
            print(f"Warning: Image not found: {img_path}")
        return False

    # Load image
    img = Image.open(img_path).convert('RGB')
    img_width, img_height = img.size

    # Extract trajectory from answer
    answer_text = result['answer'][0] if isinstance(result['answer'], list) else result['answer']

    trajectory_plotted = False
    try:
        # Parse trajectory coordinates
        trajectory_2d = parse_trajectory(answer_text)

        if len(trajectory_2d) > 0:
            # Convert to numpy array
            trajectory_2d = np.array(trajectory_2d)

            # Project to image coordinates
            traj_img, valid_mask = project_ego_to_image(trajectory_2d, cam_front)

            # Filter out points outside image
            in_image = (traj_img[:, 0] >= 0) & (traj_img[:, 0] < img_width) & \
                       (traj_img[:, 1] >= 0) & (traj_img[:, 1] < img_height)
            valid_mask = valid_mask & in_image

            if debug:
                print(f"\nSample: {result['id']}")
                print(f"  Trajectory points: {len(trajectory_2d)}")
                print(f"  Valid points: {valid_mask.sum()}")
                if valid_mask.sum() > 0:
                    print(f"  Image coords: {traj_img[valid_mask][:3]}")

            # Draw trajectory on image
            draw = ImageDraw.Draw(img, 'RGBA')

            # Draw trajectory points and lines
            valid_points = traj_img[valid_mask]
            if len(valid_points) > 1:
                trajectory_plotted = True
                # Draw lines connecting points (green)
                for i in range(len(valid_points) - 1):
                    x1, y1 = valid_points[i]
                    x2, y2 = valid_points[i + 1]
                    draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0, 255), width=5)

                # Draw points (red with yellow outline)
                for i, (x, y) in enumerate(valid_points):
                    radius = 10 if i == 0 else 7  # Larger first point
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                                fill=(255, 50, 50, 255), outline=(255, 255, 0, 255), width=3)

    except Exception as e:
        if debug:
            print(f"Warning: Could not plot trajectory for {result['id']}: {e}")
            import traceback
            traceback.print_exc()

    # Draw semi-transparent background for text
    draw = ImageDraw.Draw(img, 'RGBA')
    text_bg_height = 120
    draw.rectangle([(0, 0), (img_width, text_bg_height)], fill=(0, 0, 0, 200))

    # Convert back to RGB for text drawing
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw sample ID
    sample_id = result['id'].replace('_trajectory', '')
    draw.text((10, 8), f"Sample: {sample_id[:32]}", fill=(0, 255, 0), font=font_large)

    # Draw predicted trajectory coordinates
    draw.text((10, 35), "Predicted Trajectory:", fill=(255, 255, 0), font=font_small)
    traj_text = answer_text[:120] + "..." if len(answer_text) > 120 else answer_text
    draw.text((10, 53), traj_text, fill=(255, 255, 255), font=font_small)

    # Add legend
    status = "Plotted on image" if trajectory_plotted else "Not visible in camera view"
    draw.text((10, 75), f"Status: {status}", fill=(100, 255, 100) if trajectory_plotted else (255, 150, 0), font=font_small)
    if trajectory_plotted:
        draw.text((10, 93), "● Red: Waypoints  — Green: Path",
                  fill=(200, 200, 200), font=font_small)

    # Save visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, quality=95)
    return True

def main():
    # Load inference results
    results_file = 'output/OpenDriveVLA-0.5B/filtered_test/plan_conv.json'

    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return

    # Load NuScenes infos
    print("Loading NuScenes infos...")
    with open('data/infos/nuscenes_infos_temporal_val_filtered.pkl', 'rb') as f:
        nuscenes_data = pickle.load(f)
    nuscenes_infos = nuscenes_data['infos']
    print(f"Loaded {len(nuscenes_infos)} samples")

    # Create mapping
    token_to_sample = {}
    for sample in nuscenes_infos:
        token_to_sample[sample['token']] = sample

    # Read results and create visualizations
    print(f"\nCreating visualizations with trajectory overlay...")
    output_dir = 'output/OpenDriveVLA-0.5B/filtered_test/visualizations_plotted'

    success_count = 0
    with open(results_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            result = json.loads(line.strip())
            sample_token = result['id'].replace('_trajectory', '')
            sample_info = token_to_sample.get(sample_token)

            if sample_info is None:
                continue

            output_path = os.path.join(output_dir, f"vis_{sample_token}.jpg")
            debug = (line_num <= 3)  # Debug first 3 samples

            if create_visualization(sample_info, result, output_path, debug=debug):
                success_count += 1
                if success_count % 20 == 0:
                    print(f"  Created {success_count} visualizations...")

    print(f"\n✓ Done! Created {success_count} visualizations")
    print(f"Output directory: {output_dir}/")

if __name__ == '__main__':
    main()
