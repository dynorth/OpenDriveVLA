#!/usr/bin/env python3
"""
Create a filtered validation dataset containing only samples from available scenes.
This filters out samples that reference missing image files.
"""

import pickle
import os
from pathlib import Path

# Available scenes (23 scenes) based on status document
AVAILABLE_SCENES = {
    'n008-2018-08-01-15-16-36-0400',
    'n008-2018-08-01-15-34-25-0400',
    'n008-2018-08-01-16-03-27-0400',
    'n008-2018-08-22-16-06-57-0400',
    'n008-2018-08-27-11-48-51-0400',
    'n008-2018-08-28-15-47-40-0400',
    'n008-2018-08-28-16-05-27-0400',
    'n008-2018-08-28-16-43-51-0400',
    'n008-2018-08-30-15-16-55-0400',
    'n008-2018-08-31-12-15-24-0400',
    'n008-2018-09-18-14-18-33-0400',
    'n015-2018-07-24-11-22-45+0800',
    'n015-2018-07-25-16-15-50+0800',
    'n015-2018-08-01-16-54-05+0800',
    'n015-2018-08-03-12-54-49+0800',
    'n015-2018-10-02-10-50-40+0800',
    'n015-2018-10-02-11-11-43+0800',
    'n015-2018-10-02-11-23-23+0800',
    'n015-2018-10-08-15-36-50+0800',
    'n015-2018-11-14-19-21-41+0800',
    'n015-2018-11-14-19-45-36+0800',
    'n015-2018-11-14-19-52-02+0800',
    'n015-2018-11-21-19-38-26+0800',
}

def scene_name_from_path(path):
    """Extract scene name from a file path like samples/CAM_FRONT/n008-..._.jpg"""
    filename = Path(path).name
    # Scene name is everything before the double underscore
    if '__' in filename:
        scene = filename.split('__')[0]
        return scene
    return None

def main():
    print("Loading validation pickle...")
    input_pkl = 'data/infos/nuscenes_infos_temporal_val.pkl'
    output_pkl = 'data/infos/nuscenes_infos_temporal_val_filtered.pkl'

    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)

    print(f"Original dataset: {len(data['infos'])} samples")

    # Filter samples to only include those from available scenes
    filtered_infos = []
    skipped_count = 0

    for i, sample in enumerate(data['infos']):
        # Check if the sample's front camera image is from an available scene
        if 'cams' in sample and 'CAM_FRONT' in sample['cams']:
            cam_front_path = sample['cams']['CAM_FRONT']['data_path']
            scene = scene_name_from_path(cam_front_path)

            if scene in AVAILABLE_SCENES:
                # Verify the file actually exists
                full_path = os.path.join('data/nuscenes', cam_front_path)
                if os.path.exists(full_path):
                    filtered_infos.append(sample)
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data['infos'])} samples...")

    # Create new dataset dict with filtered samples
    filtered_data = data.copy()
    filtered_data['infos'] = filtered_infos

    print(f"\nFiltered dataset: {len(filtered_infos)} samples")
    print(f"Skipped: {skipped_count} samples")
    print(f"Retention rate: {len(filtered_infos) / len(data['infos']) * 100:.1f}%")

    # Save filtered pickle
    print(f"\nSaving to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(filtered_data, f)

    print("✓ Done! Use this filtered dataset for inference.")
    print(f"\nTo use the filtered dataset, modify the config file:")
    print(f"  projects/configs/stage1_track_map/base_track_map.py")
    print(f"  Change: ann_file_test=info_root + 'nuscenes_infos_temporal_val.pkl'")
    print(f"  To:     ann_file_test=info_root + 'nuscenes_infos_temporal_val_filtered.pkl'")

if __name__ == '__main__':
    main()
