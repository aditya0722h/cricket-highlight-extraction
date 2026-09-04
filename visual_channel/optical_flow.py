"""
Step 3: Farneback Optical Flow - Motion Feature Extraction
-------------------------------------------------------------
Computes dense optical flow between consecutive extracted frames
to measure motion intensity (camera pans, fast action, etc.)

Formula: V_motion = (1/N) * sum( sqrt(dx^2 + dy^2) )  across all pixels

Output: one motion magnitude value per frame-pair, tagged with timestamp,
saved alongside the spatial embeddings from Step 2.
"""

import os
import glob
import re
import cv2
import numpy as np
import json


def extract_timestamp_from_filename(filepath):
    match = re.search(r"_t([\d.]+)\.jpg", filepath)
    return float(match.group(1)) if match else None


def compute_optical_flow(frames_dir="frames", output_file="visual_motion.json"):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if len(frame_paths) < 2:
        raise ValueError("Need at least 2 frames to compute optical flow.")

    print(f"Found {len(frame_paths)} frames. Computing optical flow between consecutive pairs...")

    results = []

    # Load and convert the first frame to grayscale as our starting point
    prev_frame = cv2.imread(frame_paths[0])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # The very first frame has no "previous" frame to compare against,
    # so we record it with V_motion = 0.0 (baseline/static assumption)
    first_timestamp = extract_timestamp_from_filename(frame_paths[0])
    results.append({
        "t": first_timestamp,
        "frame_path": frame_paths[0],
        "V_motion": 0.0
    })

    for i in range(1, len(frame_paths)):
        curr_frame = cv2.imread(frame_paths[i])
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Farneback dense optical flow: returns a (dx, dy) vector per pixel
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,           # no initial flow estimate
            pyr_scale=0.5,  # image pyramid scale between levels
            levels=3,       # number of pyramid layers
            winsize=15,     # averaging window size
            iterations=3,   # iterations at each pyramid level
            poly_n=5,       # pixel neighborhood size for polynomial expansion
            poly_sigma=1.2, # Gaussian std dev for polynomial expansion
            flags=0
        )

        dx = flow[..., 0]
        dy = flow[..., 1]
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        v_motion = float(np.mean(magnitude))

        timestamp = extract_timestamp_from_filename(frame_paths[i])
        results.append({
            "t": timestamp,
            "frame_path": frame_paths[i],
            "V_motion": v_motion
        })

        prev_gray = curr_gray

        if (i + 1) % 10 == 0 or (i + 1) == len(frame_paths):
            print(f"  Processed {i + 1}/{len(frame_paths)} frames")

    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Saved {len(results)} motion values -> {output_file}")

    # Quick summary so you can sanity-check against the docs
    # (static baseline ~1-2, fast camera sweep ~15-20)
    motion_values = [r["V_motion"] for r in results]
    print(f"Motion magnitude range: min={min(motion_values):.2f}, "
          f"max={max(motion_values):.2f}, avg={np.mean(motion_values):.2f}")

    return results


if __name__ == "__main__":
    compute_optical_flow(frames_dir="frames", output_file="visual_motion.json")
