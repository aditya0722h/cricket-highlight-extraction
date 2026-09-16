"""Step 3: Farneback Optical Flow - Motion Feature Extraction.

Optimized for speed and sequence integrity.
"""

import glob
import json
import os
import re
import cv2
import numpy as np


def extract_timestamp_from_filename(filepath):
    match = re.search(r"_t([\d.]+)\.jpg", filepath)
    return float(match.group(1)) if match else -1.0


def compute_optical_flow(
    frames_dir="frames", output_file="visual_motion.json"
):
    # Fetch and sort strictly by actual timestamp rather than string name
    frame_paths = glob.glob(os.path.join(frames_dir, "*.jpg"))
    frame_paths.sort(key=extract_timestamp_from_filename)

    if len(frame_paths) < 2:
        raise ValueError("Need at least 2 frames to compute optical flow.")

    print(f"Found {len(frame_paths)} ordered frames. Computing flow...")

    results = []

    # First frame setup
    prev_frame = cv2.imread(frame_paths[0])
    # Downscale frame to speed up CPU optical flow calculation
    prev_resized = cv2.resize(prev_frame, (640, 360))
    prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2GRAY)

    first_timestamp = extract_timestamp_from_filename(frame_paths[0])
    results.append({"t": round(first_timestamp, 2), "V_motion": 0.0})

    for i in range(1, len(frame_paths)):
        curr_frame = cv2.imread(frame_paths[i])
        curr_resized = cv2.resize(curr_frame, (640, 360))
        curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY)

        # Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        dx = flow[..., 0]
        dy = flow[..., 1]
        v_motion = float(np.mean(np.sqrt(dx**2 + dy**2)))

        timestamp = extract_timestamp_from_filename(frame_paths[i])
        results.append({"t": round(timestamp, 2), "V_motion": round(v_motion, 4)})

        prev_gray = curr_gray

        if (i + 1) % 50 == 0 or (i + 1) == len(frame_paths):
            print(f"  Processed {i + 1}/{len(frame_paths)} frames")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    motion_values = [r["V_motion"] for r in results]
    print(
        f"Saved {len(results)} values to {output_file} | "
        f"Range: min={min(motion_values):.2f}, "
        f"max={max(motion_values):.2f}, avg={np.mean(motion_values):.2f}"
    )

    return results


if __name__ == "__main__":
    compute_optical_flow(frames_dir="frames", output_file="visual_motion.json")