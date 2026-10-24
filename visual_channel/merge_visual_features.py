"""Step 4: Merge Visual Features into Final Output Schema.

Combines V_spatial (embeddings) and V_motion (optical flow) into a single
timestamped file keyed by rounded scalar timestamp.
"""

import json


def merge_visual_features(
    embeddings_file="visual_embeddings.json",
    motion_file="visual_motion.json",
    output_file="visual_output.json",
):
    with open(embeddings_file, "r") as f:
        embeddings = json.load(f)

    with open(motion_file, "r") as f:
        motion = json.load(f)

    if len(embeddings) != len(motion):
        print(
            f"WARNING: Entry count mismatch - "
            f"embeddings={len(embeddings)}, motion={len(motion)}"
        )

    # Key by rounded float to prevent precision mismatch issues
    motion_by_t = {round(float(m["t"]), 2): m["V_motion"] for m in motion}

    merged = []
    missing_motion_count = 0

    for entry in embeddings:
        t = round(float(entry["t"]), 2)
        v_motion = motion_by_t.get(t)

        if v_motion is None:
            missing_motion_count += 1
            v_motion = 0.0  # Safe fallback default

        merged.append({
            "t": t,
            "V_spatial": [float(v) for v in entry["V_spatial"]],
            "V_motion": float(v_motion),
        })

    with open(output_file, "w") as f:
        json.dump(merged, f)

    print(f"Successfully merged {len(merged)} entries -> {output_file}")
    if missing_motion_count > 0:
        print(
            f"WARNING: {missing_motion_count} timestamps could not find matching motion scores. "
            f"Fallback value 0.0 was assigned."
        )

    if merged:
        print(
            f"Sample entry (t={merged[0]['t']}): "
            f"V_spatial len={len(merged[0]['V_spatial'])}, "
            f"V_motion={merged[0]['V_motion']}"
        )

    return merged


if __name__ == "__main__":
    merge_visual_features()
