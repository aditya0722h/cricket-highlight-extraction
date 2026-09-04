"""
Step 4: Merge Visual Features into Final Output Schema
---------------------------------------------------------
Combines V_spatial (MobileNetV3 embeddings) and V_motion (Optical Flow)
into a single timestamped file - the actual deliverable of the
Visual Channel module, ready to hand off to the Fusion module (Module D).
"""

import json


def merge_visual_features(
    embeddings_file="visual_embeddings.json",
    motion_file="visual_motion.json",
    output_file="visual_channel_output.json"
):
    with open(embeddings_file) as f:
        embeddings = json.load(f)

    with open(motion_file) as f:
        motion = json.load(f)

    if len(embeddings) != len(motion):
        print(f"WARNING: mismatch in entry counts - "
              f"embeddings={len(embeddings)}, motion={len(motion)}")

    # Index motion values by timestamp for safe lookup
    motion_by_t = {m["t"]: m["V_motion"] for m in motion}

    merged = []
    for entry in embeddings:
        t = entry["t"]
        merged.append({
            "t": t,
            "V_spatial": entry["V_spatial"],
            "V_motion": motion_by_t.get(t, None)
        })

    with open(output_file, "w") as f:
        json.dump(merged, f)

    print(f"Merged {len(merged)} entries -> {output_file}")
    print(f"Sample entry (t={merged[0]['t']}): "
          f"V_spatial length={len(merged[0]['V_spatial'])}, "
          f"V_motion={merged[0]['V_motion']}")

    return merged


if __name__ == "__main__":
    merge_visual_features()
