"""Module D: Fusion Engine & Video Renderer.

Combines multimodal signals (Visual, Audio, Speech) into unified score streams,
detects key highlight segments, and cuts the final output video.
"""

import json
import os
import subprocess
import numpy as np
from sports.config import detect_events, load_sport_config


def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"WARNING: File not found ({filepath}). Returning empty dataset.")
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def normalize(array):
    """Min-max normalization to fit values in [0, 1]."""
    arr = np.array(array, dtype=np.float32)
    if len(arr) == 0:
        return arr
    min_v, max_v = np.min(arr), np.max(arr)
    if max_v - min_v == 0:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def compute_fused_scores(
    visual_data,
    audio_data,
    speech_data,
    weights=None,
    sport="cricket",
):
    """Aligns multi-channel streams by timestamp and computes a weighted fusion score."""
    config = load_sport_config(sport)
    if weights is None:
        profile_weights = config["fusion_weights"]["channel_engine"]
        weights = (
            profile_weights["visual"],
            profile_weights["audio"],
            profile_weights["speech"],
        )
    w_vis, w_aud, w_sp = weights
    keyword_hit_bonus = config["keyword_hit_bonus"]
    neutral_sentiment = config["neutral_sentiment"]

    # Build timestamp-indexed dicts (rounding keys to 2 decimals)
    vis_dict = {
        round(float(item["t"]), 2): item.get("V_motion", 0.0)
        for item in visual_data
    }
    aud_dict = {
        round(float(item["t"]), 2): item.get("STE", 0.0)
        for item in audio_data
    }
    sp_dict = {
        round(float(item["t"]), 2): (
            item.get("sentiment_score", neutral_sentiment)
            + (keyword_hit_bonus if item.get("keyword_hit", False) else 0.0)
        )
        for item in speech_data
    }

    # Collect all unique sorted timestamps
    all_timestamps = sorted(
        list(set(vis_dict.keys()) | set(aud_dict.keys()) | set(sp_dict.keys()))
    )

    if not all_timestamps:
        raise ValueError("No valid timestamps found across input channels.")

    # Extract raw series
    v_raw = [vis_dict.get(t, 0.0) for t in all_timestamps]
    a_raw = [aud_dict.get(t, 0.0) for t in all_timestamps]
    s_raw = [sp_dict.get(t, 0.0) for t in all_timestamps]

    # Normalize each modality signal independently
    v_norm = normalize(v_raw)
    a_norm = normalize(a_raw)
    s_norm = normalize(s_raw)

    fused_stream = []
    for i, t in enumerate(all_timestamps):
        score = float(
            (w_vis * v_norm[i]) + (w_aud * a_norm[i]) + (w_sp * s_norm[i])
        )
        fused_stream.append({
            "t": t,
            "fused_score": round(score, 4),
            "V": round(float(v_norm[i]), 4),
            "A": round(float(a_norm[i]), 4),
            "S": round(float(s_norm[i]), 4),
        })

    return fused_stream


def extract_highlight_windows(
    fused_stream, top_k=None, window_sec=None, min_gap_sec=None, sport="cricket"
):
    """Finds top highlight time windows using peak score aggregation."""
    detection = load_sport_config(sport)["fusion_detection"]
    top_k = detection["top_k"] if top_k is None else top_k
    window_sec = detection["window_sec"] if window_sec is None else window_sec
    min_gap_sec = detection["min_gap_sec"] if min_gap_sec is None else min_gap_sec
    timestamps = np.array([item["t"] for item in fused_stream])
    scores = np.array([item["fused_score"] for item in fused_stream])

    # Dynamic sliding window scoring
    half_w = window_sec / 2.0
    window_scores = []

    for t in timestamps:
        mask = (timestamps >= (t - half_w)) & (timestamps <= (t + half_w))
        window_scores.append(np.mean(scores[mask]) if np.any(mask) else 0.0)

    window_scores = np.array(window_scores)

    # Pick top-K non-overlapping peaks
    selected_windows = []
    sorted_indices = np.argsort(window_scores)[::-1]

    for idx in sorted_indices:
        if len(selected_windows) >= top_k:
            break

        center_t = timestamps[idx]
        start_t = max(0.0, center_t - half_w)
        end_t = center_t + half_w

        # Check overlap against already chosen windows
        overlap = False
        for s, e in selected_windows:
            if not (end_t < (s - min_gap_sec) or start_t > (e + min_gap_sec)):
                overlap = True
                break

        if not overlap:
            selected_windows.append((round(start_t, 2), round(end_t, 2)))

    # Sort windows chronologically
    selected_windows.sort(key=lambda x: x[0])
    return selected_windows


def render_highlights_ffmpeg(
    video_path, windows, output_dir="output_highlights"
):
    """Cuts and exports highlight segments using ffmpeg CLI directly."""
    os.makedirs(output_dir, exist_ok=True)
    exported_files = []

    for idx, (start_t, end_t) in enumerate(windows, 1):
        duration = end_t - start_t
        out_path = os.path.join(output_dir, f"highlight_{idx:02d}.mp4")

        # High-speed stream copy command without re-encoding
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_t),
            "-i",
            video_path,
            "-t",
            str(duration),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            out_path,
        ]

        try:
            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            exported_files.append(out_path)
            print(f"  Rendered clip {idx}: {start_t}s - {end_t}s -> {out_path}")
        except Exception as e:
            print(f"  FFmpeg rendering failed for clip {idx}: {e}")

    return exported_files


def run_fusion_pipeline(
    source_video="sample_video.mp4",
    vis_file="visual_output.json",
    aud_file="audio_output.json",
    speech_file="speech_output.json",
    out_json="fusion_output.json",
    top_k=None,
    window_sec=None,
    sport="cricket",
):
    print("--- Module D: Fusion Engine & Video Renderer ---")
    config = load_sport_config(sport)

    # Step 1: Load stream files
    v_data = load_json(vis_file)
    a_data = load_json(aud_file)
    s_data = load_json(speech_file)

    # Step 2: Merge & compute multimodal fusion
    fused_stream = compute_fused_scores(v_data, a_data, s_data, sport=sport)

    # Step 3: Peak detection for top-k highlight windows
    windows = extract_highlight_windows(
        fused_stream,
        top_k=top_k,
        window_sec=(
            config["fusion_detection"]["pipeline_window_sec"]
            if window_sec is None
            else window_sec
        ),
        min_gap_sec=config["fusion_detection"]["min_gap_sec"],
        sport=sport,
    )

    print(f"\nExtracted {len(windows)} Highlight Windows:")
    for i, (s, e) in enumerate(windows, 1):
        print(f"  [{i}] {s:.2f}s to {e:.2f}s (Duration: {e-s:.2f}s)")

    # Save metadata JSON deliverable
    records_by_time = {}
    for channel in (v_data, a_data, s_data):
        for record in channel:
            timestamp = round(float(record["t"]), 2)
            records_by_time.setdefault(timestamp, {"t": timestamp}).update(record)
    events = detect_events(list(records_by_time.values()), config)
    metadata = {
        "source_video": source_video,
        "sport": sport,
        "highlights": windows,
        "events": events,
    }
    with open(out_json, "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 4: Render clips to disk
    if os.path.exists(source_video):
        print("\nRendering clips via FFmpeg...")
        render_highlights_ffmpeg(source_video, windows)
    else:
        print(
            f"\nSource video path '{source_video}' not found on disk. Skipped video clipping."
        )

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=("cricket", "football"), default="cricket")
    run_fusion_pipeline(sport=parser.parse_args().sport)
