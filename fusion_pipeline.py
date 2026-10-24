import argparse
import json
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd

def min_max_scale(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)

def load_and_merge_json(vis_path, aud_path, spk_path):
    with open(vis_path, "r") as f: df_vis = pd.DataFrame(json.load(f))
    with open(aud_path, "r") as f: df_aud = pd.DataFrame(json.load(f))
    with open(spk_path, "r") as f: df_spk = pd.DataFrame(json.load(f))

    df = pd.merge(df_vis, df_aud, on="t", how="outer")
    df = pd.merge(df, df_spk, on="t", how="outer")
    
    for col, default in [("V_motion", 0.0), ("V_spatial", None), ("STE", 0.0), ("sentiment_score", 0.5), ("keyword_hit", False)]:
        if col not in df.columns: df[col] = default
        elif col != "V_spatial": df[col] = df[col].fillna(default)
    df["V_spatial"] = df["V_spatial"].apply(lambda x: x if isinstance(x, list) else [])
    return df.sort_values("t").reset_index(drop=True)

def spatial_norm(value):
    if not isinstance(value, list) or len(value) == 0:
        return 0.0
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return 0.0
    if arr.ndim != 1 or arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr))

def compute_fusion_score(df, weights):
    df["V_motion_norm"] = min_max_scale(df["V_motion"])
    df["V_spatial_norm"] = min_max_scale(df["V_spatial"].apply(spatial_norm))
    df["STE_norm"] = min_max_scale(df["STE"])
    df["Speech_norm"] = min_max_scale(df["sentiment_score"] + (df["keyword_hit"].astype(float) * 0.5))

    df["S_total"] = (
        weights["wv"] * df["V_motion_norm"] +
        weights["ws"] * df["V_spatial_norm"] +
        weights["wa"] * df["STE_norm"] +
        weights["wt"] * df["Speech_norm"]
    )
    df["S_total_smoothed"] = df["S_total"].rolling(window=5, min_periods=1, center=True).mean()
    return df

def extract_highlight_segments(df, percentile=75, min_duration=4, pad_sec=2):
    scores = pd.to_numeric(df["S_total_smoothed"], errors="coerce")
    valid_scores = scores.dropna()
    if valid_scores.empty:
        return []
    threshold = np.percentile(valid_scores, percentile)
    df["is_highlight"] = scores.notna() & (scores >= threshold)

    segments, in_segment, start_t = [], False, 0.0
    for _, row in df.iterrows():
        if row["is_highlight"] and not in_segment:
            in_segment, start_t = True, max(0.0, row["t"] - pad_sec)
        elif not row["is_highlight"] and in_segment:
            in_segment = False
            if (row["t"] + pad_sec - start_t) >= min_duration:
                segments.append((start_t, row["t"] + pad_sec))
    if in_segment:
        end_t = float(df["t"].iloc[-1]) + pad_sec
        if (end_t - start_t) >= min_duration:
            segments.append((start_t, end_t))
    return segments

def export_clips(video_path, segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not segments:
        print("No highlight segments selected; skipping FFmpeg export.")
        return []
    clips = []
    concat_list = None
    for idx, (start, end) in enumerate(segments, 1):
        out_clip = os.path.join(output_dir, f"reel_{idx:02d}.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", video_path,
            "-t", f"{end - start:.2f}", "-c", "copy", "-avoid_negative_ts", "make_zero", out_clip
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        clips.append(out_clip)

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir=output_dir) as f:
            concat_list = f.name
            for c in clips:
                f.write(f"file '{os.path.abspath(c)}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy",
            os.path.join(output_dir, "final_highlights.mp4")
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    finally:
        if concat_list and os.path.exists(concat_list):
            os.remove(concat_list)
    return clips

def load_match_config(video_id):
    config_path = f"model_config_{video_id}.json"
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", type=str, required=True)
    args = parser.parse_args()

    vid = args.video_id
    base = f"dataset/{vid}"
    
    df = load_and_merge_json(f"{base}/visual_output.json", f"{base}/audio_output.json", f"{base}/speech_output.json")
    config = load_match_config(vid)
    weights = config.get("fusion_weights", {"wv": 0.25, "ws": 0.15, "wa": 0.30, "wt": 0.30})
    
    df = compute_fusion_score(df, weights)
    segments = extract_highlight_segments(df)
    output_json = f"{base}/fusion_output.json"
    df.drop(columns=["V_spatial"], errors="ignore").to_json(output_json, orient="records", indent=2)
    export_clips(f"{base}/match.mp4", segments, f"{base}/output_highlights")

if __name__ == "__main__":
    main()
