"""The script below implements Module D (Fusion + Clip Selection).
It loads the output JSONs from all three modules, merges them
 on the timestamp key (t), normalizes the metrics, computes the 
 weighted score $S_{total}(t)$, and uses FFmpeg to cut and concatenate 
 the highlight clips instantly without re-encoding."""

import argparse
import json
import os
import subprocess
import numpy as np
import pandas as pd

def min_max_scale(series):
    """Min-Max normalization returning values scaled to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)

def load_and_merge_json(visual_path, audio_path, speech_path):
    print("Loading channel outputs...")
    with open(visual_path, 'r') as f:
        df_vis = pd.DataFrame(json.load(f))
    with open(audio_path, 'r') as f:
        df_aud = pd.DataFrame(json.load(f))
    with open(speech_path, 'r') as f:
        df_spk = pd.DataFrame(json.load(f))

    # Merge all channels on time key 't'
    df = pd.merge(df_vis, df_aud, on="t", how="inner")
    df = pd.merge(df, df_spk, on="t", how="inner")
    return df.sort_values("t").reset_index(drop=True)

def compute_fusion_score(df, weights):
    print("Normalizing features and computing score S_total(t)...")
    
    # 1. Visual Motion
    df['V_motion_norm'] = min_max_scale(df['V_motion'])
    
    # 2. Visual Spatial (Vector norm of MobileNetV3 feature embedding)
    df['V_spatial_norm'] = df['V_spatial'].apply(lambda x: np.linalg.norm(x) if isinstance(x, list) else 0.0)
    df['V_spatial_norm'] = min_max_scale(df['V_spatial_norm'])
    
    # 3. Audio Energy (STE)
    df['STE_norm'] = min_max_scale(df['STE'])
    
    # 4. Speech NLP (Combine sentiment score + keyword hit boost)
    speech_signal = df['sentiment_score'] + (df['keyword_hit'].astype(float) * 0.5)
    df['Speech_norm'] = min_max_scale(speech_signal)
    
    # Weighted Fusion Formula: S_total(t)
    df['S_total'] = (
        weights['wv'] * df['V_motion_norm'] +
        weights['ws'] * df['V_spatial_norm'] +
        weights['wa'] * df['STE_norm'] +
        weights['wt'] * df['Speech_norm']
    )
    
    # Apply a 5-second centered rolling average to smooth temporal spikes
    df['S_total_smoothed'] = df['S_total'].rolling(window=5, min_periods=1, center=True).mean()
    return df

def extract_highlight_segments(df, percentile=85, min_duration=4, pad_sec=2):
    """Identifies high-score intervals and groups contiguous timestamps into start/end windows."""
    threshold = np.percentile(df['S_total_smoothed'], percentile)
    print(f"Applying score threshold: S_total >= {threshold:.4f} (Top {100-percentile}% percentile)")
    
    df['is_highlight'] = df['S_total_smoothed'] >= threshold
    
    raw_segments = []
    in_segment = False
    start_t, end_t = 0.0, 0.0
    
    for _, row in df.iterrows():
        t = row['t']
        is_hl = row['is_highlight']
        
        if is_hl and not in_segment:
            in_segment = True
            start_t = max(0.0, t - pad_sec)
        elif not is_hl and in_segment:
            in_segment = False
            end_t = t + pad_sec
            if (end_t - start_t) >= min_duration:
                raw_segments.append((start_t, end_t))
    
    if in_segment:
        end_t = df['t'].iloc[-1] + pad_sec
        if (end_t - start_t) >= min_duration:
            raw_segments.append((start_t, end_t))
            
    # Merge segments closer than 3 seconds apart
    merged_segments = []
    for start, end in raw_segments:
        if not merged_segments:
            merged_segments.append([start, end])
        else:
            prev_start, prev_end = merged_segments[-1]
            if start <= prev_end + 3.0:
                merged_segments[-1][1] = max(prev_end, end)
            else:
                merged_segments.append([start, end])
                
    return merged_segments

def export_clips(video_path, segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    clip_files = []
    
    print(f"Exporting {len(segments)} highlight clips using FFmpeg stream copy...")
    for idx, (start, end) in enumerate(segments, 1):
        duration = end - start
        out_clip = os.path.join(output_dir, f"highlight_{idx:02d}.mp4")
        
        # Fast loss-less stream copy without re-encoding
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.2f}",
            "-i", video_path,
            "-t", f"{duration:.2f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out_clip
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_files.append(out_clip)
        print(f" Saved: {out_clip} [{start:.1f}s - {end:.1f}s | Duration: {duration:.1f}s]")
        
    # Stitch clips into a single compilation reel
    concat_list_path = os.path.join(output_dir, "concat_list.txt")
    final_output = os.path.join(output_dir, "final_highlights.mp4")
    
    with open(concat_list_path, 'w') as f:
        for file in clip_files:
            f.write(f"file '{os.path.abspath(file)}'\n")
            
    concat_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        final_output
    ]
    subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"\nFinal highlight reel exported to: {final_output}")

def main():
    parser = argparse.ArgumentParser(description="Module D: Multimodal Fusion & Clip Selection.")
    parser.add_argument("video_path", type=str, help="Path to original cricket MP4 video.")
    parser.add_argument("--vis", type=str, default="visual_channel_output.json")
    parser.add_argument("--aud", type=str, default="audio_channel_output.json")
    parser.add_argument("--spk", type=str, default="speech_channel_output.json")
    parser.add_argument("--out", type=str, default="output_highlights")
    args = parser.parse_args()

    # Cricket domain weighting table
    weights = {'wv': 0.25, 'ws': 0.15, 'wa': 0.30, 'wt': 0.30}
    
    df = load_and_merge_json(args.vis, args.aud, args.spk)
    df = compute_fusion_score(df, weights)
    
    # Save unified fusion output JSON
    df.to_json("fusion_output.json", orient="records", indent=2)
    print("Saved merged database to fusion_output.json")
    
    segments = extract_highlight_segments(df, percentile=85)
    export_clips(args.video_path, segments, args.out)

if __name__ == "__main__":
    main()