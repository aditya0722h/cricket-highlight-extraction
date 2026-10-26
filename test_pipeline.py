"""
End-to-End Test Suite for Structured Channel Directories
--------------------------------------------------------
Tests the full modular pipeline across visual, audio, speech, and fusion channels.
"""

import json
import os
import cv2
import numpy as np

# Ensure directory structure exists
for folder in ["visual_channel", "audio_channel", "speech_channel", "fusion_channel"]:
    os.makedirs(folder, exist_ok=True)

# Imports relative to structured root
from visual_channel.extract_frames import extract_frames
from visual_channel.extract_embeddings import extract_embeddings
from visual_channel.optical_flow import compute_optical_flow
from visual_channel.merge_visual_features import merge_visual_features
from fusion_channel.fusion_engine import run_fusion_pipeline


def create_synthetic_video(output_path="synthetic_test.mp4", duration_sec=10, fps=30):
    print(f"Creating test video: {output_path} ({duration_sec}s at {fps} FPS)...")
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for frame_idx in range(total_frames):
        img = np.ones((height, width, 3), dtype=np.uint8) * 128
        t = frame_idx / fps
        # Motion pulse between t = 3s and 5s
        if 3.0 <= t <= 5.0:
            x_pos = int(100 + (t - 3.0) * 200)
            cv2.circle(img, (x_pos, 240), 40, (0, 0, 255), -1)

        out.write(img)

    out.release()


def generate_mock_audio_and_speech(duration_sec=10):
    audio_data = []
    speech_data = []

    for t in np.arange(0.0, duration_sec, 1.0):
        t_rounded = round(float(t), 2)
        ste = 0.9 if 3.0 <= t_rounded <= 5.0 else 0.1
        sentiment = 0.8 if 3.0 <= t_rounded <= 5.0 else 0.2

        audio_data.append({"t": t_rounded, "STE": ste})
        speech_data.append({
            "t": t_rounded,
            "sentiment_score": sentiment,
            "keyword_hit": 3.0 <= t_rounded <= 5.0,
        })

    with open("audio_channel/audio_output.json", "w") as f:
        json.dump(audio_data, f)

    with open("speech_channel/speech_output.json", "w") as f:
        json.dump(speech_data, f)


def run_full_test():
    video_file = "synthetic_test.mp4"
    frames_dir = os.path.join("visual_channel", "frames")

    # 1. Setup Video
    create_synthetic_video(output_path=video_file, duration_sec=10)

    # 2. Visual Channel Pipeline
    print("\n--- Running Visual Channel Pipeline ---")
    extract_frames(video_file, output_dir=frames_dir, target_fps=1)

    extract_embeddings(
        frames_dir=frames_dir,
        output_file=os.path.join("visual_channel", "visual_embeddings.json"),
    )

    compute_optical_flow(
        frames_dir=frames_dir,
        output_file=os.path.join("visual_channel", "visual_motion.json"),
    )

    merge_visual_features(
        embeddings_file=os.path.join("visual_channel", "visual_embeddings.json"),
        motion_file=os.path.join("visual_channel", "visual_motion.json"),
            output_file=os.path.join("visual_channel", "visual_output.json"),
    )

    # 3. Auxiliary Mock Data
    generate_mock_audio_and_speech(duration_sec=10)

    # 4. Fusion Channel Execution
    print("\n--- Running Fusion Channel ---")
    metadata = run_fusion_pipeline(
        source_video=video_file,
            vis_file=os.path.join("visual_channel", "visual_output.json"),
            aud_file=os.path.join("audio_channel", "audio_output.json"),
            speech_file=os.path.join("speech_channel", "speech_output.json"),
            out_json=os.path.join("fusion_channel", "fusion_output.json"),
        top_k=1,
        window_sec=4.0,
    )

    print("\nPipeline execution complete! Deliverable located at:")
    print(f" -> fusion_channel/fusion_output.json")


if __name__ == "__main__":
    run_full_test()
