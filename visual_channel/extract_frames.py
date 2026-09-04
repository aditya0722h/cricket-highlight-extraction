"""
Step 1: Sparse Frame Extraction
--------------------------------
Reads a video, extracts frames at a controlled rate (default 1 FPS),
and tags each extracted frame with its timestamp in seconds.

This is the foundation for the Visual Channel module — every later step
(MobileNetV3 embeddings, Optical Flow) builds on top of this frame stream.
"""

import cv2
import os

def extract_frames(video_path, output_dir="frames", target_fps=1):
    """
    Extract frames from a video at a controlled rate.

    Args:
        video_path (str): path to the input video file
        output_dir (str): folder to save extracted frames into
        target_fps (float): how many frames to extract per second of video
                             (1 for normal content, 2 for fast sports clips)

    Returns:
        list of (timestamp_seconds, frame_filepath) tuples
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / native_fps if native_fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"Native FPS: {native_fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration_sec:.2f} seconds")
    print(f"Extracting at target {target_fps} FPS...")

    # How many native frames to skip to hit our target rate
    frame_interval = max(1, round(native_fps / target_fps))

    extracted = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / native_fps
            filename = f"frame_{saved_count:05d}_t{timestamp_sec:.2f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            extracted.append((timestamp_sec, filepath))
            saved_count += 1

        frame_idx += 1

    cap.release()

    print(f"Extracted {saved_count} frames -> saved in '{output_dir}/'")
    return extracted


if __name__ == "__main__":
    # EDIT THIS: path to your test video
    VIDEO_PATH = "sample_video.mp4"

    results = extract_frames(VIDEO_PATH, output_dir="frames", target_fps=1)

    # Sanity check: print first 5 extracted (timestamp, filepath) pairs
    print("\nFirst 5 extracted frames:")
    for ts, path in results[:5]:
        print(f"  t={ts:.2f}s -> {path}")
