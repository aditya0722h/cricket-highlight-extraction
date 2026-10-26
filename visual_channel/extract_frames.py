"""Step 1: Sparse Frame Extraction.

Reads a video, extracts frames at a controlled rate (default 1 FPS),
and tags each extracted frame with a rounded timestamp in seconds.
"""

import os
import cv2


def extract_frames(video_path, output_dir="frames", target_fps=1):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if os.path.getsize(video_path) == 0:
        raise ValueError(f"Video file is empty (0 bytes): {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid native frame rate ({native_fps} FPS).")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / native_fps

    print(f"Video: {video_path}")
    print(f"Native FPS: {native_fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration_sec:.2f} seconds")
    print(f"Extracting at target {target_fps} FPS...")

    frame_interval = max(1, round(native_fps / target_fps))
    extracted = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Round timestamp explicitly to 2 decimal places
            timestamp_sec = round(frame_idx / native_fps, 2)
            filename = f"frame_{saved_count:05d}_t{timestamp_sec:.2f}.jpg"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, frame)
            extracted.append((timestamp_sec, filepath))
            saved_count += 1

        frame_idx += 1

    cap.release()

    if saved_count == 0:
        raise RuntimeError(f"No frames could be extracted from {video_path}")

    print(f"Extracted {saved_count} frames -> saved in '{output_dir}/'")
    return extracted


if __name__ == "__main__":
    VIDEO_PATH = "sample_video.mp4"
    try:
        results = extract_frames(
            VIDEO_PATH, output_dir="frames", target_fps=1
        )
        print("\nFirst 5 extracted frames:")
        for ts, path in results[:5]:
            print(f"  t={ts:.2f}s -> {path}")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")