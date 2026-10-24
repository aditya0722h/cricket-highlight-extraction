"""Visual Channel - Full Pipeline (Module A) - Low-VRAM Optimized."""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms


def extract_frames(video_path, output_dir, target_fps=1):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: '{video_path}'")
    if os.path.getsize(video_path) == 0:
        raise ValueError(f"Video file '{video_path}' is empty (0 bytes).")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open '{video_path}'.")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame rate ({native_fps} FPS).")

    frame_interval = max(1, round(native_fps / target_fps))
    extracted = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp_sec = round(frame_idx / native_fps, 2)
            filename = f"frame_{saved_count:05d}_t{timestamp_sec:.2f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            extracted.append((timestamp_sec, filepath))
            saved_count += 1
        frame_idx += 1

    cap.release()
    if saved_count == 0:
        raise RuntimeError("No frames extracted.")

    print(f"[1/4] Extracted {saved_count} frames at ~{target_fps} FPS.")
    return extracted


def load_mobilenet(device):
    model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier = torch.nn.Identity()
    model.eval()
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    return model


def get_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


def extract_embeddings(frame_paths, device, batch_size=32):
    model = load_mobilenet(device)
    preprocess = get_preprocess()
    embeddings = {}

    # Mini-batch inference for maximum GPU utilization
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i : i + batch_size]
        tensors = []
        timestamps = []

        for t, path in batch_paths:
            img = Image.open(path).convert("RGB")
            tensors.append(preprocess(img))
            timestamps.append(t)

        batch_tensor = torch.stack(tensors).to(device)
        if device == "cuda":
            batch_tensor = batch_tensor.half()

        with torch.no_grad():
            outputs = model(batch_tensor).cpu().float().numpy()

        for idx, t in enumerate(timestamps):
            embeddings[t] = outputs[idx].tolist()

    print(f"[2/4] Extracted {len(embeddings)} MobileNetV3 embeddings.")

    # Memory cleanup for RTX 2050
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return embeddings


def compute_optical_flow(frame_paths):
    motion = {}
    prev_gray = None

    for t, path in frame_paths:
        frame = cv2.imread(path)
        # Downscale frame for speed without changing motion metric ratio
        resized = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            motion[t] = 0.0
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            dx, dy = flow[..., 0], flow[..., 1]
            motion[t] = float(np.mean(np.sqrt(dx**2 + dy**2)))

        prev_gray = gray

    print(f"[3/4] Computed optical flow for {len(motion)} frames.")
    return motion


def merge(embeddings, motion, output_file):
    merged = []
    for t in sorted(embeddings.keys()):
        merged.append({
            "t": round(float(t), 2),
            "V_spatial": [float(v) for v in embeddings[t]],
            "V_motion": round(motion.get(t, 0.0), 4),
        })

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged, f)

    print(f"[4/4] Saved output -> {output_file}")
    return merged


def run_visual_channel(
    video_path, output_file=None, target_fps=1
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Visual Channel on {device}...")
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(video_path) or ".", "visual_output.json"
        )

    # Temp directory handles auto-cleanup on completion
    temp_dir = tempfile.mkdtemp(prefix="cricket_frames_")

    try:
        frame_paths = extract_frames(
            video_path, output_dir=temp_dir, target_fps=target_fps
        )
        embeddings = extract_embeddings(frame_paths, device)
        motion = compute_optical_flow(frame_paths)
        merged = merge(embeddings, motion, output_file)
        return merged
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visual_channel_pipeline_2.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    
    # UPDATED: Automatically route output to match the video's directory
    output_filename = "visual_output.json"
    output_filepath = os.path.join(
        os.path.dirname(video_path) or ".", output_filename
    )

    try:
        run_visual_channel(video_path, output_file=output_filepath)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        sys.exit(1)
