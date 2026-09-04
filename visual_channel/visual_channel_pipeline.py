"""
Visual Channel - Full Pipeline (Module A)
--------------------------------------------
Single entry point: give it a video, get back the final
timestamped visual feature file ready for Module D (fusion).

Usage:
    python visual_channel_pipeline.py path/to/video.mp4
"""

import sys
import os
import json

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image


# ---------- Step 1: Frame extraction ----------

def extract_frames(video_path, output_dir="frames", target_fps=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(native_fps / target_fps))

    extracted = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / native_fps
            filename = f"frame_{saved_count:05d}_t{timestamp_sec:.2f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            extracted.append((timestamp_sec, filepath))
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"[1/4] Extracted {saved_count} frames at ~{target_fps} FPS")
    return extracted


# ---------- Step 2: MobileNetV3 embeddings ----------

def load_mobilenet(device):
    model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier = torch.nn.Identity()
    model.eval()
    return model.to(device).half()


def get_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def extract_embeddings(frame_paths, device):
    model = load_mobilenet(device)
    preprocess = get_preprocess()
    embeddings = {}

    with torch.no_grad():
        for t, path in frame_paths:
            image = Image.open(path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device).half()
            embedding = model(input_tensor).squeeze(0).cpu().float().numpy()
            embeddings[t] = embedding.tolist()

    print(f"[2/4] Extracted {len(embeddings)} MobileNetV3 embeddings")
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"       Peak GPU memory: {peak_mb:.1f} MB")

    return embeddings


# ---------- Step 3: Optical flow ----------

def compute_optical_flow(frame_paths):
    motion = {}
    prev_gray = None

    for i, (t, path) in enumerate(frame_paths):
        frame = cv2.imread(path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            motion[t] = 0.0
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            dx, dy = flow[..., 0], flow[..., 1]
            motion[t] = float(np.mean(np.sqrt(dx ** 2 + dy ** 2)))

        prev_gray = gray

    print(f"[3/4] Computed optical flow for {len(motion)} frames")
    return motion


# ---------- Step 4: Merge ----------

def merge(embeddings, motion, output_file):
    merged = []
    for t in sorted(embeddings.keys()):
        merged.append({
            "t": t,
            "V_spatial": embeddings[t],
            "V_motion": motion.get(t, None)
        })

    with open(output_file, "w") as f:
        json.dump(merged, f)

    print(f"[4/4] Saved final output -> {output_file}")
    return merged


# ---------- Orchestration ----------

def run_visual_channel(video_path, frames_dir="frames", output_file="visual_channel_output.json", target_fps=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running visual channel pipeline on device: {device}")
    print(f"Input video: {video_path}\n")

    frame_paths = extract_frames(video_path, output_dir=frames_dir, target_fps=target_fps)
    embeddings = extract_embeddings(frame_paths, device)
    motion = compute_optical_flow(frame_paths)
    merged = merge(embeddings, motion, output_file)

    print(f"\nDone. {len(merged)} timestamped entries ready for Module D.")
    return merged


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visual_channel_pipeline.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    run_visual_channel(video_path)
