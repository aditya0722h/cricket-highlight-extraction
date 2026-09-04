"""
Step 2: MobileNetV3 Spatial Feature Extraction
------------------------------------------------
Loads a pretrained MobileNetV3-Small, strips the classification head,
and runs it in FP16 on GPU to extract a spatial saliency embedding
for each frame produced by extract_frames.py.

Output: one embedding vector per frame, paired with its timestamp.
"""

import os
import glob
import re
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import json


def load_mobilenet_feature_extractor(device):
    """
    Loads MobileNetV3-Small pretrained on ImageNet, strips the final
    classifier layer so we get the raw feature embedding instead of
    class predictions.
    """
    model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")

    # Remove the classifier head -> we just want the embedding
    model.classifier = torch.nn.Identity()

    model.eval()          # inference mode (no dropout/batchnorm updates)
    model = model.to(device)
    model = model.half()  # FP16 for VRAM savings, matches project spec

    return model


def get_preprocess_transform():
    """
    Standard ImageNet preprocessing: resize, center-crop, normalize.
    MobileNetV3 expects 224x224 inputs normalized with ImageNet stats.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_timestamp_from_filename(filepath):
    """
    Pulls the timestamp back out of filenames like:
    frame_00001_t1.00.jpg  ->  1.00
    """
    match = re.search(r"_t([\d.]+)\.jpg", filepath)
    return float(match.group(1)) if match else None


def extract_embeddings(frames_dir="frames", output_file="visual_embeddings.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_mobilenet_feature_extractor(device)
    preprocess = get_preprocess_transform()

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in '{frames_dir}/'. Run extract_frames.py first."
        )

    print(f"Found {len(frame_paths)} frames. Extracting embeddings...")

    results = []

    with torch.no_grad():  # no gradient tracking needed for inference
        for i, frame_path in enumerate(frame_paths):
            timestamp = extract_timestamp_from_filename(frame_path)

            image = Image.open(frame_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0)  # add batch dim
            input_tensor = input_tensor.to(device).half()

            embedding = model(input_tensor)               # shape: [1, 576]
            embedding = embedding.squeeze(0).cpu().float().numpy()

            results.append({
                "t": timestamp,
                "frame_path": frame_path,
                "V_spatial": embedding.tolist()
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(frame_paths):
                print(f"  Processed {i + 1}/{len(frame_paths)} frames")

    # Report peak GPU memory used, so we can confirm we're within budget
    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory used: {peak_mem_mb:.1f} MB")

    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Saved {len(results)} embeddings -> {output_file}")
    return results


if __name__ == "__main__":
    extract_embeddings(frames_dir="frames", output_file="visual_embeddings.json")
