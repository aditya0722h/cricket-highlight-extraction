"""Step 2: MobileNetV3 Spatial Feature Extraction.

Loads MobileNetV3-Small, strips the classification head, and extracts 576-dim
embeddings in mini-batches (FP16 on CUDA).
"""

import glob
import json
import os
import re
import gc
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms


def load_mobilenet_feature_extractor(device):
    model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier = torch.nn.Identity()
    model.eval()
    model = model.to(device)

    # FP16 precision ONLY if on CUDA to prevent CPU float16 crash
    if device == "cuda":
        model = model.half()

    return model


def get_preprocess_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


def extract_timestamp_from_filename(filepath):
    match = re.search(r"_t([\d.]+)\.jpg", filepath)
    return float(match.group(1)) if match else -1.0


def extract_embeddings(
    frames_dir="frames",
    output_file="visual_embeddings.json",
    batch_size=32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_mobilenet_feature_extractor(device)
    preprocess = get_preprocess_transform()

    # Sort strictly by extracted numerical timestamp
    frame_paths = glob.glob(os.path.join(frames_dir, "*.jpg"))
    frame_paths.sort(key=extract_timestamp_from_filename)

    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in '{frames_dir}/'. Run Step 1 first."
        )

    print(f"Found {len(frame_paths)} frames. Extracting embeddings in batches of {batch_size}...")

    results = []

    with torch.no_grad():
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i : i + batch_size]
            tensors = []
            timestamps = []

            for path in batch_paths:
                timestamp = extract_timestamp_from_filename(path)
                image = Image.open(path).convert("RGB")
                tensors.append(preprocess(image))
                timestamps.append(timestamp)

            batch_tensor = torch.stack(tensors).to(device)
            if device == "cuda":
                batch_tensor = batch_tensor.half()

            # Shape: [batch_size, 576]
            embeddings = model(batch_tensor).cpu().float().numpy()

            for idx, timestamp in enumerate(timestamps):
                results.append({
                    "t": round(timestamp, 2),
                    "V_spatial": embeddings[idx].tolist(),
                })

            if (i + len(batch_paths)) % 64 == 0 or (i + len(batch_paths)) == len(frame_paths):
                print(f"  Processed {i + len(batch_paths)}/{len(frame_paths)} frames")

    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU memory used: {peak_mem_mb:.1f} MB")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Saved {len(results)} embeddings -> {output_file}")
    return results


if __name__ == "__main__":
    extract_embeddings(frames_dir="frames", output_file="visual_embeddings.json")