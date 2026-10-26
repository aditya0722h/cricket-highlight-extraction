import argparse
import json
import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.mobilenet_v3_small(weights="DEFAULT")
model.classifier = torch.nn.Identity()
model.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_frame_embeddings(video_path, target_fps=1):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS ({native_fps}) for video: {video_path}")
    interval = max(1, round(native_fps / target_fps))
    
    embeddings = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = model(tensor).squeeze().cpu().numpy()
                norm = np.linalg.norm(emb)
                if norm == 0:
                    continue
                emb = emb / norm
                embeddings.append(emb)
        frame_idx += 1
        
    cap.release()
    return np.array(embeddings)

def auto_label_match(video_id, similarity_threshold=0.92):
    base_dir = f"dataset/{video_id}"
    full_match_json = f"{base_dir}/visual_output.json"
    official_highlight_mp4 = f"{base_dir}/official_highlights.mp4"
    
    print(f"Extracting features from official highlight video for {video_id}...")
    hl_embeddings = get_frame_embeddings(official_highlight_mp4)
    if hl_embeddings.size == 0:
        raise RuntimeError("No embeddings extracted from official highlight video.")
    
    with open(full_match_json, "r") as f:
        full_data = json.load(f)
        
    df_full = pd.DataFrame(full_data)
    if df_full.empty or "V_spatial" not in df_full.columns:
        raise RuntimeError("Full-match visual_output.json has no V_spatial embeddings.")
    full_embeddings = np.array([
        np.array(e, dtype=float) / max(np.linalg.norm(e), 1e-12)
        for e in df_full['V_spatial']
        if isinstance(e, list) and len(e) > 0
    ])
    if len(full_embeddings) != len(df_full):
        raise RuntimeError("Some full-match rows have missing or invalid V_spatial embeddings.")
    
    sim_matrix = np.dot(full_embeddings, hl_embeddings.T)
    max_similarities = np.max(sim_matrix, axis=1)
    
    df_full['y_true'] = (max_similarities >= similarity_threshold).astype(int)
    output_path = f"{base_dir}/ground_truth.json"
    df_full.to_json(output_path, orient="records", indent=2)
    print(f"Saved ground truth to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", type=str, required=True, help="Folder name inside dataset/")
    args = parser.parse_args()
    auto_label_match(args.video_id)
