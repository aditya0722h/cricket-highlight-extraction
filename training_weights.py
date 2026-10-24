import sys
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground_truth.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_path):
        print(f"[ERROR] Ground truth file not found: {args.gt_path}")
        sys.exit(1)
        
    with open(args.gt_path, 'r') as f:
        gt_data = json.load(f)
        
    gt_dir = os.path.basename(os.path.dirname(os.path.abspath(args.gt_path)))
    video_name = gt_dir or "default"
    config_path = f"model_config_{video_name}.json"
    
    config = {"motion_threshold": 5000, "version": "v1.0"}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    labeled_rows = [row for row in gt_data if "y_true" in row]
    total_labeled = len(labeled_rows)
    highlight_count = sum(1 for row in labeled_rows if int(row.get("y_true", 0)) == 1)
    action_density = (highlight_count / total_labeled) if total_labeled else 0.0
    current_threshold = float(config.get("motion_threshold", 5000))
    if action_density > 0.25:
        config["motion_threshold"] = max(3000, current_threshold - 300)
    else:
        config["motion_threshold"] = min(8000, current_threshold + 300)
        
    config["training_examples"] = total_labeled
    config["positive_examples"] = highlight_count
    config["action_density"] = round(float(action_density), 4)
    config["version"] = f"v_updated_{video_name}"

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[SUCCESS] Configuration saved to {config_path}")

if __name__ == "__main__":
    main()
