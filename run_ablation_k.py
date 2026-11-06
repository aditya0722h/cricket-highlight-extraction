"""Evaluate channel ablations at several top-k budgets on cricket match 1."""

import json
import os
import sys

from evaluate import calculate_metrics, load_ground_truth, validate_channel
from fusion_channel.fusion_engine import compute_fused_scores, extract_highlight_windows


TOP_K_VALUES = (3, 5, 10, 20, 50)
WINDOW_SECONDS = 6.0
MINIMUM_IOU = 0.5
VARIANTS = {
    "visual_only": (1.0, 0.0, 0.0),
    "audio_only": (0.0, 1.0, 0.0),
    "speech_only": (0.0, 0.0, 1.0),
    "full_fusion": (0.4, 0.4, 0.2),
}
CHANNEL_FILES = {
    "visual": "data/processed/cricket_full_match/visual_output.json",
    "audio": "data/processed/cricket_full_match/audio_output.json",
    "speech": "data/processed/cricket_full_match/speech_output.json",
}
GROUND_TRUTH_FILE = "results/cricket_match_1_ground_truth.json"


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def has_temporal_overlap(prediction, interval):
    return min(prediction[1], interval[1]) > max(prediction[0], interval[0])


def write_json(path, report):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def build_report():
    channels = {
        name: read_json(path)
        for name, path in CHANNEL_FILES.items()
    }
    for name, channel in channels.items():
        validate_channel(channel, name)
    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)

    variants = {}
    for name, weights in VARIANTS.items():
        stream = compute_fused_scores(
            channels["visual"],
            channels["audio"],
            channels["speech"],
            weights=weights,
        )
        by_k = {}
        for top_k in TOP_K_VALUES:
            windows = extract_highlight_windows(
                stream,
                top_k=top_k,
                window_sec=WINDOW_SECONDS,
            )
            metrics = calculate_metrics(windows, ground_truth, MINIMUM_IOU)
            hit_count = sum(
                any(
                    has_temporal_overlap(window, interval)
                    for interval in ground_truth
                )
                for window in windows
            )
            by_k[str(top_k)] = {
                "selected_window_count": len(windows),
                "predicted_highlights": windows,
                "metrics": metrics,
                "hit_count": hit_count,
                "hit_rate": hit_count / len(windows) if windows else None,
            }
        variants[name] = {
            "weights": list(weights),
            "by_k": by_k,
        }

    return {
        "schema_version": 1,
        "sport": "cricket",
        "match": "cricket_match_1",
        "evaluation_basis": (
            "Automatically generated visual-similarity pseudo-labels; "
            "not human annotations."
        ),
        "ground_truth_file": GROUND_TRUTH_FILE,
        "ground_truth_interval_count": len(ground_truth),
        "prediction_settings": {
            "top_k_values": list(TOP_K_VALUES),
            "window_sec": WINDOW_SECONDS,
            "minimum_iou_for_precision_recall_f1": MINIMUM_IOU,
            "hit_rate_rule": (
                "Selected windows with any positive-duration overlap with at "
                "least one pseudo-label interval, divided by selected windows."
            ),
        },
        "channel_files": CHANNEL_FILES,
        "variants": variants,
    }


def main():
    try:
        report = build_report()
        write_json("results/ablation_k.json", report)
        write_json("site/assets/ablation_k.json", report)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"K-budget ablation failed: {error}", file=sys.stderr)
        return 1

    print("Saved results/ablation_k.json")
    print("Saved site/assets/ablation_k.json")
    for name, variant in report["variants"].items():
        for top_k, result in variant["by_k"].items():
            metrics = result["metrics"]
            print(
                f"{name} k={top_k}: "
                f"precision={metrics['precision']}, "
                f"recall={metrics['recall']}, f1={metrics['f1']}, "
                f"hit_rate={result['hit_rate']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
