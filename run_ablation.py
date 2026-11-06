"""Compare single-channel and fused ranking against cricket match 1 pseudo-labels."""

import argparse
import json
import math
import os
import sys

from evaluate import calculate_metrics, load_ground_truth, validate_channel
from fusion_channel.fusion_engine import compute_fused_scores, extract_highlight_windows


VARIANTS = {
    "visual_only": (1.0, 0.0, 0.0),
    "audio_only": (0.0, 1.0, 0.0),
    "speech_only": (0.0, 0.0, 1.0),
    "full_fusion": (0.4, 0.4, 0.2),
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visual",
        default="data/processed/cricket_full_match/visual_output.json",
    )
    parser.add_argument(
        "--audio",
        default="data/processed/cricket_full_match/audio_output.json",
    )
    parser.add_argument(
        "--speech",
        default="data/processed/cricket_full_match/speech_output.json",
    )
    parser.add_argument(
        "--ground-truth",
        default="results/cricket_match_1_ground_truth.json",
    )
    parser.add_argument("--output", default="results/ablation.json")
    parser.add_argument("--site-output", default="site/assets/ablation.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--overlap-iou", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not math.isfinite(args.window_sec) or args.window_sec <= 0:
        parser.error("--window-sec must be a positive finite number")
    if not math.isfinite(args.overlap_iou) or not 0 < args.overlap_iou <= 1:
        parser.error("--overlap-iou must be greater than 0 and at most 1")
    return args


def build_report(args):
    channel_paths = {
        "visual": args.visual,
        "audio": args.audio,
        "speech": args.speech,
    }
    channels = {
        name: read_json(path)
        for name, path in channel_paths.items()
    }
    for name, channel in channels.items():
        validate_channel(channel, name)
    ground_truth = load_ground_truth(args.ground_truth)

    variants = {}
    for name, weights in VARIANTS.items():
        scores = compute_fused_scores(
            channels["visual"],
            channels["audio"],
            channels["speech"],
            weights=weights,
        )
        windows = extract_highlight_windows(
            scores,
            top_k=args.top_k,
            window_sec=args.window_sec,
        )
        variants[name] = {
            "weights": list(weights),
            "predicted_highlights": windows,
            "metrics": calculate_metrics(
                windows, ground_truth, args.overlap_iou
            ),
        }

    return {
        "schema_version": 1,
        "sport": "cricket",
        "match": "cricket_match_1",
        "evaluation_basis": "Automatically generated visual-similarity pseudo-labels; not human annotations.",
        "ground_truth_file": args.ground_truth,
        "ground_truth_interval_count": len(ground_truth),
        "prediction_settings": {
            "top_k": args.top_k,
            "window_sec": args.window_sec,
            "minimum_iou": args.overlap_iou,
        },
        "channel_files": channel_paths,
        "variants": variants,
    }


def write_json(path, report):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = build_report(args)
        write_json(args.output, report)
        if os.path.abspath(args.site_output) != os.path.abspath(args.output):
            write_json(args.site_output, report)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Ablation failed: {error}", file=sys.stderr)
        return 1

    print(f"Saved ablation report: {args.output}")
    print(f"Saved site table data: {args.site_output}")
    for name, variant in report["variants"].items():
        metrics = variant["metrics"]
        print(
            f"{name}: precision={metrics['precision']}, "
            f"recall={metrics['recall']}, f1={metrics['f1']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
