"""Evaluate existing channel outputs against interval or timestamp labels."""

import argparse
import json
import math
import os
import sys

from fusion_channel.fusion_engine import compute_fused_scores, extract_highlight_windows


VARIANTS = {
    "audio_only": (0.0, 1.0, 0.0),
    "speech_only": (0.0, 0.0, 1.0),
    "visual_only": (1.0, 0.0, 0.0),
    "full_fusion": (0.4, 0.4, 0.2),
}
CHANNEL_FIELDS = {
    "visual": "V_motion",
    "audio": "STE",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_channel(data, name):
    if not isinstance(data, list) or not data:
        raise ValueError(f"{name} channel must be a non-empty JSON list.")

    for index, row in enumerate(data):
        if not isinstance(row, dict) or "t" not in row:
            raise ValueError(f"{name} channel row {index} must be an object with 't'.")
        try:
            timestamp = float(row["t"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} channel row {index} has an invalid timestamp.") from error
        if not math.isfinite(timestamp):
            raise ValueError(f"{name} channel row {index} has a non-finite timestamp.")

    if name in CHANNEL_FIELDS and not any(
        CHANNEL_FIELDS[name] in row for row in data
    ):
        raise ValueError(
            f"{name} channel needs the '{CHANNEL_FIELDS[name]}' feature "
            "expected by fusion_channel/fusion_engine.py."
        )
    if name == "speech" and not any(
        "sentiment_score" in row or "keyword_hit" in row for row in data
    ):
        raise ValueError(
            "speech channel needs 'sentiment_score' or 'keyword_hit'."
        )


def validate_interval(interval, label):
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError(f"{label} must be a [start, end] pair.")
    try:
        start, end = float(interval[0]), float(interval[1])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain numeric timestamps.") from error
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"{label} must have finite timestamps with end > start.")
    return [start, end]


def binary_rows_to_intervals(rows):
    positives = []
    for index, row in enumerate(rows):
        if "t" not in row or "y_true" not in row:
            raise ValueError(
                f"Ground-truth row {index} needs both 't' and 'y_true'."
            )
        try:
            timestamp = float(row["t"])
            raw_label = float(row["y_true"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Ground-truth row {index} has invalid 't' or 'y_true'."
            ) from error
        if (
            not math.isfinite(timestamp)
            or not math.isfinite(raw_label)
            or raw_label not in (0.0, 1.0)
        ):
            raise ValueError(
                f"Ground-truth row {index} needs a finite 't' and binary 'y_true'."
            )
        label = int(raw_label)
        if label == 1:
            positives.append(timestamp)

    intervals = []
    start = end = None
    for timestamp in sorted(set(positives)):
        if start is None:
            start, end = timestamp, timestamp + 1.0
        elif timestamp <= end:
            end = max(end, timestamp + 1.0)
        else:
            intervals.append([start, end])
            start, end = timestamp, timestamp + 1.0
    if start is not None:
        intervals.append([start, end])
    return intervals


def load_ground_truth(path):
    data = read_json(path)
    if isinstance(data, dict) and "highlights" in data:
        raw_intervals = data["highlights"]
    elif isinstance(data, list) and all(
        isinstance(row, dict) and "y_true" in row for row in data
    ):
        return binary_rows_to_intervals(data)
    elif isinstance(data, list) and all(
        isinstance(interval, (list, tuple)) for interval in data
    ):
        raw_intervals = data
    else:
        raise ValueError(
            "Ground truth must be a list of [start, end] intervals, an object "
            "with a 'highlights' interval list, or timestamp rows with 't' and 'y_true'."
        )

    if not isinstance(raw_intervals, list):
        raise ValueError("Ground-truth 'highlights' must be a list.")
    return [
        validate_interval(interval, f"Ground-truth interval {index}")
        for index, interval in enumerate(raw_intervals)
    ]


def interval_iou(left, right):
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def match_intervals(predictions, ground_truth, threshold):
    neighbors = []
    for prediction in predictions:
        matches = [
            index
            for index, truth in enumerate(ground_truth)
            if interval_iou(prediction, truth) >= threshold
        ]
        neighbors.append(matches)

    matched_prediction_for_truth = [-1] * len(ground_truth)

    def assign(prediction_index, visited_truth):
        for truth_index in neighbors[prediction_index]:
            if visited_truth[truth_index]:
                continue
            visited_truth[truth_index] = True
            previous_prediction = matched_prediction_for_truth[truth_index]
            if previous_prediction == -1 or assign(
                previous_prediction, visited_truth
            ):
                matched_prediction_for_truth[truth_index] = prediction_index
                return True
        return False

    true_positives = sum(
        assign(index, [False] * len(ground_truth))
        for index in range(len(predictions))
    )
    return true_positives


def calculate_metrics(predictions, ground_truth, threshold):
    true_positives = match_intervals(predictions, ground_truth, threshold)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = (
        true_positives / len(predictions) if predictions else None
    )
    recall = (
        true_positives / len(ground_truth) if ground_truth else None
    )
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_count": len(predictions),
        "ground_truth_count": len(ground_truth),
    }


def build_report(args):
    channel_paths = {
        "visual": args.visual,
        "audio": args.audio,
        "speech": args.speech,
    }
    channels = {}
    channel_errors = {}
    for name, path in channel_paths.items():
        if path is None:
            channel_errors[name] = "No channel input file was supplied."
            continue
        try:
            data = read_json(path)
            validate_channel(data, name)
            channels[name] = data
        except (OSError, json.JSONDecodeError, ValueError) as error:
            channel_errors[name] = str(error)

    ground_truth = None
    ground_truth_error = None
    if args.ground_truth:
        try:
            ground_truth = load_ground_truth(args.ground_truth)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            ground_truth_error = str(error)

    variants = {}
    for name, weights in VARIANTS.items():
        needed = {
            "audio_only": ("audio",),
            "speech_only": ("speech",),
            "visual_only": ("visual",),
            "full_fusion": ("visual", "audio", "speech"),
        }[name]
        missing = [channel for channel in needed if channel not in channels]
        if missing:
            variants[name] = {
                "status": "unavailable",
                "reason": "; ".join(
                    f"{channel}: {channel_errors.get(channel, 'input unavailable')}"
                    for channel in missing
                ),
                "predicted_highlights": None,
                "metrics": None,
            }
            continue

        stream = compute_fused_scores(
            channels.get("visual", []),
            channels.get("audio", []),
            channels.get("speech", []),
            weights=weights,
        )
        windows = extract_highlight_windows(
            stream, top_k=args.top_k, window_sec=args.window_sec
        )
        variant = {
            "status": "evaluated" if ground_truth is not None else "metrics_unavailable",
            "predicted_highlights": windows,
            "metrics": (
                calculate_metrics(windows, ground_truth, args.overlap_iou)
                if ground_truth is not None
                else None
            ),
        }
        if ground_truth is None:
            variant["reason"] = (
                f"Ground truth was not supplied; predictions are not metrics."
                if ground_truth_error is None
                else f"Ground truth could not be loaded: {ground_truth_error}"
            )
        variants[name] = variant

    all_metrics_available = ground_truth is not None and all(
        item["status"] == "evaluated" for item in variants.values()
    )
    return {
        "schema_version": 1,
        "sport": args.sport,
        "status": "evaluated" if all_metrics_available else "metrics_unavailable",
        "ground_truth_file": args.ground_truth,
        "ground_truth_interval_count": (
            len(ground_truth) if ground_truth is not None else None
        ),
        "overlap_rule": {
            "metric": "intersection over union (IoU)",
            "minimum_iou": args.overlap_iou,
            "matching": "maximum one-to-one interval matching",
            "timestamp_labels": "each positive row covers [t, t + 1) seconds",
        },
        "prediction_settings": {
            "top_k": args.top_k,
            "window_sec": args.window_sec,
            "weights": {
                name: list(weights) for name, weights in VARIANTS.items()
            },
        },
        "channel_files": channel_paths,
        "variants": variants,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the existing fusion engine against highlight annotations."
    )
    parser.add_argument("--sport", required=True, choices=("cricket", "football"))
    parser.add_argument(
        "--ground-truth",
        help="JSON interval list, {'highlights': [...]}, or rows with t/y_true.",
    )
    parser.add_argument("--visual", help="Visual channel JSON (expects V_motion).")
    parser.add_argument("--audio", help="Audio channel JSON (expects STE).")
    parser.add_argument(
        "--speech",
        help="Speech channel JSON (expects sentiment_score and/or keyword_hit).",
    )
    parser.add_argument("--overlap-iou", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--output", help="Output JSON path (default: results/<sport>.json).")
    args = parser.parse_args(argv)
    if not 0.0 < args.overlap_iou <= 1.0:
        parser.error("--overlap-iou must be greater than 0 and at most 1.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if not math.isfinite(args.window_sec) or args.window_sec <= 0:
        parser.error("--window-sec must be a positive finite number.")
    if args.ground_truth and not os.path.isfile(args.ground_truth):
        parser.error(f"Ground-truth file does not exist: {args.ground_truth}")
    return args


def main(argv=None):
    args = parse_args(argv)
    output_path = args.output or os.path.join("results", f"{args.sport}.json")
    try:
        report = build_report(args)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1

    print(f"Saved {report['status']} report: {output_path}")
    for name, variant in report["variants"].items():
        if variant["metrics"] is None:
            print(f"{name}: metrics unavailable ({variant['reason']})")
        else:
            metrics = variant["metrics"]
            print(
                f"{name}: precision={metrics['precision']}, "
                f"recall={metrics['recall']}, f1={metrics['f1']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
