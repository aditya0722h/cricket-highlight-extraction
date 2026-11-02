"""Learn fusion weights on one match and evaluate on a separate match."""

import argparse
import json
import math
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from evaluate import calculate_metrics, load_ground_truth, read_json, validate_channel
from fusion_channel.fusion_engine import (
    compute_fused_scores,
    extract_highlight_windows,
)


DEFAULT_WEIGHTS = (0.4, 0.4, 0.2)
CHANNELS = ("visual", "audio", "speech")


def load_match_features(paths):
    channels = {}
    for name in CHANNELS:
        data = read_json(paths[name])
        validate_channel(data, name)
        channels[name] = data

    stream = compute_fused_scores(
        channels["visual"],
        channels["audio"],
        channels["speech"],
        weights=DEFAULT_WEIGHTS,
    )
    timestamps = np.asarray([row["t"] for row in stream], dtype=np.float64)
    features = np.asarray(
        [[row["V"], row["A"], row["S"]] for row in stream],
        dtype=np.float64,
    )
    return channels, timestamps, features


def labels_for_intervals(timestamps, intervals):
    return np.asarray(
        [
            int(any(start <= timestamp < end for start, end in intervals))
            for timestamp in timestamps
        ],
        dtype=np.int64,
    )


def intervals_to_windows(weights, channels, top_k, window_sec):
    stream = compute_fused_scores(
        channels["visual"],
        channels["audio"],
        channels["speech"],
        weights=weights,
    )
    return extract_highlight_windows(
        stream, top_k=top_k, window_sec=window_sec
    )


def candidate_weights(resolution):
    for visual_units in range(resolution + 1):
        for audio_units in range(resolution - visual_units + 1):
            speech_units = resolution - visual_units - audio_units
            yield (
                visual_units / resolution,
                audio_units / resolution,
                speech_units / resolution,
            )


def select_grid_weights(
    channels, ground_truth, resolution, top_k, window_sec, overlap_iou
):
    best_weights = None
    best_metrics = None
    best_f1 = -math.inf
    for weights in candidate_weights(resolution):
        predictions = intervals_to_windows(
            weights, channels, top_k, window_sec
        )
        metrics = calculate_metrics(predictions, ground_truth, overlap_iou)
        score = metrics["f1"]
        score = -math.inf if score is None else score
        if score > best_f1:
            best_weights = weights
            best_metrics = metrics
            best_f1 = score
    return best_weights, best_metrics


def make_report(args):
    if args.train_match_id.strip().casefold() == args.test_match_id.strip().casefold():
        raise ValueError("Training and test match IDs must identify different matches.")

    train_channels, train_times, train_features = load_match_features(
        {
            "visual": args.train_visual,
            "audio": args.train_audio,
            "speech": args.train_speech,
        }
    )
    test_channels, test_times, test_features = load_match_features(
        {
            "visual": args.test_visual,
            "audio": args.test_audio,
            "speech": args.test_speech,
        }
    )
    train_ground_truth = load_ground_truth(args.train_ground_truth)
    test_ground_truth = load_ground_truth(args.test_ground_truth)
    train_labels = labels_for_intervals(train_times, train_ground_truth)
    if len(np.unique(train_labels)) != 2:
        raise ValueError(
            "Training ground truth must label at least one positive and one "
            "negative per-second feature row."
        )

    baseline_train = intervals_to_windows(
        DEFAULT_WEIGHTS, train_channels, args.top_k, args.window_sec
    )
    baseline_test = intervals_to_windows(
        DEFAULT_WEIGHTS, test_channels, args.top_k, args.window_sec
    )
    grid_weights, grid_train_metrics = select_grid_weights(
        train_channels,
        train_ground_truth,
        args.grid_resolution,
        args.top_k,
        args.window_sec,
        args.overlap_iou,
    )
    grid_test_predictions = intervals_to_windows(
        grid_weights, test_channels, args.top_k, args.window_sec
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    )
    classifier.fit(train_features, train_labels)
    positive_class_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    train_probabilities = classifier.predict_proba(train_features)[
        :, positive_class_index
    ]
    test_probabilities = classifier.predict_proba(test_features)[
        :, positive_class_index
    ]
    logistic_train_predictions = extract_highlight_windows(
        [
            {"t": float(timestamp), "fused_score": float(probability)}
            for timestamp, probability in zip(train_times, train_probabilities)
        ],
        top_k=args.top_k,
        window_sec=args.window_sec,
    )
    logistic_test_predictions = extract_highlight_windows(
        [
            {"t": float(timestamp), "fused_score": float(probability)}
            for timestamp, probability in zip(test_times, test_probabilities)
        ],
        top_k=args.top_k,
        window_sec=args.window_sec,
    )

    def metrics(predictions, truth):
        return calculate_metrics(predictions, truth, args.overlap_iou)

    return {
        "schema_version": 1,
        "sport": args.sport,
        "training_match_id": args.train_match_id,
        "test_match_id": args.test_match_id,
        "split_method": "entire matches; no within-match second-level split",
        "normalization": (
            "Per-channel min-max normalization is performed independently over "
            "each full match by the existing fusion engine."
        ),
        "overlap_rule": {
            "metric": "intersection over union (IoU)",
            "minimum_iou": args.overlap_iou,
            "matching": "maximum one-to-one interval matching",
        },
        "settings": {
            "top_k": args.top_k,
            "window_sec": args.window_sec,
            "grid_resolution": args.grid_resolution,
            "grid_step": 1 / args.grid_resolution,
        },
        "methods": {
            "hand_set_baseline": {
                "weights": list(DEFAULT_WEIGHTS),
                "train_metrics": metrics(baseline_train, train_ground_truth),
                "test_metrics": metrics(baseline_test, test_ground_truth),
                "test_predictions": baseline_test,
            },
            "simplex_grid_search": {
                "weights": list(grid_weights),
                "train_metrics": grid_train_metrics,
                "test_metrics": metrics(grid_test_predictions, test_ground_truth),
                "test_predictions": grid_test_predictions,
                "candidate_count": sum(
                    1 for _ in candidate_weights(args.grid_resolution)
                ),
            },
            "logistic_regression": {
                "train_metrics": metrics(
                    logistic_train_predictions, train_ground_truth
                ),
                "test_metrics": metrics(
                    logistic_test_predictions, test_ground_truth
                ),
                "test_predictions": logistic_test_predictions,
                "feature_order": ["visual", "audio", "speech"],
                "coefficients": classifier.coef_[0].tolist(),
                "intercept": float(classifier.intercept_[0]),
                "class_weight": "balanced",
            },
        },
        "training_data": {
            "feature_rows": int(len(train_times)),
            "positive_rows": int(np.sum(train_labels)),
            "negative_rows": int(len(train_labels) - np.sum(train_labels)),
            "ground_truth_intervals": len(train_ground_truth),
        },
        "test_data": {
            "feature_rows": int(len(test_times)),
            "ground_truth_intervals": len(test_ground_truth),
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=("cricket", "football"), required=True)
    parser.add_argument("--train-match-id", required=True)
    parser.add_argument("--test-match-id", required=True)
    for prefix in ("train", "test"):
        for channel in CHANNELS:
            parser.add_argument(
                f"--{prefix}-{channel}",
                required=True,
                help=f"{prefix.title()} match {channel} channel JSON.",
            )
        parser.add_argument(
            f"--{prefix}-ground-truth",
            required=True,
            help=f"{prefix.title()} match interval ground truth JSON.",
        )
    parser.add_argument("--grid-resolution", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--overlap-iou", type=float, default=0.5)
    parser.add_argument(
        "--output",
        default=os.path.join("results", "weight_learning.json"),
    )
    args = parser.parse_args(argv)
    if args.grid_resolution < 1:
        parser.error("--grid-resolution must be at least 1.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if not math.isfinite(args.window_sec) or args.window_sec <= 0:
        parser.error("--window-sec must be a positive finite number.")
    if not 0.0 < args.overlap_iou <= 1.0:
        parser.error("--overlap-iou must be greater than 0 and at most 1.")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        report = make_report(args)
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Weight learning failed: {error}", file=sys.stderr)
        return 1

    print(f"Saved match-held-out weight report: {args.output}")
    for name, result in report["methods"].items():
        metrics = result["test_metrics"]
        print(
            f"{name}: precision={metrics['precision']}, "
            f"recall={metrics['recall']}, f1={metrics['f1']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
