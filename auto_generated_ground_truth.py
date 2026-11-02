"""Locate highlight-video segments in a full match using visual embeddings.

The resulting intervals are pseudo ground truth. Visual similarities should be
reviewed manually before using them as evaluation labels.
"""

import argparse
import json
import math
import os
import sys

import numpy as np


def load_embeddings(path, label):
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} visual features must be a non-empty JSON list.")

    timestamps = []
    embeddings = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "t" not in row:
            raise ValueError(f"{label} row {index} must contain a timestamp.")
        values = row.get("V_spatial")
        try:
            timestamp = float(row["t"])
            embedding = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} row {index} contains invalid data.") from error
        if (
            not math.isfinite(timestamp)
            or embedding.ndim != 1
            or embedding.size == 0
            or not np.isfinite(embedding).all()
        ):
            raise ValueError(f"{label} row {index} has an invalid timestamp or embedding.")
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            raise ValueError(f"{label} row {index} has a zero-length embedding.")
        timestamps.append(timestamp)
        embeddings.append(embedding / norm)

    order = np.argsort(timestamps)
    return (
        np.asarray(timestamps, dtype=np.float64)[order],
        np.stack(embeddings)[order],
    )


def best_frame_matches(full_embeddings, highlight_embeddings, batch_size):
    best_indices = np.empty(len(highlight_embeddings), dtype=np.int64)
    best_scores = np.empty(len(highlight_embeddings), dtype=np.float32)
    full_transposed = full_embeddings.T
    for start in range(0, len(highlight_embeddings), batch_size):
        end = min(start + batch_size, len(highlight_embeddings))
        similarities = highlight_embeddings[start:end] @ full_transposed
        best_indices[start:end] = np.argmax(similarities, axis=1)
        best_scores[start:end] = np.max(similarities, axis=1)
    return best_indices, best_scores


def candidate_segments(
    highlight_times,
    matched_full_times,
    similarities,
    threshold,
    min_frames,
):
    groups = []
    group_start = 0
    for index in range(1, len(highlight_times)):
        highlight_gap = highlight_times[index] - highlight_times[index - 1]
        source_gap = matched_full_times[index] - matched_full_times[index - 1]
        continuous = (
            0 < highlight_gap <= 1.5
            and -0.5 <= source_gap <= 2.5
        )
        if not continuous:
            groups.append((group_start, index))
            group_start = index
    groups.append((group_start, len(highlight_times)))

    matched = []
    unmatched = []
    for start, end in groups:
        scores = similarities[start:end]
        source_times = matched_full_times[start:end]
        is_match = (
            end - start >= min_frames
            and float(np.median(scores)) >= threshold
            and float(np.mean(scores >= threshold)) >= 0.6
        )
        candidate = {
            "highlight_start": float(highlight_times[start]),
            "highlight_end": float(highlight_times[end - 1] + 1.0),
            "frame_count": end - start,
            "median_similarity": round(float(np.median(scores)), 5),
        }
        if is_match:
            source_start = max(0.0, float(np.min(source_times)))
            source_end = float(np.max(source_times)) + 1.0
            candidate["full_match_interval"] = [source_start, source_end]
            matched.append(candidate)
        else:
            unmatched.append(candidate)
    return matched, unmatched


def merge_nearby_intervals(intervals, gap_seconds=1.0):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap_seconds:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [[round(start, 2), round(end, 2)] for start, end in merged]


def locate_highlights(
    full_visual_path,
    highlight_visual_path,
    threshold=0.92,
    min_frames=3,
    batch_size=64,
):
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be greater than zero and at most one.")
    if min_frames < 1 or batch_size < 1:
        raise ValueError("min_frames and batch_size must be positive.")

    full_times, full_embeddings = load_embeddings(full_visual_path, "Full-match")
    highlight_times, highlight_embeddings = load_embeddings(
        highlight_visual_path, "Highlight"
    )
    if full_embeddings.shape[1] != highlight_embeddings.shape[1]:
        raise ValueError("Full-match and highlight embedding dimensions differ.")

    indices, similarities = best_frame_matches(
        full_embeddings, highlight_embeddings, batch_size
    )
    matched_times = full_times[indices]
    matched, unmatched = candidate_segments(
        highlight_times,
        matched_times,
        similarities,
        threshold,
        min_frames,
    )
    intervals = merge_nearby_intervals(
        [item["full_match_interval"] for item in matched]
    )
    report = {
        "label_type": "pseudo_ground_truth_from_official_highlights",
        "matching_method": (
            "1-fps MobileNetV3 cosine nearest-neighbor matching; candidate "
            "highlight segments are consecutive frames mapping to a continuous "
            "full-match timeline."
        ),
        "similarity_threshold": threshold,
        "minimum_consecutive_frames": min_frames,
        "matched_candidate_clip_count": len(matched),
        "unmatched_candidate_clip_count": len(unmatched),
        "unique_full_match_interval_count": len(intervals),
        "matched_candidates": matched,
        "unmatched_candidates": unmatched,
        "intervals": intervals,
    }
    return intervals, report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-visual-json", required=True)
    parser.add_argument("--highlight-visual-json", required=True)
    parser.add_argument("--output", required=True, help="Writes [[start, end], ...].")
    parser.add_argument(
        "--report",
        help="Metadata/report path (default: <output stem>_report.json).",
    )
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report_path = args.report or os.path.splitext(args.output)[0] + "_report.json"
    try:
        intervals, report = locate_highlights(
            args.full_visual_json,
            args.highlight_visual_json,
            threshold=args.threshold,
            min_frames=args.min_frames,
            batch_size=args.batch_size,
        )
        for path, payload in (
            (args.output, intervals),
            (report_path, report),
        ):
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Ground-truth generation failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Matched candidate clips: {report['matched_candidate_clip_count']}; "
        f"unmatched: {report['unmatched_candidate_clip_count']}; "
        f"unique full-match intervals: {len(intervals)}."
    )
    print(f"Saved pseudo ground truth: {args.output}")
    print(f"Saved match report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
