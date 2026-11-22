"""Evaluate existing channel outputs at several selection budgets.

This script deliberately reuses the project's fusion and interval-matching code.
It reads existing JSON files only; it never runs a channel extractor or creates
pseudo labels.
"""

import argparse
import json
import math
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from zoneinfo import ZoneInfo

from evaluate import (
    VARIANTS,
    calculate_metrics,
    load_ground_truth,
    read_json,
    validate_channel,
)
from fusion_channel.fusion_engine import compute_fused_scores, extract_highlight_windows


MATCHES = {
    "cricket_match_1": {
        "sport": "cricket",
        "directory": "cricket_full_match",
        "ground_truth": "results/cricket_match_1_ground_truth.json",
    },
    "cricket_match_2": {
        "sport": "cricket",
        "directory": "cricket_full_match_2",
        "ground_truth": "results/cricket_match_2_ground_truth.json",
    },
    "football_match_1": {
        "sport": "football",
        "directory": "football_full_match",
        "ground_truth": "results/football_match_1_ground_truth.json",
    },
}
K_VALUES = (3, 10, 20, 40, 60, 80)
WINDOW_SEC = 6.0
MIN_GAP_SEC = 3.0  # The existing fusion selector's fixed exclusion gap.
RANDOM_SEEDS = tuple(range(100))


def git_value(*args):
    """Return a git value when this checkout has git metadata, else None."""
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_channels(match):
    """Load and validate precomputed full-match channels without modifying them."""
    directory = Path("data/processed") / match["directory"]
    paths = {
        "visual": directory / "visual_output.json",
        "audio": directory / "audio_output.json",
        "speech": directory / "speech_output.json",
    }
    channels = {}
    for name, path in paths.items():
        data = read_json(path)
        validate_channel(data, name)
        channels[name] = data
    return channels, {name: str(path) for name, path in paths.items()}


def stream_duration(channels):
    """Use the furthest existing channel timestamp as the observable duration."""
    return max(float(row["t"]) for rows in channels.values() for row in rows)


def random_windows(duration, k, seed):
    """Draw k six-second windows with the fusion selector's 3-second separation.

    Candidate starts are continuous uniform samples in [0, duration-window_sec].
    Rejection sampling makes the baseline reproducible and prevents a random
    baseline from receiving more densely packed opportunities than the model.
    """
    if duration < WINDOW_SEC:
        raise ValueError("Channel timestamps are shorter than one evaluation window.")
    rng = random.Random(seed)
    windows = []
    maximum_attempts = 1_000_000
    for _ in range(k):
        for _attempt in range(maximum_attempts):
            start = rng.uniform(0.0, duration - WINDOW_SEC)
            end = start + WINDOW_SEC
            if all(end < old_start - MIN_GAP_SEC or start > old_end + MIN_GAP_SEC
                   for old_start, old_end in windows):
                windows.append((start, end))
                break
        else:
            raise ValueError(
                f"Could not draw {k} separated windows within {duration:.2f} seconds."
            )
    return sorted(windows)


def evenly_spaced_windows(duration, k):
    """Place k valid six-second windows uniformly from the start to the end."""
    if duration < WINDOW_SEC:
        raise ValueError("Channel timestamps are shorter than one evaluation window.")
    available = duration - WINDOW_SEC
    if k == 1:
        starts = [available / 2.0]
    else:
        starts = [available * index / (k - 1) for index in range(k)]
    windows = [(start, start + WINDOW_SEC) for start in starts]
    if any(
        right[0] < left[1] + MIN_GAP_SEC
        for left, right in zip(windows, windows[1:])
    ):
        raise ValueError(
            f"Cannot evenly place {k} windows with a {MIN_GAP_SEC}-second gap "
            f"within {duration:.2f} seconds."
        )
    return windows


def summarize_random(ground_truth, duration, k):
    """Calculate actual metrics for each seed, then return population mean/std."""
    all_metrics = [
        calculate_metrics(random_windows(duration, k, seed), ground_truth, 0.5)
        for seed in RANDOM_SEEDS
    ]
    summary = {"seed_count": len(RANDOM_SEEDS), "seeds": list(RANDOM_SEEDS)}
    for key in ("precision", "recall", "f1"):
        values = [item[key] for item in all_metrics]
        summary[key] = {"mean": mean(values), "std": pstdev(values)}
    return summary


def evaluate_label_set(channels, ground_truth, duration):
    """Evaluate every modality and baseline for a supplied interval label set."""
    variant_streams = {
        name: compute_fused_scores(
            channels["visual"], channels["audio"], channels["speech"], weights=weights
        )
        for name, weights in VARIANTS.items()
    }
    rows = []
    for k in K_VALUES:
        random_summary = summarize_random(ground_truth, duration, k)
        even_metrics = calculate_metrics(evenly_spaced_windows(duration, k), ground_truth, 0.5)
        for name, stream in variant_streams.items():
            windows = extract_highlight_windows(stream, top_k=k, window_sec=WINDOW_SEC)
            metrics = calculate_metrics(windows, ground_truth, 0.5)
            rows.append({
                "variant": name,
                "k": k,
                "predicted_highlights": windows,
                "metrics": metrics,
                "random_baseline": random_summary,
                "even_spacing_baseline": even_metrics,
            })
    return rows


def load_human_labels(path):
    """Accept only explicitly human-marked label files created by label_tool.py."""
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("labeled_by") != "human":
        raise ValueError("Human label file must be an object with labeled_by set to 'human'.")
    if "highlights" not in payload:
        raise ValueError("Human label file must contain a highlights list.")
    # Reuse evaluate.py's established interval parsing and validation.
    return load_ground_truth(path)


def markdown_number(value):
    return "n/a" if value is None else f"{value:.4f}"


def write_markdown(report, output_path):
    lines = [
        "# Evaluation at k",
        "",
        f"Run date: {report['run']['date']}  ",
        f"Git commit: `{report['run']['git_commit'] or 'unavailable'}`",
        "",
        "Model metrics use the project's IoU ≥ 0.5 maximum one-to-one matching. "
        "Random results are the mean over 100 deterministic seeds; `±` is population standard deviation.",
        "",
    ]
    for label_name, label_report in report["label_sets"].items():
        descriptor = label_report["description"]
        lines.extend([f"## {label_name}: {descriptor}", ""])
        if label_name != "human_verified":
            lines.extend([
                "| Match | Variant | k | Precision | Recall | F1 | Random-baseline precision |",
                "|---|---|---:|---:|---:|---:|---:|",
            ])
        for match_id, match_report in label_report["matches"].items():
            if label_name == "human_verified":
                lines.extend([
                    "",
                    f"### {match_id}: human-verified, "
                    f"{match_report['ground_truth_interval_count']} intervals, single labeler",
                    "",
                    "| Match | Variant | k | Precision | Recall | F1 | Random-baseline precision |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ])
            for row in match_report["rows"]:
                random_precision = row["random_baseline"]["precision"]
                random_cell = (
                    f"{random_precision['mean']:.4f} ± {random_precision['std']:.4f}"
                )
                metrics = row["metrics"]
                lines.append(
                    "| {match} | {variant} | {k} | {precision} | {recall} | {f1} | {random} |".format(
                        match=match_id,
                        variant=row["variant"],
                        k=row["k"],
                        precision=markdown_number(metrics["precision"]),
                        recall=markdown_number(metrics["recall"]),
                        f1=markdown_number(metrics["f1"]),
                        random=random_cell,
                    )
                )
        lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default="results/eval_at_k.json")
    parser.add_argument("--output-md", default="results/eval_at_k.md")
    parser.add_argument(
        "--human-label-dir", default="results",
        help="Directory searched for human_labels_<match>.json files.",
    )
    args = parser.parse_args(argv)

    run_date = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    report = {
        "schema_version": 1,
        "run": {"date": run_date, "git_commit": git_value("rev-parse", "HEAD")},
        "settings": {
            "k_values": list(K_VALUES), "window_sec": WINDOW_SEC,
            "minimum_gap_sec": MIN_GAP_SEC, "iou_threshold": 0.5,
            "matching": "maximum one-to-one interval matching",
            "random_seed_count": len(RANDOM_SEEDS),
        },
        "label_sets": {"pseudo_labels": {"description": "pseudo-labels", "matches": {}}},
    }
    human_matches = {}
    for match_id, match in MATCHES.items():
        channels, channel_paths = load_channels(match)
        duration = stream_duration(channels)
        pseudo_truth = load_ground_truth(match["ground_truth"])
        report["label_sets"]["pseudo_labels"]["matches"][match_id] = {
            "ground_truth_file": match["ground_truth"],
            "ground_truth_interval_count": len(pseudo_truth),
            "channel_files": channel_paths,
            "observable_duration_sec": duration,
            "rows": evaluate_label_set(channels, pseudo_truth, duration),
        }

        human_path = Path(args.human_label_dir) / f"human_labels_{match_id}.json"
        if human_path.is_file():
            human_truth = load_human_labels(human_path)
            if not human_truth:
                print(
                    f"Skipping empty human label file: {human_path}. "
                    "Enter at least one manual interval before human evaluation."
                )
                continue
            human_matches[match_id] = {
                "ground_truth_file": str(human_path),
                "ground_truth_interval_count": len(human_truth),
                "channel_files": channel_paths,
                "observable_duration_sec": duration,
                "rows": evaluate_label_set(channels, human_truth, duration),
            }
    if human_matches:
        report["label_sets"]["human_verified"] = {
            "description": "human-verified, single labeler",
            "matches": human_matches,
        }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, args.output_md)
    print(f"Saved evaluation JSON: {args.output_json}")
    print(f"Saved evaluation Markdown: {args.output_md}")
    for label_name, label_report in report["label_sets"].items():
        for match_id, match_report in label_report["matches"].items():
            print(f"{label_name} {match_id}: {match_report['ground_truth_interval_count']} intervals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
