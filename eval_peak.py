"""Evaluate the new peak selector without changing existing evaluation code.

The strict metric imports evaluate.py's unchanged IoU >= 0.5 maximum matching.
The loose metric is explicitly separate: a predicted window may be within ten
seconds of a label, still with maximum one-to-one matching.
"""

import gc
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from zoneinfo import ZoneInfo

import numpy as np

from diagnostics.channel_signal import MATCHES, interval_labels, load_second_scores
from evaluate import calculate_metrics, compute_fused_scores, extract_highlight_windows, load_ground_truth, read_json
from fusion_channel.peak_fusion import (
    DEFAULT_ROLLING_SEC, DEFAULT_SMOOTH_SEC, MIN_PEAK_SEPARATION_SEC,
    peak_windows, prepare_scores,
)

K_VALUES = (10, 20, 40, 60)
SEEDS = tuple(range(100))
LOOSE_MARGIN_SEC = 10.0


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def loose_metrics(predictions, truth):
    """Maximum one-to-one matches after expanding each truth interval by 10 s."""
    expanded = [[start - LOOSE_MARGIN_SEC, end + LOOSE_MARGIN_SEC] for start, end in truth]
    # IoU is inappropriate after expansion: loose matching is overlap with this margin.
    edges = [[j for j, target in enumerate(expanded)
              if prediction[0] <= target[1] and prediction[1] >= target[0]]
             for prediction in predictions]
    assigned = [-1] * len(truth)
    def assign(i, seen):
        for j in edges[i]:
            if seen[j]:
                continue
            seen[j] = True
            if assigned[j] == -1 or assign(assigned[j], seen):
                assigned[j] = i
                return True
        return False
    tp = sum(assign(i, [False] * len(truth)) for i in range(len(predictions)))
    p = tp / len(predictions) if predictions else None
    r = tp / len(truth) if truth else None
    f1 = 0.0 if p is not None and r is not None and p + r == 0 else (2 * p * r / (p + r) if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f1, "true_positives": tp,
            "false_positives": len(predictions) - tp, "false_negatives": len(truth) - tp,
            "predicted_count": len(predictions), "ground_truth_count": len(truth)}


def as_pairs(items):
    return [[item["start"], item["end"]] for item in items]


def random_windows(duration, lengths, seed):
    """Use the method's exact selected length distribution and 30-second spacing."""
    rng, windows = random.Random(seed), []
    for length in lengths:
        for _ in range(1_000_000):
            start = rng.uniform(0.0, duration - length)
            end = start + length
            if all(end + MIN_PEAK_SEPARATION_SEC <= old_start or
                   start >= old_end + MIN_PEAK_SEPARATION_SEC
                   for old_start, old_end in windows):
                windows.append([start, end])
                break
        else:
            raise ValueError("Could not place random windows with the requested spacing.")
    return sorted(windows)


def even_windows(duration, lengths):
    """Evenly distribute the same lengths and required 30-second spacing."""
    needed = sum(lengths) + MIN_PEAK_SEPARATION_SEC * (len(lengths) - 1)
    if needed > duration:
        raise ValueError("Not enough duration for evenly spaced windows.")
    slack = (duration - needed) / (len(lengths) + 1)
    cursor, windows = slack, []
    for length in lengths:
        windows.append([cursor, cursor + length])
        cursor += length + MIN_PEAK_SEPARATION_SEC + slack
    return windows


def baselines(duration, lengths, truth):
    all_random = [random_windows(duration, lengths, seed) for seed in SEEDS]
    def summary(metric):
        result = {}
        for key in ("precision", "recall", "f1"):
            values = [metric(windows, truth)[key] for windows in all_random]
            result[key] = {"mean": mean(values), "std": pstdev(values)}
        return result
    return {"random": {"seed_count": len(SEEDS), "strict": summary(lambda p, t: calculate_metrics(p, t, 0.5)),
                       "loose": summary(loose_metrics)},
            "evenly_spaced": {"strict": calculate_metrics(even_windows(duration, lengths), truth, 0.5),
                               "loose": loose_metrics(even_windows(duration, lengths), truth)}}


def old_windows_all_k(directory):
    """Load old-engine inputs once, then reproduce its outputs for every k."""
    base = Path("data/processed") / directory
    visual, audio, speech = (read_json(base / name) for name in ("visual_output.json", "audio_output.json", "speech_output.json"))
    weights = {"visual": (1, 0, 0), "audio": (0, 1, 0), "speech": (0, 0, 1), "equal_fusion": (1/3, 1/3, 1/3)}
    output = {}
    for name, value in weights.items():
        stream = compute_fused_scores(visual, audio, speech, weights=value)
        output[name] = {
            k: extract_highlight_windows(stream, top_k=k, window_sec=6.0)
            for k in K_VALUES
        }
    del visual, audio, speech
    return output


def evaluate_label_set(match_id, directory, truth, description):
    raw = load_second_scores(directory)
    duration = float(len(next(iter(raw.values()))) - 1)
    prepared = {name: prepare_scores(values) for name, values in raw.items() if name != "current_fused"}
    variants = {"visual": prepared["visual"], "audio": prepared["audio"], "speech": prepared["speech"]}
    variants["equal_fusion"] = (prepared["visual"] + prepared["audio"] + prepared["speech"]) / 3.0
    ranked = {name: peak_windows(values, duration_sec=duration)[0] for name, values in variants.items()}
    old = old_windows_all_k(directory)
    rows = []
    for k in K_VALUES:
        for name, peaks in ranked.items():
            selected = peaks[:k]
            windows = as_pairs(selected)
            lengths = [end - start for start, end in windows]
            rows.append({"variant": name, "k": k, "peak_windows": selected,
                         "strict": calculate_metrics(windows, truth, 0.5), "loose_metric": loose_metrics(windows, truth),
                         "old_method": {"strict": calculate_metrics(old[name][k], truth, 0.5),
                                        "loose_metric": loose_metrics(old[name][k], truth)},
                         "baselines": baselines(duration, lengths, truth)})
    del raw, prepared, variants, ranked, old
    gc.collect()
    return {"description": description, "interval_count": len(truth), "duration_sec": duration, "rows": rows}


def sensitivity():
    """All fixed grid cells, reported together; not used to choose defaults."""
    raw = load_second_scores("cricket_full_match")
    truth = load_ground_truth("results/human_labels_cricket_match_1.json")
    rows = []
    for rolling in (60, 120, 240):
        for smoothing in (3, 5, 10):
            scores = sum(prepare_scores(raw[name], rolling, smoothing)
                         for name in ("visual", "audio", "speech")) / 3.0
            windows = as_pairs(peak_windows(scores, rolling, smoothing, len(scores) - 1)[0][:40])
            rows.append({"rolling_sec": rolling, "smoothing_sec": smoothing,
                         "k": 40, "strict": calculate_metrics(windows, truth, 0.5),
                         "loose_metric": loose_metrics(windows, truth)})
    return rows


def f(value):
    return "n/a" if value is None else f"{value:.4f}"


def markdown(report):
    lines = ["# Peak-based selection evaluation", "", f"Run date: {report['run']['date']}",
             f"Git commit: `{report['run']['git_commit']}`", "",
             "Strict = existing IoU >= 0.5 maximum one-to-one matching. Looser metric = a window overlaps a truth interval expanded by +/-10 seconds, also one-to-one.", ""]
    for label_name in ("human_verified", "pseudo_label"):
        lines += [f"## {label_name.replace('_', ' ')}", ""]
        for match_id, data in report["label_sets"].get(label_name, {}).items():
            lines += [f"### {match_id}: {data['description']}", "", "| Variant | k | New strict F1 | Old strict F1 | New loose F1 | Old loose F1 | Random strict F1 | Even strict F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for row in data["rows"]:
                random_f1 = row["baselines"]["random"]["strict"]["f1"]
                lines.append("| {v} | {k} | {ns} | {os} | {nl} | {ol} | {r} +/- {sd} | {e} |".format(
                    v=row["variant"], k=row["k"], ns=f(row["strict"]["f1"]), os=f(row["old_method"]["strict"]["f1"]),
                    nl=f(row["loose_metric"]["f1"]), ol=f(row["old_method"]["loose_metric"]["f1"]),
                    r=f(random_f1["mean"]), sd=f(random_f1["std"]), e=f(row["baselines"]["evenly_spaced"]["strict"]["f1"])))
            lines.append("")
    lines += ["## Sensitivity: equal fusion, human labels, k=40", "", "All combinations are shown; defaults were fixed before this run and no cell was selected using the human labels.", "", "| Rolling seconds | Smoothing seconds | Strict F1 | Loose F1 |", "|---:|---:|---:|---:|"]
    for row in report["sensitivity"]:
        lines.append(f"| {row['rolling_sec']} | {row['smoothing_sec']} | {f(row['strict']['f1'])} | {f(row['loose_metric']['f1'])} |")
    Path("results/peak_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    report = {"run": {"date": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(), "git_commit": commit()},
              "defaults": {"rolling_zscore_sec": DEFAULT_ROLLING_SEC, "smoothing_sec": DEFAULT_SMOOTH_SEC,
                           "minimum_peak_separation_sec": MIN_PEAK_SEPARATION_SEC, "k_values": list(K_VALUES)},
              "label_sets": {"human_verified": {}, "pseudo_label": {}}}
    human = load_ground_truth("results/human_labels_cricket_match_1.json")
    report["label_sets"]["human_verified"]["cricket_match_1"] = evaluate_label_set("cricket_match_1", "cricket_full_match", human, "human-verified, 20 intervals, single labeler")
    for match_id, (directory, truth_path) in MATCHES.items():
        truth = load_ground_truth(truth_path)
        report["label_sets"]["pseudo_label"][match_id] = evaluate_label_set(match_id, directory, truth, "pseudo-label")
    report["sensitivity"] = sensitivity()
    Path("results/peak_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown(report)
    print("Saved peak evaluation JSON: results/peak_evaluation.json")
    print("Saved peak evaluation Markdown: results/peak_evaluation.md")
    print(f"human_verified cricket_match_1: {len(human)} intervals")
    for match_id, data in report["label_sets"]["pseudo_label"].items():
        print(f"pseudo_label {match_id}: {data['interval_count']} intervals")


if __name__ == "__main__":
    main()
