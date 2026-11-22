"""Fixed-seed interval bootstrap confidence intervals for both human matches.

Intervals, rather than seconds, are sampled with replacement 1,000 times. The
channel scores and peak predictions remain fixed. Seed 20261122 was selected
before execution. Pooled results resample each match independently, then join
them with a timeline offset so one match cannot match another's intervals.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from confirm_match2 import build_peak_data
from diagnostics.channel_signal import interval_labels, load_second_scores, metrics
from evaluate import calculate_metrics, load_ground_truth

SEED = 20261122
RESAMPLES = 1000
MATCHES = {
    "cricket_match_1": ("cricket_full_match", "results/human_labels_cricket_match_1.json"),
    "cricket_match_2": ("cricket_full_match_2", "results/human_labels_cricket_match_2.json"),
}


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def interval_sample(intervals, rng):
    indices = rng.integers(0, len(intervals), len(intervals))
    return [intervals[int(index)] for index in indices]


def ci(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def point_and_ci(point, samples):
    return {"value": float(point), "ci_95": ci(samples)}


def fixed_match_data(match_id, directory, label_path):
    truth = load_ground_truth(label_path)
    raw = load_second_scores(directory)
    _, ranked, _ = build_peak_data(directory)
    predictions = {
        "audio": [[item["start"], item["end"]] for item in ranked["audio"][:10]],
        "equal_fusion": [[item["start"], item["end"]] for item in ranked["equal_fusion"][:10]],
    }
    return {"match_id": match_id, "truth": truth, "raw": raw, "predictions": predictions,
            "duration": len(next(iter(raw.values()))) - 1}


def calculate_match(data, rng):
    truth = data["truth"]
    labels = interval_labels(truth, len(next(iter(data["raw"].values()))))
    point_auc = {name: metrics(data["raw"][name], labels)["roc_auc"]
                 for name in ("visual", "audio", "speech")}
    point_f1 = {name: calculate_metrics(prediction, truth, 0.5)["f1"]
                for name, prediction in data["predictions"].items()}
    samples_auc = {name: [] for name in point_auc}
    samples_f1 = {name: [] for name in point_f1}
    for _ in range(RESAMPLES):
        sampled = interval_sample(truth, rng)
        sample_labels = interval_labels(sampled, len(labels))
        for name in samples_auc:
            samples_auc[name].append(metrics(data["raw"][name], sample_labels)["roc_auc"])
        for name, prediction in data["predictions"].items():
            samples_f1[name].append(calculate_metrics(prediction, sampled, 0.5)["f1"])
    return {"n_intervals": len(truth),
            "roc_auc": {name: point_and_ci(point_auc[name], samples_auc[name]) for name in point_auc},
            "strict_f1_k10": {name: point_and_ci(point_f1[name], samples_f1[name]) for name in point_f1}}


def calculate_pooled(data_by_match, rng):
    """Bootstrap each match independently, then compute one pooled statistic."""
    originals = {name: data["truth"] for name, data in data_by_match.items()}
    raw_scores = {channel: np.concatenate([data["raw"][channel] for data in data_by_match.values()])
                  for channel in ("visual", "audio", "speech")}
    def combine_truth(sampled):
        offset, combined = 0.0, []
        for name, data in data_by_match.items():
            combined.extend([[start + offset, end + offset] for start, end in sampled[name]])
            offset += data["duration"] + 1000.0
        return combined
    def combine_predictions(variant):
        offset, combined = 0.0, []
        for data in data_by_match.values():
            combined.extend([[start + offset, end + offset] for start, end in data["predictions"][variant]])
            offset += data["duration"] + 1000.0
        return combined
    original_labels = np.concatenate([
        interval_labels(data["truth"], len(next(iter(data["raw"].values()))))
        for data in data_by_match.values()
    ])
    point_auc = {name: metrics(raw_scores[name], original_labels)["roc_auc"] for name in raw_scores}
    point_f1 = {name: calculate_metrics(combine_predictions(name), combine_truth(originals), 0.5)["f1"]
                for name in ("audio", "equal_fusion")}
    samples_auc = {name: [] for name in point_auc}
    samples_f1 = {name: [] for name in point_f1}
    for _ in range(RESAMPLES):
        sampled = {name: interval_sample(data["truth"], rng) for name, data in data_by_match.items()}
        labels = np.concatenate([
            interval_labels(sampled[name], len(next(iter(data["raw"].values()))))
            for name, data in data_by_match.items()
        ])
        for name in samples_auc:
            samples_auc[name].append(metrics(raw_scores[name], labels)["roc_auc"])
        truth = combine_truth(sampled)
        for name in samples_f1:
            samples_f1[name].append(calculate_metrics(combine_predictions(name), truth, 0.5)["f1"])
    return {"n_intervals": sum(len(value) for value in originals.values()),
            "roc_auc": {name: point_and_ci(point_auc[name], samples_auc[name]) for name in point_auc},
            "strict_f1_k10": {name: point_and_ci(point_f1[name], samples_f1[name]) for name in point_f1}}


def formatted(result):
    return f"{result['value']:.4f} [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]"


def write_markdown(report):
    lines = ["# Bootstrap confidence intervals", "", f"Run date: {report['run']['date']}",
             f"Git commit: `{report['run']['git_commit']}`", "",
             f"Interval bootstrap: {RESAMPLES} resamples, fixed seed {SEED}. Predictions are fixed; labeled intervals are sampled with replacement.", ""]
    for name, data in report["results"].items():
        lines += [f"## {name}: n={data['n_intervals']} intervals", "", "| Metric | Result |", "|---|---:|"]
        for channel, result in data["roc_auc"].items():
            lines.append(f"| ROC-AUC {channel} | {formatted(result)} |")
        for variant, result in data["strict_f1_k10"].items():
            lines.append(f"| Strict F1 at k=10 {variant} | {formatted(result)} |")
        lines.append("")
    Path("results/bootstrap_ci.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    rng = np.random.default_rng(SEED)
    data = {name: fixed_match_data(name, *paths) for name, paths in MATCHES.items()}
    report = {"run": {"date": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(), "git_commit": git_commit()},
              "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "unit": "human intervals with replacement"},
              "results": {}}
    for name, match_data in data.items():
        report["results"][name] = calculate_match(match_data, rng)
    report["results"]["pooled_both_matches"] = calculate_pooled(data, rng)
    Path("results/bootstrap_ci.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print("Saved bootstrap JSON: results/bootstrap_ci.json")
    print("Saved bootstrap Markdown: results/bootstrap_ci.md")
    for name, result in report["results"].items():
        print(f"{name}: n={result['n_intervals']} intervals")


if __name__ == "__main__":
    main()
