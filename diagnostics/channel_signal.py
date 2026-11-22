"""Measure channel signal against human and pseudo interval labels.

Reads existing channel JSONs only. Scores are aligned to one-second bins; a
missing score is zero, matching the existing fusion engine's missing-data rule.
"""

import gc
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Allow `python diagnostics/channel_signal.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import load_ground_truth, read_json
from fusion_channel.fusion_engine import compute_fused_scores

MATCHES = {
    "cricket_match_1": ("cricket_full_match", "results/cricket_match_1_ground_truth.json"),
    "cricket_match_2": ("cricket_full_match_2", "results/cricket_match_2_ground_truth.json"),
    "football_match_1": ("football_full_match", "results/football_match_1_ground_truth.json"),
}


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def interval_labels(intervals, length):
    labels = np.zeros(length, dtype=int)
    for start, end in intervals:
        labels[max(0, int(np.floor(start))):min(length, int(np.ceil(end)))] = 1
    return labels


def _seconds(rows, value):
    mapped = {}
    for row in rows:
        mapped[int(round(float(row["t"])))] = float(value(row))
    return mapped


def load_second_scores(directory):
    """Load three existing streams, return dense one-second raw and fused scores."""
    base = Path("data/processed") / directory
    visual = read_json(base / "visual_output.json")
    audio = read_json(base / "audio_output.json")
    speech = read_json(base / "speech_output.json")
    fused = compute_fused_scores(visual, audio, speech)
    sources = {
        "visual": _seconds(visual, lambda row: row.get("V_motion", 0.0)),
        "audio": _seconds(audio, lambda row: row.get("STE", 0.0)),
        "speech": _seconds(speech, lambda row: row.get("sentiment_score", 0.5) + (0.5 if row.get("keyword_hit", False) else 0.0)),
        "current_fused": _seconds(fused, lambda row: row["fused_score"]),
    }
    length = max(max(values, default=0) for values in sources.values()) + 1
    return {name: np.array([values.get(second, 0.0) for second in range(length)])
            for name, values in sources.items()}


def metrics(scores, labels):
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "average_precision": None, "mean_positive_percentile": None}
    order = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
    percentiles = (order + 1) / len(scores)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "mean_positive_percentile": float(percentiles[labels == 1].mean()),
    }


def top_windows(scores, labels, match_id):
    """Top ten independently scored centered six-second windows for review."""
    candidates = []
    for center in range(len(scores)):
        start, end = max(0, center - 3), min(len(scores), center + 3)
        candidates.append((float(scores[start:end].mean()), start, end))
    return [
        {"match": match_id, "start": start, "end": end, "score": score,
         "overlaps_human_interval": bool(labels[start:end].any())}
        for score, start, end in sorted(candidates, reverse=True)[:10]
    ]


def main():
    output = {"run": {"date": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                      "git_commit": git_commit()},
              "binning": "one-second bins; missing channel values are zero",
              "label_sets": {"human_verified": {}, "pseudo_label": {}}}
    human = load_ground_truth("results/human_labels_cricket_match_1.json")
    for match_id, (directory, pseudo_path) in MATCHES.items():
        scores = load_second_scores(directory)
        pseudo = interval_labels(load_ground_truth(pseudo_path), len(next(iter(scores.values()))))
        output["label_sets"]["pseudo_label"][match_id] = {
            "interval_count": len(load_ground_truth(pseudo_path)),
            "metrics": {name: metrics(values, pseudo) for name, values in scores.items()},
        }
        if match_id == "cricket_match_1":
            labels = interval_labels(human, len(next(iter(scores.values()))))
            output["label_sets"]["human_verified"][match_id] = {
                "description": "human-verified, 20 intervals, single labeler",
                "metrics": {name: metrics(values, labels) for name, values in scores.items()},
                "top_ten_six_second_windows": {name: top_windows(values, labels, match_id)
                                                for name, values in scores.items()},
            }
        del scores
        gc.collect()
    Path("results").mkdir(exist_ok=True)
    Path("results/channel_signal.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = ["# Channel signal diagnostic", "", f"Run date: {output['run']['date']}",
             f"Git commit: `{output['run']['git_commit']}`", "",
             "Scores use one-second bins. AUC/AP assess ranking; mean positive percentile is the average percentile of labeled seconds.", ""]
    for label_set, matches in output["label_sets"].items():
        lines += [f"## {label_set.replace('_', ' ')}", "", "| Match | Channel | ROC-AUC | Average precision | Mean positive percentile |", "|---|---|---:|---:|---:|"]
        for match_id, data in matches.items():
            for channel, value in data["metrics"].items():
                lines.append(f"| {match_id} | {channel} | {value['roc_auc']:.4f} | {value['average_precision']:.4f} | {value['mean_positive_percentile']:.4f} |")
        lines.append("")
    lines += ["## Top ten six-second windows: cricket_match_1 human labels", "", "| Channel | Start | End | Score | Overlaps human interval |", "|---|---:|---:|---:|---|"]
    for channel, windows in output["label_sets"]["human_verified"]["cricket_match_1"]["top_ten_six_second_windows"].items():
        for item in windows:
            lines.append(f"| {channel} | {item['start']} | {item['end']} | {item['score']:.6f} | {item['overlaps_human_interval']} |")
    Path("results/channel_signal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved diagnostics JSON: results/channel_signal.json")
    print("Saved diagnostics Markdown: results/channel_signal.md")


if __name__ == "__main__":
    main()
