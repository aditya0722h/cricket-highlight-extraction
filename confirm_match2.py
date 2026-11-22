"""Untouched-match confirmation using the fixed peak_fusion defaults.

The preregistered yes/no question uses strict F1 at k=10 only: audio beats
random when its F1 is greater than the mean of 100 fixed-seed random baselines.
All requested k values are reported; none is selected after the run.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from diagnostics.channel_signal import interval_labels, load_second_scores, metrics
from eval_peak import as_pairs, baselines, loose_metrics
from evaluate import calculate_metrics, load_ground_truth
from fusion_channel.peak_fusion import (
    DEFAULT_ROLLING_SEC, DEFAULT_SMOOTH_SEC, MIN_PEAK_SEPARATION_SEC,
    peak_windows, prepare_scores,
)

MATCH_ID = "cricket_match_2"
DIRECTORY = "cricket_full_match_2"
LABEL_FILE = "results/human_labels_cricket_match_2.json"
K_VALUES = (5, 10, 20, 40)
PRIMARY_K = 10


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def build_peak_data(directory= DIRECTORY):
    """Read existing channel JSONs and apply the unchanged peak defaults once."""
    raw = load_second_scores(directory)
    duration = float(len(next(iter(raw.values()))) - 1)
    prepared = {name: prepare_scores(raw[name]) for name in ("visual", "audio", "speech")}
    scores = dict(prepared)
    scores["equal_fusion"] = sum(prepared.values()) / 3.0
    ranked = {name: peak_windows(values, duration_sec=duration)[0] for name, values in scores.items()}
    return raw, ranked, duration


def evaluate(truth, ranked, duration):
    rows = []
    for k in K_VALUES:
        for variant, all_peaks in ranked.items():
            selected = all_peaks[:k]
            windows = as_pairs(selected)
            lengths = [end - start for start, end in windows]
            rows.append({
                "variant": variant,
                "k": k,
                "windows": selected,
                "strict": calculate_metrics(windows, truth, 0.5),
                "looser_metric": loose_metrics(windows, truth),
                "baselines": baselines(duration, lengths, truth),
            })
    return rows


def number(value):
    return "n/a" if value is None else f"{value:.4f}"


def write_markdown(report):
    lines = [
        "# Second-match confirmation",
        "",
        f"Run date: {report['run']['date']}",
        f"Git commit: `{report['run']['git_commit']}`",
        "",
        "This is the untouched cricket_match_2, human-verified, single-labeler set. "
        "Peak defaults were fixed before this run: 120 s rolling z-score, 5 s smoothing, "
        "30 s separation, and 8--30 s variable windows.",
        "",
        "## Channel scores against human labels",
        "",
        "| Channel | ROC-AUC | Average precision |",
        "|---|---:|---:|",
    ]
    for channel, result in report["channel_metrics"].items():
        lines.append(f"| {channel} | {number(result['roc_auc'])} | {number(result['average_precision'])} |")
    lines += [
        "",
        "## Peak-selection results",
        "",
        "Strict uses IoU >= 0.5 maximum one-to-one matching. The looser metric allows a "
        "window to overlap a label expanded by +/-10 seconds, also one-to-one.",
        "",
        "| Variant | k | Strict P | Strict R | Strict F1 | Looser P | Looser R | Looser F1 | Random strict F1 | Even strict F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        random_f1 = row["baselines"]["random"]["strict"]["f1"]
        even_f1 = row["baselines"]["evenly_spaced"]["strict"]["f1"]
        lines.append(
            "| {variant} | {k} | {sp} | {sr} | {sf} | {lp} | {lr} | {lf} | {rm} +/- {rs} | {ef} |".format(
                variant=row["variant"], k=row["k"],
                sp=number(row["strict"]["precision"]), sr=number(row["strict"]["recall"]), sf=number(row["strict"]["f1"]),
                lp=number(row["looser_metric"]["precision"]), lr=number(row["looser_metric"]["recall"]), lf=number(row["looser_metric"]["f1"]),
                rm=number(random_f1["mean"]), rs=number(random_f1["std"]), ef=number(even_f1),
            )
        )
    answer = report["preregistered_audio_question"]
    lines += ["", "## Preregistered audio question", "", answer["plain_language"], ""]
    Path("results/confirm_match2.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    truth = load_ground_truth(LABEL_FILE)
    raw, ranked, duration = build_peak_data()
    labels = interval_labels(truth, len(next(iter(raw.values()))))
    channel_metrics = {name: metrics(raw[name], labels) for name in ("visual", "audio", "speech")}
    rows = evaluate(truth, ranked, duration)
    primary = next(row for row in rows if row["variant"] == "audio" and row["k"] == PRIMARY_K)
    random_mean = primary["baselines"]["random"]["strict"]["f1"]["mean"]
    audio_beats_random = primary["strict"]["f1"] > random_mean
    report = {
        "run": {"date": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(), "git_commit": git_commit()},
        "match_id": MATCH_ID, "label_file": LABEL_FILE, "label_interval_count": len(truth),
        "label_description": "human-verified, single labeler; labels created before model output",
        "defaults": {"rolling_zscore_sec": DEFAULT_ROLLING_SEC, "smoothing_sec": DEFAULT_SMOOTH_SEC,
                     "minimum_peak_separation_sec": MIN_PEAK_SEPARATION_SEC, "k_values": list(K_VALUES)},
        "channel_metrics": channel_metrics, "rows": rows,
        "preregistered_audio_question": {
            "primary_metric": "strict F1 at k=10 versus the 100-seed random strict-F1 mean",
            "answer": "yes" if audio_beats_random else "no",
            "plain_language": (
                f"Answer: {'yes' if audio_beats_random else 'no'}. At the preregistered k=10, audio strict F1 was "
                f"{primary['strict']['f1']:.4f}; the random mean was {random_mean:.4f}."
            ),
        },
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/confirm_match2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    invalid = [window for row in rows for window in row["windows"]
               if window["start"] < 0 or window["end"] > duration or window["end"] <= window["start"]]
    print("Saved confirmation JSON: results/confirm_match2.json")
    print("Saved confirmation Markdown: results/confirm_match2.md")
    print(f"human_verified {MATCH_ID}: {len(truth)} intervals")
    print(f"invalid_or_beyond_duration_windows: {len(invalid)}")
    print(report["preregistered_audio_question"]["plain_language"])


if __name__ == "__main__":
    main()
