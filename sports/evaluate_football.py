"""Evaluate football event predictions against timestamped ground truth."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sports.config import detect_events, load_sport_config


def load_events(path):
    with open(path, "r", encoding="utf-8") as event_file:
        payload = json.load(event_file)
    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("Ground truth must be a list or an object with an 'events' list")
    normalized = []
    for event in events:
        if not isinstance(event, dict) or "event_type" not in event:
            raise ValueError("Each event must contain an event_type")
        if "t" in event:
            start = end = float(event["t"])
        elif "start" in event and "end" in event:
            start, end = float(event["start"]), float(event["end"])
            if end < start:
                raise ValueError("Ground-truth event end must be >= start")
        else:
            raise ValueError("Each event must have t or both start and end")
        normalized.append({
            "event_type": str(event["event_type"]),
            "start": start,
            "end": end,
        })
    return normalized


def evaluate(predictions, ground_truth, tolerance):
    unmatched = set(range(len(ground_truth)))
    matches = 0
    for prediction in predictions:
        timestamp = float(prediction["t"])
        event_type = prediction["event_type"]
        candidates = [
            index for index in unmatched
            if ground_truth[index]["event_type"] == event_type
            and ground_truth[index]["start"] - tolerance
            <= timestamp
            <= ground_truth[index]["end"] + tolerance
        ]
        if candidates:
            nearest = min(
                candidates,
                key=lambda index: min(
                    abs(timestamp - ground_truth[index]["start"]),
                    abs(timestamp - ground_truth[index]["end"]),
                    0.0 if ground_truth[index]["start"] <= timestamp <= ground_truth[index]["end"] else float("inf"),
                ),
            )
            unmatched.remove(nearest)
            matches += 1

    precision = matches / len(predictions) if predictions else 0.0
    recall = matches / len(ground_truth) if ground_truth else 0.0
    return precision, recall, matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Per-second feature JSON list")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth JSON file")
    parser.add_argument(
        "--tolerance", type=float, default=10.0,
        help="Seconds of tolerance around ground-truth timestamps (default: 10)",
    )
    arguments = parser.parse_args()
    if arguments.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    with open(arguments.features, "r", encoding="utf-8") as feature_file:
        records = json.load(feature_file)
    if not isinstance(records, list):
        raise ValueError("Feature file must contain a list of per-second records")

    config = load_sport_config("football")
    predictions = detect_events(records, config)
    ground_truth = load_events(arguments.ground_truth)
    precision, recall, matches = evaluate(
        predictions, ground_truth, arguments.tolerance
    )
    print(f"Football events predicted: {len(predictions)}")
    print(f"Ground-truth events: {len(ground_truth)}")
    print(f"Matched events: {matches}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")


if __name__ == "__main__":
    main()
