"""Save manual highlight intervals entered by a human labeler.

This tool never suggests, derives, or fills in highlight intervals. Watch the
match in your own video player, then type each start/end time when prompted.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


MATCH_IDS = ("cricket_match_1", "cricket_match_2", "football_match_1")


def parse_interval(text):
    """Parse one manually entered `start,end` interval in seconds."""
    try:
        start_text, end_text = text.split(",", 1)
        start, end = float(start_text.strip()), float(end_text.strip())
    except ValueError as error:
        raise ValueError("Use two numbers separated by a comma, e.g. 123.5,129.5.") from error
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Times must be finite seconds with 0 <= start < end.")
    return [start, end]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, choices=MATCH_IDS)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing label file only after entering manual intervals.",
    )
    args = parser.parse_args(argv)

    intervals = []
    print("Watch the match yourself before and while entering labels.")
    print("Enter one highlight as start,end seconds (for example 123.5,129.5).")
    print("Press Enter on a blank line when finished. Nothing is auto-filled.")
    while True:
        raw = input(f"Interval {len(intervals) + 1}: ").strip()
        if not raw:
            break
        try:
            interval = parse_interval(raw)
        except ValueError as error:
            print(f"Invalid interval: {error}")
            continue
        intervals.append(interval)
        print(f"Recorded [{interval[0]}, {interval[1]}].")

    if not intervals:
        print("No intervals entered; no label file was written.")
        return 1
    output_path = Path(args.output_dir) / f"human_labels_{args.match}.json"
    if output_path.exists() and not args.replace:
        print(f"Refusing to overwrite existing labels: {output_path}")
        print("After manually entering labels, use --replace to replace this file.")
        return 1
    payload = {
        "match_id": args.match,
        "labeled_by": "human",
        "labeler_count": 1,
        "created_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "highlights": intervals,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(intervals)} manual intervals to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
