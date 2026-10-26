# Sport profiles

The `cricket.json` and `football.json` files use the same schema. Event-specific
settings include audio-energy and motion thresholds, commentary keywords, and
overlay captions. Profiles also define fusion weights, clip padding, smoothing,
and event-window settings. `sports/config.py` validates profiles and exposes
keyword and multimodal event detection helpers.

Cricket values preserve the previous pipeline defaults. Football thresholds,
weights, and timing values are initial guesses, not trained values. In the
absence of a football keyword, simultaneous audio-energy and motion evidence
is provisionally labeled as a near-miss; tune this with labeled footage.

Football ground-truth JSON accepts either an event list or an object with an
`events` list. Events use seconds and one of the configured event types:

```json
{
  "events": [
    { "event_type": "goal", "t": 42.0 },
    { "event_type": "near_miss", "start": 61.0, "end": 64.0 },
    { "event_type": "card", "t": 78.0 }
  ]
}
```

Evaluate feature records produced by the football pipeline against your
annotated ground-truth file. The default matching tolerance is 10 seconds:

```powershell
..\.venv\Scripts\python.exe sports\evaluate_football.py --features output_highlights\job_1\fusion_output.json --ground-truth C:\data\football_ground_truth.json
```

Use `--tolerance 5` to tighten event matching. The script prints event counts,
matched events, precision, and recall.
