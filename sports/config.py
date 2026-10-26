"""Load and validate per-sport highlight pipeline configuration."""

import json
import math
import os


SPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_SPORTS = frozenset({"cricket", "football"})
REQUIRED_PIPELINE_WEIGHTS = frozenset({"wv", "ws", "wa", "wt"})
REQUIRED_ENGINE_WEIGHTS = frozenset({"visual", "audio", "speech"})


def load_sport_config(sport="cricket"):
    """Read a named profile and reject missing or inconsistent settings."""
    if not isinstance(sport, str) or sport not in SUPPORTED_SPORTS:
        raise ValueError(
            f"Unsupported sport {sport!r}; choose one of {sorted(SUPPORTED_SPORTS)}"
        )

    config_path = os.path.join(SPORTS_DIR, f"{sport}.json")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    event_types = config.get("event_types")
    if not isinstance(event_types, list) or not event_types:
        raise ValueError(f"{config_path}: event_types must be a non-empty list")
    if any(not isinstance(event, str) or not event for event in event_types):
        raise ValueError(f"{config_path}: event_types must contain non-empty strings")

    for section in (
        "audio_energy_thresholds",
        "motion_thresholds",
        "visual_motion_thresholds",
        "keyword_lists",
        "overlay_keywords",
        "overlay_captions",
    ):
        values = config.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"{config_path}: {section} must be an object")

    for event in event_types:
        energy = config["audio_energy_thresholds"].get(event)
        if energy is not None and (
            not isinstance(energy, (int, float))
            or not math.isfinite(energy)
            or energy < 0
        ):
            raise ValueError(
                f"{config_path}: invalid audio energy threshold for {event}"
            )
        for section in ("motion_thresholds", "visual_motion_thresholds"):
            motion = config[section].get(event)
            if not isinstance(motion, (int, float)) or not math.isfinite(motion) or motion < 0:
                raise ValueError(f"{config_path}: invalid {section} for {event}")
        keywords = config["keyword_lists"].get(event)
        if not isinstance(keywords, list) or any(
            not isinstance(word, str) or not word.strip() for word in keywords
        ):
            raise ValueError(f"{config_path}: invalid keyword list for {event}")
        overlay_keywords = config["overlay_keywords"].get(event)
        if not isinstance(overlay_keywords, list) or any(
            not isinstance(word, str) or not word.strip() for word in overlay_keywords
        ):
            raise ValueError(f"{config_path}: invalid overlay keywords for {event}")
        caption = config["overlay_captions"].get(event)
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"{config_path}: missing overlay caption for {event}")

    fusion_weights = config.get("fusion_weights")
    if not isinstance(fusion_weights, dict):
        raise ValueError(f"{config_path}: fusion_weights must be an object")
    for group, required in (
        ("pipeline", REQUIRED_PIPELINE_WEIGHTS),
        ("channel_engine", REQUIRED_ENGINE_WEIGHTS),
    ):
        weights = fusion_weights.get(group)
        if not isinstance(weights, dict) or set(weights) != required:
            raise ValueError(f"{config_path}: invalid {group} fusion weights")
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in weights.values()
        ) or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"{config_path}: {group} fusion weights must sum to 1")

    padding = config.get("clip_padding")
    if not isinstance(padding, dict) or any(
        not isinstance(padding.get(key), (int, float))
        or not math.isfinite(padding[key])
        or padding[key] < 0
        for key in ("before", "after")
    ):
        raise ValueError(f"{config_path}: invalid clip_padding")

    detection = config.get("fusion_detection")
    if not isinstance(detection, dict):
        raise ValueError(f"{config_path}: fusion_detection must be an object")
    for key in (
        "percentile", "min_duration", "smoothing_window", "top_k",
        "window_sec", "pipeline_window_sec", "min_gap_sec",
        "event_merge_gap_seconds",
    ):
        value = detection.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{config_path}: invalid fusion_detection.{key}")
    if detection["percentile"] > 100:
        raise ValueError(f"{config_path}: percentile must not exceed 100")
    if int(detection["smoothing_window"]) % 2 == 0:
        raise ValueError(f"{config_path}: smoothing_window must be odd")

    if config.get("sport") != sport:
        raise ValueError(f"{config_path}: sport must match filename")
    for key in (
        "silence_gap_seconds",
        "keyword_hit_bonus",
        "neutral_sentiment",
        "motion_event_lead_seconds",
        "motion_cooldown_seconds",
        "slowmo_window_before_seconds",
        "slowmo_window_after_seconds",
    ):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{config_path}: invalid {key}")
    if config.get("speech_model_size") not in {"tiny", "base", "small", "medium", "large"}:
        raise ValueError(f"{config_path}: invalid speech_model_size")
    fallback_events = config.get("audio_motion_fallback_events", [])
    if not isinstance(fallback_events, list) or any(
        event not in event_types for event in fallback_events
    ):
        raise ValueError(f"{config_path}: invalid audio_motion_fallback_events")
    return config


def keyword_events(text, config):
    """Return configured event types whose whole-word keywords occur in text."""
    import re

    words = set(re.findall(r"\b\w+\b", text.lower()))
    normalized_text = text.lower()
    matches = []
    for event in config["event_types"]:
        for keyword in config["keyword_lists"][event]:
            keyword = keyword.lower().strip()
            if " " in keyword:
                expression = r"\b" + r"\s+".join(
                    re.escape(part) for part in keyword.split()
                ) + r"\b"
                found = re.search(expression, normalized_text) is not None
            else:
                found = keyword in words
            if found:
                matches.append(event)
                break
    return matches


def detect_events(records, config):
    """Fuse per-second speech, audio-energy, and motion evidence into events."""
    records_by_time = {}
    for record in records:
        try:
            timestamp = round(float(record["t"]), 2)
        except (KeyError, TypeError, ValueError):
            continue
        records_by_time.setdefault(timestamp, []).append(record)

    events = []
    for timestamp, timestamp_records in sorted(records_by_time.items()):
        combined_text = " ".join(
            str(record.get("transcript_segment", record.get("text", "")))
            for record in timestamp_records
        )
        spoken = set(keyword_events(combined_text, config))
        event_signals = {}
        fallback_candidates = []
        for event in config["event_types"]:
            threshold = config["audio_energy_thresholds"][event]
            audio_energy = max(
                (float(record.get("STE", 0.0) or 0.0) for record in timestamp_records),
                default=0.0,
            )
            motion = max(
                (
                    float(record.get("V_motion", record.get("motion_score", 0.0)) or 0.0)
                    for record in timestamp_records
                ),
                default=0.0,
            )
            audio_hit = threshold is not None and audio_energy >= threshold
            motion_hit = motion >= config["motion_thresholds"][event]
            event_signals[event] = (audio_hit, motion_hit)
            if (
                event in config.get("audio_motion_fallback_events", [])
                and audio_hit
                and motion_hit
            ):
                fallback_candidates.append(event)
        selected = spoken
        if not selected and fallback_candidates:
            selected = {
                max(
                    fallback_candidates,
                    key=lambda event: (
                        config["audio_energy_thresholds"][event]
                        + config["motion_thresholds"][event]
                    ),
                )
            }
        for event in config["event_types"]:
            if event in selected:
                audio_hit, motion_hit = event_signals[event]
                events.append({
                    "t": timestamp,
                    "event_type": event,
                    "overlay_caption": config["overlay_captions"][event],
                    "keyword_hit": event in spoken,
                    "audio_hit": audio_hit,
                    "motion_hit": motion_hit,
                })
    grouped = []
    for event in events:
        prior = next(
            (
                item for item in reversed(grouped)
                if item["event_type"] == event["event_type"]
                and event["t"] - item["end"] <= config["fusion_detection"]["event_merge_gap_seconds"]
            ),
            None,
        )
        if prior is None:
            grouped.append({
                **event,
                "start": event["t"],
                "end": event["t"],
            })
        else:
            prior["end"] = event["t"]
            prior["keyword_hit"] = prior["keyword_hit"] or event["keyword_hit"]
            prior["audio_hit"] = prior["audio_hit"] or event["audio_hit"]
            prior["motion_hit"] = prior["motion_hit"] or event["motion_hit"]
            prior["t"] = (prior["start"] + prior["end"]) / 2
    return grouped
