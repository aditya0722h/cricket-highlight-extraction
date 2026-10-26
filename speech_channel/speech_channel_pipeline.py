import argparse
import json
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from faster_whisper import WhisperModel
from sports.config import keyword_events, load_sport_config

# Download VADER lexicon silently if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(sia, text):
    """Calculates compound sentiment score normalized from [-1, 1] to [0, 1]."""
    if not text.strip():
        return 0.5  # Neutral default for silence
    scores = sia.polarity_scores(text)
    # Convert compound (-1 to 1) range to (0 to 1)
    normalized = (scores['compound'] + 1.0) / 2.0
    return round(normalized, 4)

def check_keyword_hit(text, config=None):
    config = config or load_sport_config("cricket")
    return bool(keyword_events(text, config) or any(
        word in set(re.findall(r"\b\w+\b", text.lower()))
        for word in config["keyword_lists"].get("general", [])
    ))

def process_speech(video_path, output_json, model_size=None, sport="cricket"):
    config = load_sport_config(sport)
    model_size = model_size or config["speech_model_size"]
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading faster-whisper ({model_size}) on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Transcribing commentary from {video_path}...")
    # faster-whisper handles video inputs directly via FFmpeg
    segments, info = model.transcribe(video_path, beam_size=5, language="en")
    segment_list = list(segments)

    sia = SentimentIntensityAnalyzer()
    duration_sec = int(info.duration)

    # Process transcripts, sentiment, and keyword hits per audio segment
    processed_segments = []
    for i, seg in enumerate(segment_list):
        text = seg.text.strip()
        sent_score = analyze_sentiment(sia, text)
        events = keyword_events(text, config)
        kw_hit = check_keyword_hit(text, config)
        
        # Flag silence if the gap to the next segment exceeds the sport profile.
        if i < len(segment_list) - 1:
            gap = segment_list[i + 1].start - seg.end
            silence_after = gap > config["silence_gap_seconds"]
        else:
            silence_after = True

        processed_segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": text,
            "sentiment": float(sent_score),
            "keyword_hit": bool(kw_hit),
            "events": events,
            "silence_after": bool(silence_after)
        })

    print("Mapping speech features to discrete 1-second timestamps...")
    results = []

    for t in range(duration_sec):
        t_float = float(t)
        # Find active segment covering current second 't'
        active_seg = next((s for s in processed_segments if s["start"] <= t_float < s["end"]), None)

        if active_seg:
            results.append({
                "t": t_float,
                "transcript_segment": active_seg["text"],
                "sentiment_score": active_seg["sentiment"],
                "keyword_hit": active_seg["keyword_hit"],
                "events": active_seg["events"],
                "silence_after": active_seg["silence_after"]
            })
        else:
            # Default values during silent intervals
            results.append({
                "t": t_float,
                "transcript_segment": "",
                "sentiment_score": 0.5,
                "keyword_hit": False,
                "events": [],
                "silence_after": True
            })

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"Saving output to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Speech/NLP Channel processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract speech/NLP features for sport highlights.")
    parser.add_argument("video_path", type=str, help="Path to input video file.")
    parser.add_argument("--model", type=str, help="Whisper model size (defaults to the sport profile).")
    parser.add_argument("--sport", choices=("cricket", "football"), default="cricket")
    args = parser.parse_args()

    output_filename = "speech_output.json"
    output_filepath = os.path.join(os.path.dirname(args.video_path) or ".", output_filename)

    process_speech(args.video_path, output_filepath, args.model, sport=args.sport)
