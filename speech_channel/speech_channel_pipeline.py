import argparse
import json
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from faster_whisper import WhisperModel

# Download VADER lexicon silently if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# High-excitement cricket keywords
CRICKET_KEYWORDS = {
    "six", "four", "out", "wicket", "catch", "bowled", "huge", 
    "shot", "boundary", "review", "appeal", "gone", "taken", 
    "maximum", "stumped", "lbw", "dropped", "unbelievable", "wided"
}

def analyze_sentiment(sia, text):
    """Calculates compound sentiment score normalized from [-1, 1] to [0, 1]."""
    if not text.strip():
        return 0.5  # Neutral default for silence
    scores = sia.polarity_scores(text)
    # Convert compound (-1 to 1) range to (0 to 1)
    normalized = (scores['compound'] + 1.0) / 2.0
    return round(normalized, 4)

def check_keyword_hit(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return any(w in CRICKET_KEYWORDS for w in words)

def process_speech(video_path, output_json, model_size="small"):
    print(f"Loading faster-whisper ({model_size}) on GPU...")
    # compute_type='float16' uses ~1.2 GB VRAM on RTX 2050
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

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
        kw_hit = check_keyword_hit(text)
        
        # Flag silence if gap to next segment is > 1.5 seconds
        if i < len(segment_list) - 1:
            gap = segment_list[i + 1].start - seg.end
            silence_after = gap > 1.5
        else:
            silence_after = True

        processed_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": text,
            "sentiment": sent_score,
            "keyword_hit": kw_hit,
            "silence_after": silence_after
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
                "silence_after": active_seg["silence_after"]
            })
        else:
            # Default values during silent intervals
            results.append({
                "t": t_float,
                "transcript_segment": "",
                "sentiment_score": 0.5,
                "keyword_hit": False,
                "silence_after": True
            })

    print(f"Saving output to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Speech/NLP Channel processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract speech/NLP features for cricket highlights.")
    parser.add_argument("video_path", type=str, help="Path to input cricket video file.")
    parser.add_argument("--model", type=str, default="small", help="Whisper model size (default: small).")
    args = parser.parse_args()

    output_filename = "speech_channel_output.json"
    output_filepath = os.path.join(os.path.dirname(args.video_path) or ".", output_filename)

    process_speech(args.video_path, output_filepath, args.model)