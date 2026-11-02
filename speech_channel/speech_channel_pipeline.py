import argparse
import gc
import json
import os
import re
import tempfile

import av
import numpy as np
import nltk
from faster_whisper import WhisperModel
from faster_whisper.audio import (
    _group_frames,
    _ignore_invalid_frames,
    _resample_frames,
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


CRICKET_KEYWORDS = {
    "six", "four", "out", "wicket", "catch", "bowled", "huge",
    "shot", "boundary", "review", "appeal", "gone", "taken",
    "maximum", "stumped", "lbw", "dropped", "unbelievable", "wided",
}


def analyze_sentiment(sia, text):
    """Calculates compound sentiment score normalized from [-1, 1] to [0, 1]."""
    if not text.strip():
        return 0.5
    scores = sia.polarity_scores(text)
    return round((scores["compound"] + 1.0) / 2.0, 4)


def check_keyword_hit(text):
    words = re.findall(r"\b\w+\b", text.lower())
    return any(word in CRICKET_KEYWORDS for word in words)


def extract_speech_audio(video_path, audio_path, sampling_rate=16000):
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sampling_rate,
    )
    sample_count = 0
    with open(audio_path, "wb") as output_file:
        with av.open(video_path, mode="r", metadata_errors="ignore") as container:
            frames = container.decode(audio=0)
            frames = _ignore_invalid_frames(frames)
            frames = _group_frames(frames, 500000)
            frames = _resample_frames(frames, resampler)

            for frame in frames:
                samples = frame.to_ndarray()
                if samples.dtype != np.int16:
                    raise RuntimeError(
                        f"Expected 16-bit speech samples, got {samples.dtype}."
                    )
                waveform = samples.reshape(-1).astype(np.float32) / 32768.0
                waveform.tofile(output_file)
                sample_count += len(waveform)

    del resampler
    gc.collect()
    if sample_count == 0:
        raise RuntimeError(f"No speech audio was found in {video_path}.")
    return sample_count


class BoundedFeatureExtractor:
    def __init__(self, feature_extractor, batch_frames=8192):
        if batch_frames < 1:
            raise ValueError("Feature batch size must be positive.")
        self._feature_extractor = feature_extractor
        self._batch_frames = batch_frames

    def __getattr__(self, name):
        return getattr(self._feature_extractor, name)

    def __call__(self, waveform, padding=160, chunk_length=None):
        extractor = self._feature_extractor
        if chunk_length is not None:
            extractor.n_samples = chunk_length * extractor.sampling_rate
            extractor.nb_max_frames = extractor.n_samples // extractor.hop_length

        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        if waveform.ndim != 1:
            raise ValueError("Speech waveform must be a one-dimensional array.")

        padded_length = len(waveform) + padding
        center_padding = extractor.n_fft // 2
        centered_length = padded_length + 2 * center_padding
        stft_frame_count = (
            1 + (centered_length - extractor.n_fft) // extractor.hop_length
        )
        frame_count = stft_frame_count - 1
        log_mel = np.empty(
            (extractor.mel_filters.shape[0], frame_count),
            dtype=np.float32,
        )
        window = np.hanning(extractor.n_fft + 1)[:-1].astype(np.float32)
        sample_offsets = np.arange(extractor.n_fft, dtype=np.int64)
        waveform_length = len(waveform)

        for frame_start in range(0, frame_count, self._batch_frames):
            frame_end = min(frame_start + self._batch_frames, frame_count)
            frame_indices = np.arange(frame_start, frame_end, dtype=np.int64)
            sample_indices = (
                frame_indices[:, None] * extractor.hop_length
                + sample_offsets[None, :]
                - center_padding
            )
            sample_indices[sample_indices < 0] *= -1
            right_padding = sample_indices >= padded_length
            sample_indices[right_padding] = (
                2 * padded_length - 2 - sample_indices[right_padding]
            )

            frames = np.zeros(sample_indices.shape, dtype=np.float32)
            valid_samples = sample_indices < waveform_length
            frames[valid_samples] = waveform[sample_indices[valid_samples]]
            frames *= window

            spectrum = np.fft.rfft(
                frames,
                n=extractor.n_fft,
                axis=-1,
            ).astype(np.complex64)
            magnitudes = np.abs(spectrum) ** 2
            log_mel[:, frame_start:frame_end] = (
                extractor.mel_filters @ magnitudes.T
            )

        np.maximum(log_mel, 1e-10, out=log_mel)
        np.log10(log_mel, out=log_mel)
        maximum = float(log_mel.max())
        np.maximum(log_mel, maximum - 8.0, out=log_mel)
        log_mel += 4.0
        log_mel /= 4.0
        return log_mel


def process_speech(video_path, output_json, model_size="small", feature_batch_frames=8192):
    import torch

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    audio_file = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
    audio_path = audio_file.name
    audio_file.close()
    audio = None

    try:
        print(f"Extracting speech audio from {video_path}...")
        sample_count = extract_speech_audio(video_path, audio_path)
        audio = np.memmap(
            audio_path,
            dtype=np.float32,
            mode="r",
            shape=(sample_count,),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Loading faster-whisper ({model_size}) on {device}...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        model.feature_extractor = BoundedFeatureExtractor(
            model.feature_extractor,
            batch_frames=feature_batch_frames,
        )

        print(f"Transcribing commentary from {video_path}...")
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            language="en",
        )
        segment_list = list(segments)
        duration_sec = int(info.duration)
        del model
    finally:
        if audio is not None:
            audio._mmap.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)

    sia = SentimentIntensityAnalyzer()
    processed_segments = []
    for index, segment in enumerate(segment_list):
        text = segment.text.strip()
        silence_after = (
            index == len(segment_list) - 1
            or segment_list[index + 1].start - segment.end > 1.5
        )
        processed_segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": text,
            "sentiment": float(analyze_sentiment(sia, text)),
            "keyword_hit": bool(check_keyword_hit(text)),
            "silence_after": bool(silence_after),
        })

    print("Mapping speech features to discrete 1-second timestamps...")
    results = []
    for timestamp in range(duration_sec):
        active_segment = next(
            (
                segment
                for segment in processed_segments
                if segment["start"] <= timestamp < segment["end"]
            ),
            None,
        )
        if active_segment:
            results.append({
                "t": float(timestamp),
                "transcript_segment": active_segment["text"],
                "sentiment_score": active_segment["sentiment"],
                "keyword_hit": active_segment["keyword_hit"],
                "silence_after": active_segment["silence_after"],
            })
        else:
            results.append({
                "t": float(timestamp),
                "transcript_segment": "",
                "sentiment_score": 0.5,
                "keyword_hit": False,
                "silence_after": True,
            })

    print(f"Saving output to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print("Speech/NLP Channel processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract speech/NLP features for cricket highlights."
    )
    parser.add_argument(
        "video_path", type=str, help="Path to input cricket video file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Whisper model size (default: small).",
    )
    parser.add_argument(
        "--feature-batch-frames",
        type=int,
        default=8192,
        help="Number of spectrogram frames processed at a time.",
    )
    args = parser.parse_args()

    output_filepath = os.path.join(
        os.path.dirname(args.video_path) or ".", "speech_output.json"
    )
    process_speech(
        args.video_path,
        output_filepath,
        model_size=args.model,
        feature_batch_frames=args.feature_batch_frames,
    )
