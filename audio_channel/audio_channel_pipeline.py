import argparse
import json
import os
import subprocess
import tempfile
import librosa
import numpy as np


def extract_audio_from_video(video_path, audio_path):
    """Extracts audio from video using FFmpeg, downsampling to 22050 Hz Mono."""
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-q:a",
        "0",
        "-map",
        "a",
        "-ar",
        "22050",
        "-ac",
        "1",
        "-y",
        audio_path,
    ]
    subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def process_audio(video_path, output_json):
    print(f"Extracting audio from {video_path}...")
    temp_audio = tempfile.mktemp(suffix=".wav")
    try:
        extract_audio_from_video(video_path, temp_audio)

        print("Loading audio into librosa...")
        y, sr = librosa.load(temp_audio, sr=22050)

        hop_length = 512
        frame_length = 2048

        print("Computing audio features: STE, Spectral Centroid, ZCR...")
        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]

        times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )
        duration_sec = int(np.ceil(librosa.get_duration(y=y, sr=sr)))

        results = []
        for t in range(duration_sec):
            mask = (times >= t) & (times < t + 1)
            if not np.any(mask):
                continue

            ste_val = float(np.mean(rms[mask] ** 2))
            centroid_val = float(np.mean(centroid[mask]))
            zcr_val = float(np.mean(zcr[mask]))

            results.append({
                "t": float(t),
                "STE": round(ste_val, 4),
                "spectral_centroid": round(centroid_val, 4),
                "ZCR": round(zcr_val, 4),
            })

        print(f"Saving audio features to {output_json}...")
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    print("Audio Channel processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process audio features for cricket highlights."
    )
    parser.add_argument(
        "video_path", type=str, help="Path to input cricket video."
    )
    args = parser.parse_args()

    output_filename = "audio_channel_output.json"
    output_filepath = os.path.join(
        os.path.dirname(args.video_path) or ".", output_filename
    )
    process_audio(args.video_path, output_filepath)