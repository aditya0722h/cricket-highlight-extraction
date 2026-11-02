import argparse
import json
import os
import subprocess
import tempfile
import librosa
import numpy as np
import soundfile as sf


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
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
    )


def process_audio(video_path, output_json, chunk_seconds=60):
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than zero.")

    print(f"Extracting audio from {video_path}...")
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio = temp_file.name
    temp_file.close()
    try:
        extract_audio_from_video(video_path, temp_audio)
        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
            raise RuntimeError(f"FFmpeg did not create a usable audio file for {video_path}")

        hop_length = 512
        frame_length = 2048
        sums = counts = centroid_sums = zcr_sums = None

        print(
            "Computing audio features in "
            f"{chunk_seconds}-second chunks: STE, Spectral Centroid, ZCR..."
        )
        with sf.SoundFile(temp_audio) as audio_file:
            sr = audio_file.samplerate
            total_samples = len(audio_file)
            duration_sec = int(np.ceil(total_samples / sr))
            samples_per_chunk = max(
                hop_length,
                (chunk_seconds * sr // hop_length) * hop_length,
            )
            sums = np.zeros(duration_sec, dtype=np.float64)
            centroid_sums = np.zeros(duration_sec, dtype=np.float64)
            zcr_sums = np.zeros(duration_sec, dtype=np.float64)
            counts = np.zeros(duration_sec, dtype=np.int64)
            context_samples = frame_length // 2

            for core_start in range(0, total_samples, samples_per_chunk):
                core_end = min(total_samples, core_start + samples_per_chunk)
                read_start = max(0, core_start - context_samples)
                read_end = min(total_samples, core_end + context_samples)
                audio_file.seek(read_start)
                y = audio_file.read(
                    read_end - read_start, dtype="float32", always_2d=False
                )

                rms = librosa.feature.rms(
                    y=y, frame_length=frame_length, hop_length=hop_length
                )[0]
                centroid = librosa.feature.spectral_centroid(
                    y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
                )[0]
                zcr = librosa.feature.zero_crossing_rate(
                    y=y, frame_length=frame_length, hop_length=hop_length
                )[0]

                global_frame_indices = (
                    np.arange(len(rms)) + read_start // hop_length
                )
                times = librosa.frames_to_time(
                    global_frame_indices, sr=sr, hop_length=hop_length
                )
                core_mask = (times >= core_start / sr) & (times < core_end / sr)
                frame_seconds = np.floor(times[core_mask]).astype(np.int64)
                valid_seconds = (frame_seconds >= 0) & (
                    frame_seconds < duration_sec
                )
                frame_seconds = frame_seconds[valid_seconds]
                selected_rms = rms[core_mask][valid_seconds]
                selected_centroid = centroid[core_mask][valid_seconds]
                selected_zcr = zcr[core_mask][valid_seconds]

                counts += np.bincount(
                    frame_seconds, minlength=duration_sec
                )[:duration_sec]
                sums += np.bincount(
                    frame_seconds,
                    weights=selected_rms**2,
                    minlength=duration_sec,
                )[:duration_sec]
                centroid_sums += np.bincount(
                    frame_seconds,
                    weights=selected_centroid,
                    minlength=duration_sec,
                )[:duration_sec]
                zcr_sums += np.bincount(
                    frame_seconds,
                    weights=selected_zcr,
                    minlength=duration_sec,
                )[:duration_sec]

                print(
                    f"Processed audio samples through "
                    f"{core_end / sr:.0f}/{duration_sec} seconds."
                )

        results = []
        for t in range(duration_sec):
            if counts[t] == 0:
                continue
            results.append({
                "t": float(t),
                "STE": round(float(sums[t] / counts[t]), 4),
                "spectral_centroid": round(
                    float(centroid_sums[t] / counts[t]), 4
                ),
                "ZCR": round(float(zcr_sums[t] / counts[t]), 4),
            })

        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
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

    # UPDATED: Matches fusion pipeline expectation
    output_filename = "audio_output.json"
    output_filepath = os.path.join(
        os.path.dirname(args.video_path) or ".", output_filename
    )
    process_audio(args.video_path, output_filepath)
