import librosa
import numpy as np
import json
import os
import subprocess
import tempfile
import argparse

def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from the video using FFmpeg.
    Downsamples to 22050 Hz and mono to speed up librosa processing.
    """
    command = [
        "ffmpeg", "-i", video_path, 
        "-q:a", "0", "-map", "a", 
        "-ar", "22050", "-ac", "1", 
        "-y", audio_path
    ]
    # Run silently
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_audio(video_path, output_json):
    print(f"Extracting audio from {video_path}...")
    temp_audio = tempfile.mktemp(suffix=".wav")
    extract_audio_from_video(video_path, temp_audio)

    print("Loading audio into librosa...")
    # sr=22050 matches our ffmpeg output
    y, sr = librosa.load(temp_audio, sr=22050)
    
    # Define frame and hop length for feature extraction
    # 512 hop length at 22050 Hz gives ~43 frames per second
    hop_length = 512
    frame_length = 2048

    print("Computing features: STE, Spectral Centroid, ZCR...")
    
    # 1. Energy (Librosa uses RMS; STE is proportional to RMS squared)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # 2. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    # 3. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Map each frame index to an exact timestamp in seconds
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    print("Aggregating into 1-second bins...")
    duration_sec = int(np.ceil(librosa.get_duration(y=y, sr=sr)))
    
    results = []
    
    for t in range(duration_sec):
        # Create a boolean mask for frames that fall into the [t, t+1) second window
        mask = (times >= t) & (times < t + 1)
        
        if not np.any(mask):
            continue
        
        # Average the features for this 1-second block
        # Square the RMS to get Short-Time Energy (STE)
        ste_val = float(np.mean(rms[mask]**2)) 
        centroid_val = float(np.mean(centroid[mask]))
        zcr_val = float(np.mean(zcr[mask]))
        
        # Rounding to 4 decimals to keep the JSON file lightweight
        results.append({
            "t": float(t),
            "STE": round(ste_val, 4),
            "spectral_centroid": round(centroid_val, 4),
            "ZCR": round(zcr_val, 4)
        })

    print(f"Saving output to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Cleanup temporary wav file
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    
    print("Done! Audio Channel processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio features for cricket highlights.")
    parser.add_argument("video_path", type=str, help="Path to the input cricket video file.")
    args = parser.parse_args()
    
    # Automatically name the output JSON
    output_filename = "audio_channel_output.json"
    output_filepath = os.path.join(os.path.dirname(args.video_path) or ".", output_filename)
    
    process_audio(args.video_path, output_filepath)