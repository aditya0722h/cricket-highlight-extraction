import sys
import os
import json
import time
import argparse
import re
import subprocess
import tempfile
from urllib.parse import urlsplit
import cv2
from visual_channel.reframe import render_portrait
from sports.config import detect_events, load_sport_config
try:
    from moviepy import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy.editor import VideoFileClip, concatenate_videoclips


MAX_STREAM_READ_FAILURES = 10


def load_fusion_records(fusion_path):
    with open(fusion_path, "r", encoding="utf-8") as fusion_file:
        records = json.load(fusion_file)
    if not isinstance(records, list):
        raise ValueError("fusion_output.json must contain a list of records")
    return records


def find_peak_window(records, duration, config):
    scored_records = []
    for record in records:
        try:
            timestamp = float(record["t"])
            score_value = record.get("S_total_smoothed")
            if score_value is None:
                score_value = record["S_total"]
            score = float(score_value)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= timestamp <= duration:
            scored_records.append((score, timestamp))
    if not scored_records:
        raise ValueError("Fusion output has no valid scores for this video")
    peak_time = max(scored_records)[1]
    return (
        max(0.0, peak_time - config["slowmo_window_before_seconds"]),
        min(duration, peak_time + config["slowmo_window_after_seconds"]),
    )


def caption_for_window(records, start_time, end_time, config):
    labels = []
    for record in records:
        try:
            timestamp = float(record.get("t", -1))
        except (TypeError, ValueError):
            continue
        if not start_time <= timestamp <= end_time:
            continue
        configured_events = record.get("events", [])
        labels.extend(
            configured_events if isinstance(configured_events, list) else []
        )
        text = str(record.get("transcript_segment", "")).lower()
        for event, keywords in config["overlay_keywords"].items():
            if any(
                re.search(rf"\b{re.escape(keyword.lower())}\b", text)
                for keyword in keywords
            ) and event not in labels:
                labels.append(event)
    for event in config["event_types"]:
        if event in labels:
            return config["overlay_captions"][event]
    return None


def add_landscape_caption(frame, label):
    if not label:
        return frame
    frame = frame.copy()
    height, width = frame.shape[:2]
    font_scale = max(0.7, width / 1200)
    thickness = max(2, int(width / 600))
    position = (int(width * 0.05), int(height * 0.12))
    cv2.putText(
        frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (0, 0, 0), thickness + 4, cv2.LINE_AA,
    )
    cv2.putText(
        frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return frame


def render_slow_motion(video_path, output_path, start_time, end_time, has_audio):
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError("Peak moment interval contains no video frames")
    command = [
        "ffmpeg", "-y", "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}", "-i", video_path,
        "-vf", "setpts=2.0*PTS",
        "-map", "0:v:0",
    ]
    if has_audio:
        command.extend(["-map", "0:a:0", "-af", "atempo=0.5", "-c:a", "aac"])
    else:
        command.append("-an")
    command.extend([
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", output_path,
    ])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Slow-motion render failed: {result.stderr[-1200:]}")


def is_network_source(source):
    parsed = urlsplit(source)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def open_video_capture(source, is_stream):
    backend = cv2.CAP_FFMPEG if is_stream else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if is_stream and capture.isOpened():
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def read_next_frame(capture, source, is_stream, consecutive_failures):
    ret, frame = capture.read()
    if ret:
        return capture, frame, 0, True
    if not is_stream:
        return capture, None, consecutive_failures, False

    consecutive_failures += 1
    if consecutive_failures >= MAX_STREAM_READ_FAILURES:
        print(
            "[WARNING] Live stream stopped producing frames after "
            f"{consecutive_failures} attempts; finalizing captured highlights."
        )
        return capture, None, consecutive_failures, False

    print(
        "[WARNING] Live stream frame read failed "
        f"({consecutive_failures}/{MAX_STREAM_READ_FAILURES}); reconnecting.",
        flush=True,
    )
    capture.release()
    time.sleep(min(0.5 * consecutive_failures, 2.0))
    capture = open_video_capture(source, is_stream)
    if not capture.isOpened():
        print("[WARNING] Could not reconnect to live stream; will retry.", flush=True)
    return capture, None, consecutive_failures, True


def main(
    video_path=None, output_dir="output_highlights", aspect="landscape",
    resolution="1080x1920", edit_style="plain", fusion_path="fusion_output.json",
    sport="cricket",
):
    if video_path is None:
        parser = argparse.ArgumentParser(description="Extract cricket highlights from video")
        parser.add_argument("video_path", help="Video path or stream URL")
        parser.add_argument("--output_dir", default=output_dir)
        parser.add_argument("--aspect", choices=("landscape", "portrait", "both"), default=aspect)
        parser.add_argument("--resolution", choices=("1080x1920", "720x1280"), default=resolution)
        parser.add_argument("--edit_style", choices=("plain", "slowmo", "overlays"), default=edit_style)
        parser.add_argument("--fusion_path", default=fusion_path)
        parser.add_argument("--sport", choices=("cricket", "football"), default=sport)
        arguments = parser.parse_args()
        video_path, output_dir = arguments.video_path, arguments.output_dir
        aspect, resolution = arguments.aspect, arguments.resolution
        edit_style, fusion_path = arguments.edit_style, arguments.fusion_path
        sport = arguments.sport

    config = load_sport_config(sport)

    video_path = video_path.strip()
    is_stream = is_network_source(video_path)

    if not is_stream:
        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            print(f"[ERROR] Video file does not exist: {video_path}")
            sys.exit(1)

    print(f"[INFO] Starting Pipeline for video: {video_path}")

    motion_threshold = config["legacy_motion_threshold"]
    if sport == "cricket" and os.path.exists("model_config.json"):
        with open("model_config.json", "r", encoding="utf-8") as model_config_file:
            model_config = json.load(model_config_file)
        motion_threshold = model_config.get("motion_threshold", motion_threshold)

    cap = open_video_capture(video_path, is_stream)
    if is_stream and not cap.isOpened():
        for attempt in range(1, MAX_STREAM_READ_FAILURES + 1):
            print(
                f"[WARNING] Could not open live stream "
                f"({attempt}/{MAX_STREAM_READ_FAILURES}); retrying.",
                flush=True,
            )
            time.sleep(min(0.5 * attempt, 2.0))
            cap.release()
            cap = open_video_capture(video_path, is_stream)
            if cap.isOpened():
                break

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaled_threshold = motion_threshold * width * height / (640 * 480)
    motion_threshold = max(1, int(min(scaled_threshold, width * height * 0.01)))
    print(f"[INFO] Motion threshold: {motion_threshold} changed pixels")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Video loaded: {width}x{height} @ {fps}fps")

    frame_count = 0
    is_recording = False
    highlight_start = 0.0
    cooldown = 0
    highlight_windows = []
    motion_by_second = {}
    last_progress = -1
    channel_records = []

    if sport == "football":
        from audio_channel.audio_channel_pipeline import process_audio
        from speech_channel.speech_channel_pipeline import process_speech

        audio_path = os.path.join(output_dir, "audio_features.json")
        speech_path = os.path.join(output_dir, "speech_features.json")
        process_audio(video_path, audio_path, sport=sport)
        process_speech(
            video_path, speech_path,
            model_size=config["speech_model_size"], sport=sport,
        )
        with open(audio_path, "r", encoding="utf-8") as audio_file:
            audio_records = json.load(audio_file)
        with open(speech_path, "r", encoding="utf-8") as speech_file:
            speech_records = json.load(speech_file)
        channel_records = [
            {"t": record["t"], **record}
            for record in audio_records + speech_records
        ]

    stream_failures = 0
    while True:
        cap, prev_frame, stream_failures, keep_reading = read_next_frame(
            cap, video_path, is_stream, stream_failures
        )
        if prev_frame is not None:
            break
        if not keep_reading:
            cap.release()
            print("[ERROR] Failed to read the first frame.")
            sys.exit(1)

    if prev_frame is None:
        cap.release()
        print("[ERROR] Failed to read the first frame.")
        sys.exit(1)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    while True:
        cap, frame, stream_failures, keep_reading = read_next_frame(
            cap, video_path, is_stream, stream_failures
        )
        if frame is None:
            if keep_reading:
                continue
            break
        stream_failures = 0

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_score = cv2.countNonZero(thresh)
        second = int(frame_count / fps)
        motion_by_second[second] = max(
            motion_by_second.get(second, 0.0),
            motion_score / max(1, width * height),
        )

        if motion_score > motion_threshold:
            if not is_recording:
                print(f"[EVENT] Highlight triggered at frame {frame_count}!")
                is_recording = True
                highlight_start = max(
                    0.0,
                    (frame_count - int(fps * config["motion_event_lead_seconds"])) / fps,
                )
            cooldown = int(fps * config["motion_cooldown_seconds"])

        if is_recording:
            if cooldown > 0:
                cooldown -= 1
            else:
                end_time = frame_count / fps
                if end_time > highlight_start:
                    highlight_windows.append((highlight_start, end_time))
                    print(
                        f"[INFO] Highlight segment detected: "
                        f"{highlight_start:.2f}s - {end_time:.2f}s"
                    )
                is_recording = False

        if total_frames > 0:
            progress = min(90, int(frame_count * 90 / total_frames))
            if progress != last_progress:
                print(f"[PROGRESS] {progress}", flush=True)
                last_progress = progress
        elif frame_count % max(1, int(fps * 5)) == 0:
            print(f"[FRAME] {frame_count}", flush=True)

        prev_gray = gray

    cap.release()
    if is_recording:
        end_time = frame_count / fps
        if end_time > highlight_start:
            highlight_windows.append((highlight_start, end_time))

    if sport == "football":
        channel_records.extend(
            {"t": second, "V_motion": motion_score}
            for second, motion_score in motion_by_second.items()
        )
        events = detect_events(channel_records, config)
        event_windows = []
        for event in events:
            event_start = event.get("start", event["t"])
            event_end = event.get("end", event["t"])
            start = max(0.0, event_start - config["clip_padding"]["before"])
            end = min(
                (total_frames / fps) if total_frames > 0 else frame_count / fps,
                event_end + config["clip_padding"]["after"],
            )
            if end > start:
                event_windows.append((start, end))

        merged_windows = []
        for start, end in sorted(event_windows):
            if merged_windows and start <= merged_windows[-1][1]:
                merged_windows[-1] = (
                    merged_windows[-1][0], max(merged_windows[-1][1], end)
                )
            else:
                merged_windows.append((start, end))
        highlight_windows = merged_windows

        by_time = {}
        for record in channel_records:
            try:
                timestamp = round(float(record["t"]), 2)
            except (KeyError, TypeError, ValueError):
                continue
            by_time.setdefault(timestamp, {"t": timestamp}).update(record)
        events_by_time = {}
        for event in events:
            events_by_time.setdefault(round(event["t"]), []).append(event["event_type"])
        score_weights = config["fusion_weights"]["pipeline"]
        max_motion = max(motion_by_second.values(), default=0.0) or 1.0
        max_audio = max(
            (float(record.get("STE", 0.0) or 0.0) for record in audio_records),
            default=0.0,
        ) or 1.0
        for timestamp, record in by_time.items():
            normalized_motion = float(record.get("V_motion", 0.0) or 0.0) / max_motion
            normalized_audio = float(record.get("STE", 0.0) or 0.0) / max_audio
            speech_score = (
                float(record.get("sentiment_score", config["neutral_sentiment"]) or config["neutral_sentiment"])
                + (
                    config["keyword_hit_bonus"]
                    if record.get("keyword_hit")
                    else 0.0
                )
            )
            record["S_total_smoothed"] = (
                score_weights["wv"] * normalized_motion
                + score_weights["wa"] * normalized_audio
                + score_weights["wt"] * speech_score
            )
            record["events"] = events_by_time.get(round(timestamp), [])
        channel_records = list(by_time.values())
        fusion_path = os.path.join(output_dir, "fusion_output.json")
        with open(fusion_path, "w", encoding="utf-8") as fusion_file:
            json.dump(channel_records, fusion_file, indent=2)

    if highlight_windows or edit_style == "slowmo":
        print("[INFO] Compiling master highlight reel...")
        source_clip = None
        clips = []
        portrait_clips = []
        final_reel = None
        portrait_reel = None
        slowmo_source = None
        try:
            source_clip = VideoFileClip(video_path)
            has_audio = source_clip.audio is not None
            subclip = getattr(source_clip, "subclipped", None)
            if subclip is None:
                subclip = source_clip.subclip
            fusion_records = []
            if edit_style in {"slowmo", "overlays"}:
                if not os.path.isfile(fusion_path):
                    raise FileNotFoundError(f"Fusion output file not found: {fusion_path}")
                fusion_records = load_fusion_records(fusion_path)

            if edit_style == "slowmo":
                peak_start, peak_end = find_peak_window(
                    fusion_records,
                    float(source_clip.duration or frame_count / fps),
                    config,
                )
                slowmo_handle = tempfile.NamedTemporaryFile(
                    suffix=".mp4", dir=output_dir, delete=False
                )
                slowmo_source = slowmo_handle.name
                slowmo_handle.close()
                render_slow_motion(
                    video_path, slowmo_source, peak_start, peak_end, has_audio
                )
                source_clip.close()
                source_clip = VideoFileClip(slowmo_source)
                subclip = getattr(source_clip, "subclipped", None)
                if subclip is None:
                    subclip = source_clip.subclip
                highlight_windows = [(0.0, float(source_clip.duration))]

            for idx, (start_time, end_time) in enumerate(highlight_windows, 1):
                if source_clip.duration is not None:
                    end_time = min(end_time, source_clip.duration)
                if end_time <= start_time:
                    continue

                event_label = (
                    caption_for_window(fusion_records, start_time, end_time, config)
                    if edit_style == "overlays"
                    else None
                )
                landscape_clip = subclip(start_time, end_time)
                if event_label:
                    image_transform = getattr(landscape_clip, "image_transform", None)
                    if image_transform is not None:
                        landscape_clip = image_transform(
                            lambda frame, label=event_label: add_landscape_caption(frame, label)
                        )
                    else:
                        landscape_clip = landscape_clip.fl_image(
                            lambda frame, label=event_label: add_landscape_caption(frame, label)
                        )

                if aspect in {"landscape", "both"}:
                    output_filename = os.path.join(output_dir, f"highlight_{idx}.mp4")
                    landscape_clip.write_videofile(
                        output_filename, codec="libx264", audio=has_audio,
                        audio_codec="aac" if has_audio else None, logger=None,
                    )
                    clips.append(landscape_clip)
                    print(f"[INFO] Highlight segment saved: {output_filename}")
                else:
                    landscape_clip.close()

                if aspect in {"portrait", "both"}:
                    portrait_filename = os.path.join(
                        output_dir,
                        f"highlight_{idx}_portrait.mp4" if aspect == "both"
                        else f"highlight_{idx}.mp4",
                    )
                    render_portrait(
                        slowmo_source or video_path, portrait_filename,
                        start_time, end_time, resolution=resolution,
                        event_label=event_label,
                        progress_callback=lambda fraction, segment=idx: print(
                            f"[PROGRESS] {90 + int((((segment - 1) + fraction) / len(highlight_windows)) * 8)}",
                            flush=True,
                        ),
                    )
                    portrait_clips.append(VideoFileClip(portrait_filename))

            if clips:
                final_reel = concatenate_videoclips(clips)
                final_reel.write_videofile(
                    os.path.join(output_dir, "reel_landscape.mp4"),
                    codec="libx264", audio=has_audio,
                    audio_codec="aac" if has_audio else None, logger=None,
                )
                final_reel.write_videofile(
                    os.path.join(output_dir, "master_highlight_reel.mp4"),
                    codec="libx264", audio=has_audio,
                    audio_codec="aac" if has_audio else None, logger=None,
                )
            if portrait_clips:
                portrait_reel = concatenate_videoclips(portrait_clips)
                portrait_reel.write_videofile(
                    os.path.join(output_dir, "reel_portrait.mp4"),
                    codec="libx264", audio=portrait_reel.audio is not None,
                    audio_codec="aac" if portrait_reel.audio is not None else None,
                    logger=None,
                )
            if not clips and not portrait_clips:
                print("[WARNING] No valid highlight clips found to compile.")
        except Exception as e:
            print(f"[ERROR] Failed to compile master reel: {e}")
            sys.exit(1)
        finally:
            if final_reel is not None:
                final_reel.close()
            if portrait_reel is not None:
                portrait_reel.close()
            for clip in clips:
                clip.close()
            for clip in portrait_clips:
                clip.close()
            if source_clip is not None:
                source_clip.close()
            if slowmo_source and os.path.exists(slowmo_source):
                os.remove(slowmo_source)
    else:
        print("[INFO] No highlight events detected to compile.")

    print("[PROGRESS] 100", flush=True)
    print("[INFO] Pipeline processing completed successfully.")

if __name__ == "__main__":
    main()