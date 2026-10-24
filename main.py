import sys
import os
import json
import time
from urllib.parse import urlsplit
import cv2
try:
    from moviepy import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy.editor import VideoFileClip, concatenate_videoclips


MAX_STREAM_READ_FAILURES = 10


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


def main():
    if len(sys.argv) < 2:
        print("[ERROR] No video path or stream URL provided.")
        sys.exit(1)

    video_path = sys.argv[1].strip()
    is_stream = is_network_source(video_path)

    if not is_stream:
        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            print(f"[ERROR] Video file does not exist: {video_path}")
            sys.exit(1)

    print(f"[INFO] Starting Pipeline for video: {video_path}")

    config_path = "model_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        MOTION_THRESHOLD = config.get("motion_threshold", 5000)
    else:
        MOTION_THRESHOLD = 5000

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
    scaled_threshold = MOTION_THRESHOLD * width * height / (640 * 480)
    motion_threshold = max(1, int(min(scaled_threshold, width * height * 0.01)))
    print(f"[INFO] Motion threshold: {motion_threshold} changed pixels")

    os.makedirs("output_highlights", exist_ok=True)

    print(f"[INFO] Video loaded: {width}x{height} @ {fps}fps")

    frame_count = 0
    is_recording = False
    highlight_start = 0.0
    cooldown = 0
    highlight_windows = []
    last_progress = -1

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

        if motion_score > motion_threshold:
            if not is_recording:
                print(f"[EVENT] Highlight triggered at frame {frame_count}!")
                is_recording = True
                highlight_start = max(0.0, (frame_count - int(fps)) / fps)
            cooldown = int(fps * 2)

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

    if highlight_windows:
        print("[INFO] Compiling master highlight reel...")
        source_clip = None
        clips = []
        final_reel = None
        try:
            source_clip = VideoFileClip(video_path)
            has_audio = source_clip.audio is not None
            subclip = getattr(source_clip, "subclipped", None)
            if subclip is None:
                subclip = source_clip.subclip
            for idx, (start_time, end_time) in enumerate(highlight_windows, 1):
                if source_clip.duration is not None:
                    end_time = min(end_time, source_clip.duration)
                if end_time <= start_time:
                    continue

                clip = subclip(start_time, end_time)
                clips.append(clip)
                output_filename = (
                    f"output_highlights/highlight_{idx}.mp4"
                )
                clip.write_videofile(
                    output_filename,
                    codec="libx264",
                    audio=has_audio,
                    audio_codec="aac" if has_audio else None,
                    logger=None,
                )
                print(f"[INFO] Highlight segment saved: {output_filename}")

            if clips:
                final_reel = concatenate_videoclips(clips)
                master_output = "output_highlights/master_highlight_reel.mp4"
                final_reel.write_videofile(
                    master_output,
                    codec="libx264",
                    audio=has_audio,
                    audio_codec="aac" if has_audio else None,
                    logger=None,
                )
                print(f"[SUCCESS] Master highlight reel created at {master_output}")
            else:
                print("[WARNING] No valid highlight clips found to compile.")
        except Exception as e:
            print(f"[ERROR] Failed to compile master reel: {e}")
            sys.exit(1)
        finally:
            if final_reel is not None:
                final_reel.close()
            for clip in clips:
                clip.close()
            if source_clip is not None:
                source_clip.close()
    else:
        print("[INFO] No highlight events detected to compile.")

    print("[PROGRESS] 100", flush=True)
    print("[INFO] Pipeline processing completed successfully.")

if __name__ == "__main__":
    main()