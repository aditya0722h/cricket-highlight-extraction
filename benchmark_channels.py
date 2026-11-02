"""Run existing channel pipelines with per-channel wall-time/RSS monitoring."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import psutil


def video_duration_seconds(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Could not determine video duration: {video_path}")
        return frame_count / fps
    finally:
        capture.release()


def run_worker(channel, video_path, output_path, model_size):
    if channel == "visual":
        from visual_channel.visual_channel_pipeline import run_visual_channel

        run_visual_channel(video_path, output_file=output_path)
    elif channel == "audio":
        from audio_channel.audio_channel_pipeline import process_audio

        process_audio(video_path, output_path)
    elif channel == "speech":
        from speech_channel.speech_channel_pipeline import process_speech

        process_speech(video_path, output_path, model_size=model_size)
    else:
        raise ValueError(f"Unsupported channel: {channel}")


def process_tree_rss(process):
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    total = 0
    for item in processes:
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def run_measured_channel(channel, video_path, output_path, model_size):
    started = time.perf_counter()
    peak_rss = 0
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        channel,
        video_path,
        output_path,
        model_size,
    ]

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
        child = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process = psutil.Process(child.pid)
        last_report = started
        while child.poll() is None:
            peak_rss = max(peak_rss, process_tree_rss(process))
            now = time.perf_counter()
            if now - last_report >= 60:
                print(
                    f"{channel}: elapsed {now - started:.0f}s; "
                    f"peak process-tree RSS {peak_rss / (1024**2):.1f} MiB",
                    flush=True,
                )
                last_report = now
            time.sleep(0.25)
        peak_rss = max(peak_rss, process_tree_rss(process))
        elapsed = time.perf_counter() - started
        log_file.seek(0)
        output = log_file.read()

    if child.returncode != 0:
        print(output, file=sys.stderr)
        raise RuntimeError(
            f"{channel} channel failed with exit code {child.returncode}."
        )

    print(output, end="", flush=True)
    return {
        "status": "completed",
        "wall_seconds": round(elapsed, 3),
        "peak_process_tree_rss_mb": round(peak_rss / (1024**2), 2),
        "output_file": output_path,
    }


def run_benchmark(args):
    video_path = os.path.abspath(args.video)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    duration = video_duration_seconds(video_path)
    results_path = os.path.join(output_dir, "channel_benchmark.json")
    if os.path.isfile(results_path):
        with open(results_path, "r", encoding="utf-8") as file:
            report = json.load(file)
    else:
        report = {
            "video": video_path,
            "duration_seconds": round(duration, 3),
            "model_size": args.model_size,
            "channels": {},
        }

    for channel in args.channels:
        output_path = os.path.join(output_dir, f"{channel}_output.json")
        if (
            not args.force
            and os.path.isfile(output_path)
            and os.path.getsize(output_path) > 0
        ):
            try:
                with open(output_path, "r", encoding="utf-8") as file:
                    existing = json.load(file)
                if isinstance(existing, list) and existing:
                    report["channels"].setdefault(
                        channel,
                        {
                            "status": "resumed_existing_output",
                            "wall_seconds": None,
                            "peak_process_tree_rss_mb": None,
                            "output_file": output_path,
                        },
                    )
                    print(
                        f"{channel}: reusing existing output {output_path}",
                        flush=True,
                    )
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        print(f"Running {channel} channel on {video_path}", flush=True)
        if args.force and os.path.exists(output_path):
            os.remove(output_path)
        report["channels"][channel] = run_measured_channel(
            channel, video_path, output_path, args.model_size
        )
        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

    report["video_minutes"] = round(duration / 60.0, 4)
    for channel, metrics in report["channels"].items():
        wall_seconds = metrics.get("wall_seconds")
        metrics["seconds_per_video_minute"] = (
            round(wall_seconds / (duration / 60.0), 3)
            if wall_seconds is not None and duration > 0
            else None
        )
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Saved channel measurements: {results_path}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Local input video.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON outputs.")
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=("visual", "audio", "speech"),
        default=("visual", "audio", "speech"),
    )
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--force", action="store_true", help="Recompute existing outputs.")
    return parser.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        if len(argv) != 5:
            raise ValueError("Invalid internal worker arguments.")
        _, channel, video_path, output_path, model_size = argv
        run_worker(channel, video_path, output_path, model_size)
        return 0

    args = parse_args(argv)
    try:
        run_benchmark(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Channel benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
