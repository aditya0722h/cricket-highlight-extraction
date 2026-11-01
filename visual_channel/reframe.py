"""Content-aware portrait reframing for sport highlight clips.

Planning uses low-resolution optical flow and can be saved as JSON independently
of rendering. Rendering uses OpenCV for frames and FFmpeg to restore source audio.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class ReframePlan:
    """Per-frame crop positions and per-frame shot layout decisions."""

    fps: float
    frame_width: int
    frame_height: int
    crop_width: int
    crop_x: list[int]
    layouts: list[str]
    cuts: list[int]


def _even_crop_width(height: int) -> int:
    return max(2, min(height, int(height * 9 / 16) // 2 * 2))


def _histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])


def _analyze_video(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    analysis_width: int = 320,
    alpha: float = 0.12,
    max_pan_fraction: float = 0.015,
    cut_threshold: float = 0.5,
    motion_threshold: float = 0.15,
    scatter_threshold: float = 0.18,
) -> ReframePlan:
    if start_time < 0 or (end_time is not None and end_time <= start_time):
        raise ValueError("The requested video interval must have positive duration")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise ValueError("Video has invalid frame dimensions")

    start_frame = max(0, int(start_time * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    stop_frame = None if end_time is None else max(start_frame + 1, int(end_time * fps))
    analysis_height = max(1, int(round(frame_height * analysis_width / frame_width)))
    crop_width = _even_crop_width(frame_height)
    crop_max = max(0, (frame_width - crop_width) // 2 * 2)
    previous_gray = None
    previous_hist = None
    previous_x = crop_max / 2
    crop_positions: list[int] = []
    motion_maps: list[np.ndarray] = []
    cuts: list[int] = [0]

    try:
        while stop_frame is None or start_frame + len(crop_positions) < stop_frame:
            ok, frame = capture.read()
            if not ok:
                break
            small = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            histogram = _histogram(small)
            cut = (
                previous_hist is not None
                and cv2.compareHist(previous_hist, histogram, cv2.HISTCMP_BHATTACHARYYA) > cut_threshold
            )
            if cut:
                cuts.append(len(crop_positions))

            motion_map = np.zeros((analysis_height, analysis_width), dtype=np.float32)
            target_x = crop_max / 2 if cut else previous_x
            if previous_gray is not None and not cut:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                residual = flow - np.median(flow.reshape(-1, 2), axis=0)
                magnitude = np.linalg.norm(residual, axis=2)
                motion_cutoff = max(float(np.percentile(magnitude, 90)), 0.5)
                active = magnitude >= motion_cutoff
                selected = np.where(active, magnitude, 0.0)
                mass = float(selected.sum())
                motion_map = selected
                if mass > 1e-4 and float(np.max(magnitude)) > 0.5:
                    center_small = float((selected * np.arange(analysis_width)[None, :]).sum() / mass)
                    target_x = center_small * frame_width / analysis_width - crop_width / 2
                    target_x = float(np.clip(target_x, 0, crop_max))

            if cut:
                smoothed_x = target_x
            elif len(crop_positions) == 0:
                smoothed_x = target_x
            elif float(motion_map.max(initial=0.0)) <= 0.5:
                smoothed_x = previous_x
            else:
                smoothed_x = previous_x + alpha * (target_x - previous_x)
                max_pan = frame_width * max_pan_fraction
                smoothed_x = float(np.clip(smoothed_x, previous_x - max_pan, previous_x + max_pan))

            previous_x = float(np.clip(smoothed_x, 0, crop_max))
            crop_positions.append(int(round(previous_x / 2) * 2))
            motion_maps.append(motion_map)
            previous_gray = gray
            previous_hist = histogram
    finally:
        capture.release()

    if not crop_positions:
        raise ValueError("Video contains no readable frames in the requested interval")

    boundaries = cuts + [len(crop_positions)]
    layouts = ["smart_crop"] * len(crop_positions)
    for shot_start, shot_end in zip(boundaries, boundaries[1:]):
        shot_maps = motion_maps[shot_start:shot_end]
        weights = np.sum([motion_map.sum(axis=0) for motion_map in shot_maps], axis=0)
        total = float(weights.sum())
        if total > 1e-4:
            x_positions = np.arange(len(weights))
            center = float((weights * x_positions).sum() / total)
            variance = float((weights * (x_positions - center) ** 2).sum() / total)
            spread = np.sqrt(variance) / max(1, analysis_width)
        else:
            spread = 0.0
        if spread > scatter_threshold:
            layouts[shot_start:shot_end] = ["blurred_background"] * (shot_end - shot_start)

    return ReframePlan(
        fps=float(fps), frame_width=frame_width, frame_height=frame_height,
        crop_width=crop_width, crop_x=crop_positions, layouts=layouts, cuts=cuts,
    )


def plan_crop_x(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    analysis_width: int = 320,
) -> list[int]:
    """Return a smoothed, bounded crop-left position for every analyzed frame."""
    return _analyze_video(video_path, start_time, end_time, analysis_width).crop_x


def plan_reframe(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    analysis_width: int = 320,
) -> ReframePlan:
    """Analyze a clip and return crop coordinates plus its shot layouts."""
    return _analyze_video(video_path, start_time, end_time, analysis_width)


def save_plan(plan: ReframePlan, path: str) -> None:
    """Save a plan in a JSON format accepted by :func:`load_plan`."""
    with open(path, "w", encoding="utf-8") as plan_file:
        json.dump(asdict(plan), plan_file, indent=2)


def load_plan(path: str) -> ReframePlan:
    """Load a previously cached reframe plan."""
    with open(path, "r", encoding="utf-8") as plan_file:
        return ReframePlan(**json.load(plan_file))


def _draw_caption(frame: np.ndarray, event_label: Optional[str], score: Optional[str]) -> None:
    caption = " | ".join(text for text in (event_label, score) if text)
    if not caption:
        return
    height, width = frame.shape[:2]
    font_scale = max(0.6, width / 1200)
    thickness = max(1, int(width / 600))
    baseline_y = int(height * 0.1)
    cv2.putText(frame, caption, (int(width * 0.05), baseline_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, caption, (int(width * 0.05), baseline_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_portrait(
    video_path: str,
    output_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    resolution: str = "1080x1920",
    event_label: Optional[str] = None,
    score: Optional[str] = None,
    plan: Optional[ReframePlan] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> ReframePlan:
    """Render a portrait clip and mux its original audio when present."""
    if start_time < 0 or (end_time is not None and end_time <= start_time):
        raise ValueError("The requested video interval must have positive duration")
    try:
        output_width, output_height = (int(value) for value in resolution.lower().split("x", 1))
    except (ValueError, AttributeError) as error:
        raise ValueError("resolution must be formatted as WIDTHxHEIGHT") from error
    if output_width <= 0 or output_height <= 0:
        raise ValueError("resolution dimensions must be positive")

    plan = plan or plan_reframe(video_path, start_time, end_time)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or plan.fps or 30.0
    start_frame = max(0, int(start_time * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = len(plan.crop_x)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(suffix=".mp4", dir=output_dir, delete=False)
    temp_video = temp_handle.name
    temp_handle.close()
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (output_width, output_height))
    if not writer.isOpened():
        capture.release()
        os.unlink(temp_video)
        raise RuntimeError("OpenCV could not initialize the portrait video encoder")

    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            frame_height, frame_width = frame.shape[:2]
            crop_width = min(plan.crop_width, frame_width)
            crop_x = int(np.clip(plan.crop_x[frame_index], 0, max(0, frame_width - crop_width)))
            if frame_width != plan.frame_width or frame_height != plan.frame_height:
                crop_width = _even_crop_width(frame_height)
                crop_x = min(frame_width - crop_width, int(crop_x * frame_width / plan.frame_width))

            if plan.layouts[frame_index] == "blurred_background":
                scale = max(output_width / frame_width, output_height / frame_height)
                background = cv2.resize(
                    frame,
                    (int(frame_width * scale), int(frame_height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
                bg_y = (background.shape[0] - output_height) // 2
                bg_x = (background.shape[1] - output_width) // 2
                background = background[bg_y:bg_y + output_height, bg_x:bg_x + output_width]
                background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(12, output_width / 35))
                foreground_height = max(1, int(frame_height * output_width / frame_width))
                foreground = cv2.resize(frame, (output_width, foreground_height), interpolation=cv2.INTER_AREA)
                top = (output_height - foreground_height) // 2
                if top < 0:
                    foreground = cv2.resize(foreground, (output_width, output_height))
                    background = foreground.copy()
                else:
                    background[top:top + foreground_height] = foreground
                canvas = background
            else:
                crop = frame[:, crop_x:crop_x + crop_width]
                canvas = cv2.resize(crop, (output_width, output_height), interpolation=cv2.INTER_AREA)
            _draw_caption(canvas, event_label, score)
            writer.write(canvas)
            if progress_callback and (frame_index % max(1, frame_count // 100) == 0):
                progress_callback((frame_index + 1) / frame_count)
    finally:
        capture.release()
        writer.release()

    rendered_duration = min(frame_count, int((end_time - start_time) * fps)) / fps if end_time else frame_count / fps
    command = [
        "ffmpeg", "-y", "-i", temp_video,
        "-ss", str(start_time), "-i", video_path,
        "-map", "0:v:0", "-map", "1:a:0?", "-t", str(rendered_duration),
        "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", output_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio mux failed: {result.stderr[-1200:]}")
    finally:
        if os.path.exists(temp_video):
            os.unlink(temp_video)
    LOGGER.info("Portrait render complete: %s", output_path)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source video path")
    parser.add_argument("--output", required=True, help="Portrait output path")
    parser.add_argument("--aspect", choices=("landscape", "portrait"), default="portrait")
    parser.add_argument("--resolution", choices=("1080x1920", "720x1280"), default="1080x1920")
    parser.add_argument("--event-label")
    parser.add_argument("--score")
    parser.add_argument("--plan-json", help="Load or save a cached plan")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if arguments.aspect == "landscape":
        raise SystemExit("Landscape rendering is handled by the existing export pipeline.")
    cached_plan = load_plan(arguments.plan_json) if arguments.plan_json and os.path.exists(arguments.plan_json) else None
    plan = render_portrait(
        arguments.input, arguments.output, arguments.start, arguments.end,
        arguments.resolution, arguments.event_label, arguments.score, cached_plan,
        progress_callback=lambda fraction: LOGGER.info("Render %.0f%%", fraction * 100),
    )
    if arguments.plan_json and cached_plan is None:
        save_plan(plan, arguments.plan_json)


if __name__ == "__main__":
    main()