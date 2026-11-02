import cv2
import numpy as np

from visual_channel.extract_frames import extract_frames


def test_grab_retrieve_keeps_sampled_frames_and_timestamps(tmp_path):
    video_path = str(tmp_path / "sample.mp4")
    output_dir = str(tmp_path / "frames")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 64),
    )
    assert writer.isOpened()
    for frame_index in range(20):
        frame = np.full((64, 64, 3), frame_index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    extracted = extract_frames(video_path, output_dir, target_fps=2)
    capture = cv2.VideoCapture(video_path)
    expected_frames = []
    for frame_index in range(20):
        ok, frame = capture.read()
        assert ok
        if frame_index % 5 == 0:
            expected_frames.append(frame)
    capture.release()

    assert [timestamp for timestamp, _ in extracted] == [0.0, 0.5, 1.0, 1.5]
    for (_, path), expected in zip(extracted, expected_frames):
        actual = cv2.imread(path)
        assert np.mean(np.abs(actual.astype(np.int16) - expected.astype(np.int16))) < 3
