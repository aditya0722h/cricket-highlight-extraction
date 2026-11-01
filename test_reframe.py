"""Focused tests for content-aware portrait crop planning."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from visual_channel.reframe import plan_crop_x, plan_reframe


class CropPlanningTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
        self.video_path = handle.name
        handle.close()

    def tearDown(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)

    def write_video(self, positions, fps=12, frame_size=(640, 360)):
        width, height = frame_size
        writer = cv2.VideoWriter(
            self.video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            frame_size,
        )
        self.assertTrue(writer.isOpened())
        for x_position in positions:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            if x_position is not None:
                cv2.rectangle(frame, (x_position, 150), (x_position + 35, 185), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

    def test_moving_rectangle_returns_bounded_even_positions(self):
        positions = list(range(70, 430, 20))
        self.write_video(positions)

        crop_x = plan_crop_x(self.video_path)

        self.assertEqual(len(crop_x), len(positions))
        self.assertTrue(all(value % 2 == 0 for value in crop_x))
        self.assertTrue(all(0 <= value <= 640 - 202 for value in crop_x))
        self.assertLess(min(crop_x[:len(crop_x) // 2]), crop_x[0])
        self.assertGreater(crop_x[-1], min(crop_x[:len(crop_x) // 2]))

    def test_pan_is_clamped_per_frame(self):
        self.write_video([60 + index * 25 for index in range(20)])

        crop_x = plan_crop_x(self.video_path)

        self.assertLessEqual(max(abs(b - a) for a, b in zip(crop_x, crop_x[1:])), 10)

    def test_static_frames_hold_crop_and_return_a_plan(self):
        self.write_video([None] * 8)

        plan = plan_reframe(self.video_path)

        self.assertEqual(len(set(plan.crop_x)), 1)
        self.assertEqual(plan.crop_width, 202)
        self.assertEqual(plan.layouts, ["smart_crop"] * 8)

    def test_odd_frame_width_keeps_even_crop_origin_in_bounds(self):
        self.write_video([None] * 4, frame_size=(641, 361))

        plan = plan_reframe(self.video_path)

        self.assertTrue(all(value % 2 == 0 for value in plan.crop_x))
        self.assertTrue(all(value + plan.crop_width <= plan.frame_width for value in plan.crop_x))


if __name__ == "__main__":
    unittest.main()