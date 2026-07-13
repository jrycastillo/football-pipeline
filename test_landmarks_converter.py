#!/usr/bin/env python3
"""
Unit tests for the pitch keypoint dataset preparation pipeline.

Tests landmarks-to-YOLO conversion logic (bounding box math, padding, clamping,
normalization, keypoint format) and calibration frame selection logic.
"""

import sys
import types
import unittest
import numpy as np

# Rule: Stub cv2 before importing project modules to support local machines
sys.modules["cv2"] = types.ModuleType("cv2")

# Import the code to test
from tools.landmarks_to_yolo_pose import convert_annotation_to_yolo_line
from tools.extract_calibration_frames import (
    compute_pitch_score,
    compute_mad,
    select_best_frame
)

class TestLandmarksConverter(unittest.TestCase):

    def test_yolo_pose_conversion(self):
        """
        Verify that pixel coordinates convert correctly to normalized YOLO pose line.
        """
        points = {
            "0": [100.0, 200.0],
            "13": [400.0, 300.0],
            "31": [600.0, 400.0]
        }
        w_img = 1000.0
        h_img = 500.0
        
        # Expected Box math:
        # min_x=100, max_x=600 -> width=500, center_x=350
        # min_y=200, max_y=400 -> height=200, center_y=300
        # Padded (x1.05): width=525, height=210
        # Clamped: x1=87.5, x2=612.5 (no clamp) -> center_x=350, width=525
        #          y1=195.0, y2=405.0 (no clamp) -> center_y=300, height=210
        # Normalized box: cx=0.35, cy=0.6, w=0.525, h=0.42
        
        yolo_line = convert_annotation_to_yolo_line(points, w_img, h_img)
        parts = yolo_line.strip().split()
        
        # 1 class + 4 box coords + 32 keypoints * 3 values = 101 parts
        self.assertEqual(len(parts), 101)
        self.assertEqual(parts[0], "0") # class
        
        # Check normalized box coordinates
        self.assertAlmostEqual(float(parts[1]), 0.35, places=4)
        self.assertAlmostEqual(float(parts[2]), 0.60, places=4)
        self.assertAlmostEqual(float(parts[3]), 0.525, places=4)
        self.assertAlmostEqual(float(parts[4]), 0.42, places=4)
        
        # Keypoints:
        # k0: [100.0, 200.0] -> cx=0.1, cy=0.4, v=2
        # k13: [400.0, 300.0] -> cx=0.4, cy=0.6, v=2
        # k31: [600.0, 400.0] -> cx=0.6, cy=0.8, v=2
        # Others: cx=0.0, cy=0.0, v=0
        
        # k0 index start in parts: 5
        self.assertAlmostEqual(float(parts[5]), 0.1, places=4)
        self.assertAlmostEqual(float(parts[6]), 0.4, places=4)
        self.assertEqual(parts[7], "2")
        
        # k13 index start in parts: 5 + 13 * 3 = 44
        self.assertAlmostEqual(float(parts[44]), 0.4, places=4)
        self.assertAlmostEqual(float(parts[45]), 0.6, places=4)
        self.assertEqual(parts[46], "2")
        
        # k31 index start in parts: 5 + 31 * 3 = 98
        self.assertAlmostEqual(float(parts[98]), 0.6, places=4)
        self.assertAlmostEqual(float(parts[99]), 0.8, places=4)
        self.assertEqual(parts[100], "2")
        
        # Check a missing one: e.g. k5 (index 5 + 5 * 3 = 20)
        self.assertAlmostEqual(float(parts[20]), 0.0, places=4)
        self.assertAlmostEqual(float(parts[21]), 0.0, places=4)
        self.assertEqual(parts[22], "0")

    def test_yolo_pose_clamping(self):
        """
        Verify that coordinates going beyond image bounds are clamped.
        """
        points = {
            "0": [0.0, 0.0],
            "1": [10.0, 10.0]
        }
        w_img = 10.0
        h_img = 10.0
        
        # Expected Box math:
        # min=0, max=10 -> width=10, center=5
        # Padded (x1.05): width=10.5 -> x1 = 5 - 5.25 = -0.25 -> clamped to 0
        #                              x2 = 5 + 5.25 = 10.25 -> clamped to 10
        # Final: width=10, center=5
        yolo_line = convert_annotation_to_yolo_line(points, w_img, h_img)
        parts = yolo_line.strip().split()
        
        self.assertAlmostEqual(float(parts[1]), 0.5, places=4)
        self.assertAlmostEqual(float(parts[2]), 0.5, places=4)
        self.assertAlmostEqual(float(parts[3]), 1.0, places=4)
        self.assertAlmostEqual(float(parts[4]), 1.0, places=4)

    def test_degenerate_points(self):
        """
        Verify correct handling of zero points and single points.
        """
        with self.assertRaises(ValueError):
            convert_annotation_to_yolo_line({}, 100, 100)
            
        # Single point: should not fail and should have a non-zero size box
        points = {"0": [50.0, 50.0]}
        yolo_line = convert_annotation_to_yolo_line(points, 100.0, 100.0)
        parts = yolo_line.strip().split()
        self.assertAlmostEqual(float(parts[1]), 0.505, places=4)
        self.assertAlmostEqual(float(parts[2]), 0.505, places=4)
        self.assertTrue(float(parts[3]) > 0.0)
        self.assertTrue(float(parts[4]) > 0.0)


    def test_pitch_scoring_and_selection(self):
        """
        Verify the numpy-based pitch scoring and frame selection logic.
        """
        # Create a synthetic "perfect grass" image (green dominant, some white pixels)
        h, w = 100, 100
        # BGR: Green channel high, Red/Blue low
        grass_frame = np.zeros((h, w, 3), dtype=np.uint8)
        grass_frame[:, :, 1] = 120 # G
        grass_frame[:, :, 2] = 40  # R
        grass_frame[:, :, 0] = 30  # B
        
        # Add a white stripe (pitch line)
        grass_frame[40:45, :, :] = 255
        
        # Create a "non-grass" image (e.g. blue sky or gray crowd)
        crowd_frame = np.zeros((h, w, 3), dtype=np.uint8)
        crowd_frame[:, :, 0] = 200 # B
        crowd_frame[:, :, 1] = 200 # G
        crowd_frame[:, :, 2] = 200 # R
        
        # Verify scores
        grass_score = compute_pitch_score(grass_frame)
        crowd_score = compute_pitch_score(crowd_frame)
        
        self.assertTrue(grass_score > 0.0)
        self.assertEqual(crowd_score, 0.0)
        
        # Verify MAD
        mad_same = compute_mad(grass_frame, grass_frame)
        self.assertEqual(mad_same, 0.0)
        
        mad_diff = compute_mad(grass_frame, crowd_frame)
        self.assertTrue(mad_diff > 50.0)
        
        # Test selection logic
        candidates = [
            (10, crowd_frame),
            (20, grass_frame)
        ]
        
        # No last frame: should pick grass frame since it has the highest score
        idx, frame, score = select_best_frame(candidates, last_selected_frame=None)
        self.assertEqual(idx, 20)
        self.assertEqual(score, grass_score)
        
        # Last frame is grass frame: crowd frame is not a duplicate, but its score is 0.0.
        # So we should still pick the best frame from those that are not duplicates.
        # But grass_frame is a duplicate of last_selected_frame.
        idx, frame, score = select_best_frame(candidates, last_selected_frame=grass_frame, duplicate_threshold=15.0)
        self.assertEqual(idx, 10)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
