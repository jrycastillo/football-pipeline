#!/usr/bin/env python3
import sys
import types

# Rule constraint: Stub cv2 before importing project modules to support local machines
sys.modules["cv2"] = types.ModuleType("cv2")

import unittest
from collections import defaultdict
from tools.eval_events import (
    match_events,
    parse_type_tolerances,
    resolve_player_info,
    normalize_event_type,
    run_sweep
)

class TestEvalEvents(unittest.TestCase):

    def setUp(self):
        self.default_tolerances = defaultdict(lambda: 3.0)
        self.fps = 25.0
        self.vid_stride = 3

    def test_exact_match(self):
        """
        Verify that a detection at the exact same time as a GT event matches correctly.
        """
        # 10.0 seconds corresponds to frame 83.33 -> frame 83 is close (9.96s)
        # Let's use clean numbers: frame 100 * stride 3 / 25.0 = 12.0 seconds
        gt_events = [{"type": "pass", "time_s": 12.0}]
        det_events = [{"type": "pass", "frame": 100, "confidence": 0.8, "from": 10}]
        
        metrics, unmatched_gt, unmatched_det = match_events(
            gt_events=gt_events,
            det_events=det_events,
            player_stats=None,
            fps=self.fps,
            vid_stride=self.vid_stride,
            tolerances=self.default_tolerances,
            strictness="type-only",
            min_conf=0.5
        )
        
        self.assertEqual(metrics["pass"]["true_positives"], 1)
        self.assertEqual(metrics["pass"]["false_positives"], 0)
        self.assertEqual(metrics["pass"]["false_negatives"], 0)
        self.assertEqual(len(unmatched_gt), 0)
        self.assertEqual(len(unmatched_det), 0)

    def test_tolerance_edge(self):
        """
        Verify that tolerance boundary checks are enforced:
        - Within tolerance is matched.
        - Outside tolerance is rejected.
        """
        # GT is at 10.0s. Tolerance is 3.0s.
        gt_events = [{"type": "pass", "time_s": 10.0}]
        
        # Case 1: Within tolerance (12.4s -> diff 2.4s)
        # 12.4s * 25.0 / 3 = 103.3 -> frame 103 corresponds to 12.36s
        det_within = [{"type": "pass", "frame": 103, "confidence": 0.8}]
        metrics, unmatched_gt, unmatched_det = match_events(
            gt_events=gt_events,
            det_events=det_within,
            player_stats=None,
            fps=self.fps,
            vid_stride=self.vid_stride,
            tolerances=self.default_tolerances,
            strictness="type-only",
            min_conf=0.5
        )
        self.assertEqual(metrics["pass"]["true_positives"], 1)
        self.assertEqual(len(unmatched_gt), 0)

        # Case 2: Outside tolerance (13.2s -> diff 3.2s)
        # 13.2s * 25.0 / 3 = 110 -> frame 110 corresponds to 13.20s
        det_outside = [{"type": "pass", "frame": 110, "confidence": 0.8}]
        metrics, unmatched_gt, unmatched_det = match_events(
            gt_events=gt_events,
            det_events=det_outside,
            player_stats=None,
            fps=self.fps,
            vid_stride=self.vid_stride,
            tolerances=self.default_tolerances,
            strictness="type-only",
            min_conf=0.5
        )
        self.assertEqual(metrics["pass"]["true_positives"], 0)
        self.assertEqual(metrics["pass"]["false_positives"], 1)
        self.assertEqual(metrics["pass"]["false_negatives"], 1)
        self.assertEqual(len(unmatched_gt), 1)
        self.assertEqual(len(unmatched_det), 1)

    def test_double_match_prevention(self):
        """
        Verify that a single detection matches at most one GT event and vice versa.
        """
        # Two GT events at 10.0s and 11.0s. One detection at 10.5s.
        gt_events = [
            {"type": "pass", "time_s": 10.0},
            {"type": "pass", "time_s": 11.0}
        ]
        # frame 88 * 3 / 25 = 10.56s
        det_events = [{"type": "pass", "frame": 88, "confidence": 0.8}]
        
        metrics, unmatched_gt, unmatched_det = match_events(
            gt_events=gt_events,
            det_events=det_events,
            player_stats=None,
            fps=self.fps,
            vid_stride=self.vid_stride,
            tolerances=self.default_tolerances,
            strictness="type-only",
            min_conf=0.5
        )
        # Should match only 1 GT event.
        self.assertEqual(metrics["pass"]["true_positives"], 1)
        self.assertEqual(metrics["pass"]["false_negatives"], 1)
        self.assertEqual(metrics["pass"]["false_positives"], 0)
        self.assertEqual(len(unmatched_gt), 1)
        self.assertEqual(len(unmatched_det), 0)

        # One GT event at 10.0s. Two detections at 10.1s and 10.2s.
        gt_events = [{"type": "pass", "time_s": 10.0}]
        det_events = [
            {"type": "pass", "frame": 84, "confidence": 0.8}, # 10.08s
            {"type": "pass", "frame": 85, "confidence": 0.8}  # 10.20s
        ]
        metrics, unmatched_gt, unmatched_det = match_events(
            gt_events=gt_events,
            det_events=det_events,
            player_stats=None,
            fps=self.fps,
            vid_stride=self.vid_stride,
            tolerances=self.default_tolerances,
            strictness="type-only",
            min_conf=0.5
        )
        # Should match only 1 detection. The other detection is unmatched (FP).
        self.assertEqual(metrics["pass"]["true_positives"], 1)
        self.assertEqual(metrics["pass"]["false_negatives"], 0)
        self.assertEqual(metrics["pass"]["false_positives"], 1)
        self.assertEqual(len(unmatched_gt), 0)
        self.assertEqual(len(unmatched_det), 1)

    def test_strictness_levels(self):
        """
        Verify that different strictness levels filter matches correctly based on player/team.
        """
        player_stats = {
            "10": {"jersey_number": 10, "team": "Red"},
            "4": {"jersey_number": 4, "team": "White"}
        }

        # GT: Red Team, Player 10
        gt_events = [{"type": "pass", "time_s": 10.0, "team": "Red", "player": 10}]
        
        # Det: White Team, Player 4 (matches in type and time but not player/team)
        det_events = [{"type": "pass", "frame": 83, "confidence": 0.8, "from": 4}] # 9.96s

        # 1. Type-only should match successfully
        metrics, unmatched_gt, _ = match_events(
            gt_events=gt_events, det_events=det_events, player_stats=player_stats,
            fps=self.fps, vid_stride=self.vid_stride, tolerances=self.default_tolerances,
            strictness="type-only", min_conf=0.5
        )
        self.assertEqual(metrics["pass"]["true_positives"], 1)
        self.assertEqual(len(unmatched_gt), 0)

        # 2. Type+team should fail (Red vs White)
        metrics, unmatched_gt, _ = match_events(
            gt_events=gt_events, det_events=det_events, player_stats=player_stats,
            fps=self.fps, vid_stride=self.vid_stride, tolerances=self.default_tolerances,
            strictness="type+team", min_conf=0.5
        )
        self.assertEqual(metrics["pass"]["true_positives"], 0)
        self.assertEqual(len(unmatched_gt), 1)

        # 3. Type+player should fail (Player 10 vs 4)
        metrics, unmatched_gt, _ = match_events(
            gt_events=gt_events, det_events=det_events, player_stats=player_stats,
            fps=self.fps, vid_stride=self.vid_stride, tolerances=self.default_tolerances,
            strictness="type+player", min_conf=0.5
        )
        self.assertEqual(metrics["pass"]["true_positives"], 0)
        self.assertEqual(len(unmatched_gt), 1)

        # 4. Correct team and player should match successfully in type+player
        det_correct = [{"type": "pass", "frame": 83, "confidence": 0.8, "from": 10}]
        metrics, unmatched_gt, _ = match_events(
            gt_events=gt_events, det_events=det_correct, player_stats=player_stats,
            fps=self.fps, vid_stride=self.vid_stride, tolerances=self.default_tolerances,
            strictness="type+player", min_conf=0.5
        )
        self.assertEqual(metrics["pass"]["true_positives"], 1)
        self.assertEqual(len(unmatched_gt), 0)

    def test_confidence_sweep(self):
        """
        Verify that confidence sweep correctly computes metrics at multiple thresholds.
        """
        gt_events = [{"type": "pass", "time_s": 10.0}]
        det_events = [
            {"type": "pass", "frame": 83, "confidence": 0.35}, # match if threshold <= 0.35
            {"type": "pass", "frame": 84, "confidence": 0.65}  # match if threshold <= 0.65
        ]

        sweep_results = run_sweep(
            gt_events=gt_events,
            det_events=det_events,
            player_stats=None,
            fps=self.fps,
            vid_stride=self.vid_stride,
            tolerances=self.default_tolerances,
            strictness="type-only"
        )

        # Check threshold results
        # Threshold 0.3: both detections kept. 1 matched (TP=1), 1 unmatched (FP=1).
        res_03 = next(r for r in sweep_results if abs(r["threshold"] - 0.3) < 0.01)
        self.assertEqual(res_03["metrics"]["overall"]["true_positives"], 1)
        self.assertEqual(res_03["metrics"]["overall"]["false_positives"], 1)

        # Threshold 0.6: only 0.65 detection kept. 1 matched (TP=1), 0 unmatched (FP=0).
        res_06 = next(r for r in sweep_results if abs(r["threshold"] - 0.6) < 0.01)
        self.assertEqual(res_06["metrics"]["overall"]["true_positives"], 1)
        self.assertEqual(res_06["metrics"]["overall"]["false_positives"], 0)

        # Threshold 0.9: no detections kept. 0 matched (TP=0, FP=0, FN=1).
        res_09 = next(r for r in sweep_results if abs(r["threshold"] - 0.9) < 0.01)
        self.assertEqual(res_09["metrics"]["overall"]["true_positives"], 0)
        self.assertEqual(res_09["metrics"]["overall"]["false_positives"], 0)
        self.assertEqual(res_09["metrics"]["overall"]["false_negatives"], 1)

if __name__ == "__main__":
    unittest.main()
