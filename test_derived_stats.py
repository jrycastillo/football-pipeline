#!/usr/bin/env python3
import sys
import types

# Rule constraint: Stub cv2 before importing project modules to support local machines
sys.modules["cv2"] = types.ModuleType("cv2")

import unittest
from stats.derived import compute_derived_stats

class TestDerivedStats(unittest.TestCase):

    def setUp(self):
        # Base player stats mapping
        self.player_stats = {
            "10": {
                "jersey_number": 10,
                "team": "Red",
                "player_name": "Player 10"
            },
            "4": {
                "jersey_number": 4,
                "team": "White",
                "player_name": "Player 4"
            }
        }

    def test_pass_accuracy_and_crosses(self):
        """
        Verify pass accuracy calculation and verify that crosses are counted as passes.
        """
        events = [
            # Player 10: 2 passes (1 complete, 1 incomplete) + 2 crosses (both complete)
            # Total passes = 4, complete = 3 -> accuracy = 75.0%
            {"type": "pass", "from": 10, "complete": True},
            {"type": "pass", "from": 10, "complete": False},
            {"type": "cross", "from": 10, "complete": True},
            {"type": "cross", "from": 10, "complete": True},
        ]

        derived = compute_derived_stats(events, self.player_stats, match_minutes=90.0, verified_only=False)
        
        self.assertEqual(derived["10"]["derived_stats"]["pass_accuracy_pct"], 75.0)
        # Player 4 has 0 passes -> accuracy should be None
        self.assertIsNone(derived["4"]["derived_stats"]["pass_accuracy_pct"])

    def test_shot_conversion(self):
        """
        Verify shot conversion calculation.
        """
        events = [
            # Player 10: 4 shots, 2 goals -> 50.0% conversion
            {"type": "shot", "player": 10},
            {"type": "shot", "player": 10},
            {"type": "shot", "player": 10},
            {"type": "shot", "player": 10},
            {"type": "goal", "player": 10},
            {"type": "goal", "player": 10},
        ]

        derived = compute_derived_stats(events, self.player_stats, match_minutes=90.0, verified_only=False)
        self.assertEqual(derived["10"]["derived_stats"]["shot_conversion_pct"], 50.0)
        self.assertIsNone(derived["4"]["derived_stats"]["shot_conversion_pct"])

    def test_dribble_success_rate(self):
        """
        Verify dribble success rate calculation.
        """
        events = [
            # Player 10: 3 dribbles, 1 successful -> 33.33% success
            {"type": "dribble", "player": 10, "successful": True},
            {"type": "dribble", "player": 10, "successful": False},
            {"type": "dribble", "player": 10, "successful": False},
        ]

        derived = compute_derived_stats(events, self.player_stats, match_minutes=90.0, verified_only=False)
        self.assertEqual(derived["10"]["derived_stats"]["dribble_success_rate_pct"], 33.33)
        self.assertIsNone(derived["4"]["derived_stats"]["dribble_success_rate_pct"])

    def test_per_90_normalizations(self):
        """
        Verify passes/90 and tackles/90 normalization.
        """
        # Duration: 45 minutes
        # Player 10: 10 passes -> 10 / 45 * 90 = 20.0 passes/90
        # Player 10: 3 tackles -> 3 / 45 * 90 = 6.0 tackles/90
        events = [
            {"type": "pass", "from": 10, "complete": True}] * 10 + [
            {"type": "tackle", "by": 10}] * 3

        derived = compute_derived_stats(events, self.player_stats, match_minutes=45.0, verified_only=False)
        self.assertEqual(derived["10"]["derived_stats"]["passes_per_90"], 20.0)
        self.assertEqual(derived["10"]["derived_stats"]["tackles_per_90"], 6.0)

    def test_verified_only_filtering(self):
        """
        Verify that setting verified_only=True correctly ignores events without status='verified'.
        """
        events = [
            # Verified pass (complete)
            {"type": "pass", "from": 10, "complete": True, "status": "verified"},
            # Unverified pass (complete)
            {"type": "pass", "from": 10, "complete": True, "status": "unverified"},
            # Incomplete pass without status (unverified)
            {"type": "pass", "from": 10, "complete": False}
        ]

        # Case 1: verified_only = False (default: all events counted)
        # Total passes = 3, complete = 2 -> 66.67%
        derived_all = compute_derived_stats(events, self.player_stats, match_minutes=90.0, verified_only=False)
        self.assertEqual(derived_all["10"]["derived_stats"]["pass_accuracy_pct"], 66.67)

        # Case 2: verified_only = True (only status='verified' counted)
        # Total passes = 1, complete = 1 -> 100.0%
        derived_verified = compute_derived_stats(events, self.player_stats, match_minutes=90.0, verified_only=True)
        self.assertEqual(derived_verified["10"]["derived_stats"]["pass_accuracy_pct"], 100.0)

if __name__ == "__main__":
    unittest.main()
