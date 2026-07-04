import sys
import types
import unittest
from collections import defaultdict

sys.modules["cv2"] = types.ModuleType("cv2")

from stats.metrics import StatsEngine


def _stats(**values):
    out = defaultdict(int)
    out.update(values)
    return out


class FakeDetector:
    def calculate_ownership(self, player_tracks, ball_track):
        return [None] * len(ball_track)

    def analyze(self, ownership, player_tracks, ball_track, team_map=None, raw_ball_frames=None):
        events = [
            {"type": "pass", "from": 100, "to": 200, "frame": 0, "confidence": 0.80},
            {"type": "shot", "player": 300, "frame": 1, "confidence": 0.70},
            {"type": "save", "player": 400, "frame": 2, "confidence": 0.70},
            {"type": "foul", "by": 999, "on": 10, "frame": 3, "confidence": 0.40},
        ]
        raw_stats = {
            100: _stats(passes_total=5, passes_complete=3, touch_frames=10),
            200: _stats(passes_total=7, passes_complete=4, touch_frames=10),
            300: _stats(passes_total=2, passes_complete=1, touch_frames=10),
            400: _stats(passes_total=1, passes_complete=1, touch_frames=10),
            999: _stats(passes_total=1, passes_complete=0, touch_frames=10),
        }
        return events, raw_stats


class FakeIdentityManager:
    def __init__(self):
        self.active_bindings = {100: 10, 200: 20, 400: 40}
        self.jersey_registry = {
            10: {"track_id": 100},
            20: {"track_id": 200},
            40: {"track_id": 400},
        }
        self.vote_counts = {
            100: {10: 3.2, 11: 1.0},
            200: {20: 1.2, 21: 1.0},
            400: {40: 3.2, 41: 1.0},
        }
        self.vote_tallies = {
            100: {10: 3},
            200: {20: 2},
            400: {40: 3},
        }
        self.vote_recovered_jerseys = {20}
        self.raw_read_counts = {300: {30: 20}}
        self.track_colors = {}
        self.player_colors = {}

    def is_jersey_number(self, value):
        try:
            return int(value) in self.jersey_registry
        except (TypeError, ValueError):
            return False

    def get_track_color(self, track_id):
        return self.track_colors.get(track_id, "Unknown")

    def get_player_color(self, jersey):
        return self.player_colors.get(jersey, "Unknown")


class FakeRoster:
    def is_valid_number(self, number, team_name=None):
        return int(number) in {10, 20, 30}


def _frame():
    return {
        "boxes": [
            {"cls": 2, "id": 100, "xyxy": [90, 90, 110, 130], "conf": 0.9},
            {"cls": 2, "id": 200, "xyxy": [190, 90, 210, 130], "conf": 0.9},
            {"cls": 2, "id": 300, "xyxy": [290, 90, 310, 130], "conf": 0.9},
            {"cls": 2, "id": 400, "xyxy": [390, 90, 410, 130], "conf": 0.9},
            {"cls": 2, "id": 999, "xyxy": [490, 90, 510, 130], "conf": 0.9},
            {"cls": 32, "id": None, "xyxy": [10, 10, 16, 16], "conf": 0.9},
        ],
        "orig_shape": (720, 1280),
    }


class IdentityConfidenceTest(unittest.TestCase):
    def test_identity_confidence_tiers_and_stats_are_preserved(self):
        engine = StatsEngine()
        engine.detector = FakeDetector()
        engine._cluster_teams = lambda *args, **kwargs: None

        formatted_stats, events = engine.process_events(
            [_frame() for _ in range(40)],
            id_manager=FakeIdentityManager(),
            roster_prior=FakeRoster(),
        )

        by_type = {event["type"]: event for event in events}
        self.assertEqual(0.95, by_type["pass"]["identity_confidence"])
        self.assertEqual(0.65, by_type["pass"]["identity_confidence_receiver"])
        self.assertEqual(0.50, by_type["shot"]["identity_confidence"])
        self.assertEqual(0.50, by_type["save"]["identity_confidence"])
        self.assertEqual(0.20, by_type["foul"]["identity_confidence"])

        self.assertEqual(5, formatted_stats["10"]["stats"]["passes_total"])
        self.assertEqual(7, formatted_stats["20"]["stats"]["passes_total"])
        self.assertEqual(2, formatted_stats["30"]["stats"]["passes_total"])
        self.assertEqual(1, formatted_stats["40"]["stats"]["passes_total"])
        self.assertEqual(1, formatted_stats["Unknown_999"]["stats"]["passes_total"])


if __name__ == "__main__":
    unittest.main()
