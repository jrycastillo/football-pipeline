
import unittest
import numpy as np
import math
from collections import defaultdict
import sys
import os
import cv2

sys.path.append(".")
import football_pipeline as fp

class TestFootballPipeline(unittest.TestCase):

    def setUp(self):
        self.team_map = {1: 0, 2: 0, 3: 1, 4: 1}
        self.fps = 25
        self.w = 1920
        self.h = 1080
        self.orig_shape = (self.h, self.w)

    def create_dummy_frame(self, ball_pos, players_pos):
        boxes = []
        if ball_pos:
            boxes.append({
                "id": None,
                "cls": fp.CLASS["ball"],
                "conf": 0.9,
                "xyxy": [ball_pos[0]-10, ball_pos[1]-10, ball_pos[0]+10, ball_pos[1]+10]
            })

        for pid, pos in players_pos.items():
            boxes.append({
                "id": pid,
                "cls": fp.CLASS["player"],
                "conf": 0.9,
                "xyxy": [pos[0]-20, pos[1]-40, pos[0]+20, pos[1]+40]
            })

        return {
            "orig_shape": self.orig_shape,
            "boxes": boxes
        }

    def test_shot_detection_with_max_owner_px(self):
        # Scenario: Player 1 kicks the ball hard.
        # We enforce max_owner_px so ownership is lost immediately when ball leaves foot.

        frames = []
        # 1. Dribble/Hold phase
        for i in range(5):
            frames.append(self.create_dummy_frame((100, 540), {1: (100, 540)}))

        # 2. Kick phase
        # Frame 5: Ball at 100.
        # Frame 6: Ball at 200. (Speed 100 px/frame). Dist to player (at 100) is 100.
        # If max_owner_px = 50, ownership is LOST at frame 6.
        frames.append(self.create_dummy_frame((200, 540), {1: (100, 540)})) # v=100

        # Frame 7: Ball at 300.
        frames.append(self.create_dummy_frame((300, 540), {1: (100, 540)})) # v=100

        # Frame 8... towards goal (goal at x=0 or x=1920). Moving towards 1920.
        # Let's make it travel fast to goal.
        for i in range(20):
            bx = 300 + (i+1)*50
            frames.append(self.create_dummy_frame((bx, 540), {1: (100, 540)}))

        # Calculate Phase 1 stats to get ownership
        p1 = fp.phase1_stats(frames, self.team_map, fps=self.fps, max_owner_px=50)
        ownership = p1["ownership"]

        # ownership[0..4] should be 1.
        # ownership[5] (ball at 200, player at 100, dist 100 > 50) should be None.
        self.assertEqual(ownership[4], 1)
        self.assertIsNone(ownership[5])

        # Now detect shots
        shots, _, _ = fp.detect_shots_and_xg(
            frames, ownership, self.team_map, fps=self.fps,
            speed_px_thr_frac=0.015, # 1920 * 0.015 = 28.8 px
            near_goal_frac=1.0, # Allow shots from far away for test
            opp_thr_px=90
        )

        # Logic check:
        # Segment of ownership ends at index 4 (Frame 4).
        # range [lo, hi] = [start, 4].
        # v[0..4] are 0.
        # v[5] (calculated from frame 4 to 5) is 100.
        # Is v[5] checked?
        # t_star = max(range(lo, hi + 1), key=lambda k: v[k])
        # range is up to 4+1 = 5? No, python range(a, b) excludes b.
        # code says: `t_star = max(range(lo, hi + 1), key=lambda k: v[k])`
        # `hi = min(end, len(frames) - 1)`
        # `end = t - 1`.
        # Here `t` is where ownership changes.
        # ownership[4]=1, ownership[5]=None.
        # Loop: t=5. ownership[5] != ownership[4].
        # pid = 1. end = 4.
        # hi = 4.
        # range(lo, 5) -> indices 0, 1, 2, 3, 4.
        # v[4] is velocity at frame 4 (using frame 3 and 4).
        # v[5] is velocity at frame 5 (using frame 4 and 5). This is the KICK velocity.
        # v[5] is NOT checked because the range stops at 4.

        # So the shot might NOT be detected if the kick happens exactly when ownership is lost?
        # Unless v[4] was already high? In my data v[4] is 0.

        print(f"Detected shots: {len(shots)}")
        if shots:
            print(f"Shot details: {shots[0]}")

        self.assertTrue(len(shots) > 0, "Shot should be detected but likely missed due to range issue")

if __name__ == '__main__':
    unittest.main()
