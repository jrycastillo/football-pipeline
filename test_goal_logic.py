
import unittest
import numpy as np
import sys
import os

sys.path.append(".")
import football_pipeline as fp

class TestGoalDetection(unittest.TestCase):

    def setUp(self):
        self.team_map = {1: 0}
        self.fps = 25
        self.w = 1000
        self.h = 500
        self.orig_shape = (self.h, self.w)

    def create_dummy_frame(self, ball_pos, players_pos):
        boxes = []
        if ball_pos:
            boxes.append({
                "id": None,
                "cls": fp.CLASS["ball"],
                "conf": 0.9,
                "xyxy": [ball_pos[0]-5, ball_pos[1]-5, ball_pos[0]+5, ball_pos[1]+5]
            })

        for pid, pos in players_pos.items():
            boxes.append({
                "id": pid,
                "cls": fp.CLASS["player"],
                "conf": 0.9,
                "xyxy": [pos[0]-10, pos[1]-20, pos[0]+10, pos[1]+20]
            })

        return {
            "orig_shape": self.orig_shape,
            "boxes": boxes
        }

    def test_goal_vs_save(self):
        # Scenario 1: GOAL
        # Player 1 shoots from (200, 250) to Left Goal (x < 20).
        frames = []
        ownership = []

        # 0-4: Dribble at 200
        for i in range(5):
            frames.append(self.create_dummy_frame((200, 250), {1: (200, 250)}))
            ownership.append(1)

        # 5-20: Shot towards 0. Speed = 200/15 ~ 13 px/frame.
        # Threshold: 0.015 * 1000 = 15.
        # Let's make it faster. 20 px/frame.
        # 5: 180. 6: 160 ...
        for i in range(1, 15): # 14 frames. 200 - 20*14 = -80.
            bx = 200 - i*20
            frames.append(self.create_dummy_frame((bx, 250), {1: (200, 250)}))
            ownership.append(None)

        # Ball enters strip (x <= 20) around frame 5+(200-20)/20 = 5+9 = 14.
        # Frame 14: x=20. Frame 15: x=0.

        # After entering, it disappears (Goal)
        for i in range(10):
             frames.append(self.create_dummy_frame(None, {1: (200, 250)}))
             ownership.append(None)

        shots, _, _ = fp.detect_shots_and_xg(
            frames, ownership, self.team_map, fps=self.fps,
            speed_px_thr_frac=0.015,
            near_goal_frac=1.0,
            opp_thr_px=90
        )

        # Check shots found
        self.assertTrue(len(shots) > 0, "Shot should be detected")
        goal_shot = shots[0]
        self.assertTrue(goal_shot["on_target"], "Shot should be on target")

        # Now check if it's detected as a goal (currently it is NOT)
        # We simulate the logic I want to implement

        # Scenario 2: SAVE
        # Same shot, but ball bounces back.
        frames2 = frames[:19] # Up to frame 19 (x < 0).
        ownership2 = ownership[:19]

        # Frame 19: x = 200 - 14*20 = -80. (Simplified, tracking might lose it, but let's say it bounces)
        # Let's make it bounce back from 0 to 100.
        for i in range(1, 10):
            bx = 0 + i*20 # Back to field
            frames2.append(self.create_dummy_frame((bx, 250), {1: (200, 250)}))
            ownership2.append(None)

        shots2, _, _ = fp.detect_shots_and_xg(
            frames2, ownership2, self.team_map, fps=self.fps,
            speed_px_thr_frac=0.015,
            near_goal_frac=1.0,
            opp_thr_px=90
        )

        self.assertTrue(len(shots2) > 0)
        save_shot = shots2[0]
        self.assertTrue(save_shot["on_target"], "Shot should be on target (even if saved)")

        # Now the critical part: how do we distinguish?
        # We expect the goal shot to have is_goal=True
        self.assertTrue(goal_shot.get("is_goal", False), "Goal not detected")

        # And the save shot to have is_goal=False
        self.assertFalse(save_shot.get("is_goal", False), "Save detected as goal")

    def test_shallow_goal(self):
        # Scenario 3: Shallow Goal (only reaches x=40 on w=1000, i.e. 0.04w)
        # Current threshold is 0.02 (20px). So this should FAIL currently.
        frames = []
        ownership = []

        # Approach
        for i in range(5):
            frames.append(self.create_dummy_frame((200, 250), {1: (200, 250)}))
            ownership.append(1)

        # Shot to 40
        for i in range(1, 9): # 200 - 8*20 = 40.
            bx = 200 - i*20
            frames.append(self.create_dummy_frame((bx, 250), {1: (200, 250)}))
            ownership.append(None)

        # Stays at 40 (in net)
        for i in range(10):
            frames.append(self.create_dummy_frame((40, 250), {1: (200, 250)}))
            ownership.append(None)

        shots, _, _ = fp.detect_shots_and_xg(
            frames, ownership, self.team_map, fps=self.fps,
            speed_px_thr_frac=0.015,
            near_goal_frac=1.0,
            opp_thr_px=90
        )

        self.assertTrue(len(shots) > 0)
        # This assertion expects True. It will fail if threshold is 0.02.
        self.assertTrue(shots[0].get("is_goal", False), "Shallow goal (0.04w) not detected")

if __name__ == '__main__':
    unittest.main()
