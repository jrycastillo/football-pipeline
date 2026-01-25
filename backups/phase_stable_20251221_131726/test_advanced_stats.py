import unittest
from stats.event_logic import AdvancedEventDetector

class TestAdvancedStats(unittest.TestCase):
    def test_possession_and_pass(self):
        detector = AdvancedEventDetector()
        
        # Mock Tracks: 10 frames
        # Player 10 at (500, 500)
        # Player 7 at (800, 500)
        # Ball moves from Player 10 to Player 7
        
        tracks = []
        ball_track = []
        
        for i in range(10):
            frame = {"boxes": [
                {"id": 10, "xyxy": [490, 490, 510, 510], "cls": 1},
                {"id": 7, "xyxy": [790, 490, 810, 510], "cls": 1}
            ]}
            tracks.append(frame)
            
            # Ball position
            if i < 5:
                # With Player 10
                ball_track.append((500, 500))
            else:
                # Move to Player 7
                ball_track.append((800, 500))
                
        # 1. Ownership
        ownership = detector.calculate_ownership(tracks, ball_track)
        # Frame 0-4: Player 10 (Dist=0)
        # Frame 5-9: Player 7 (Dist=0)
        print("Ownership:", ownership)
        
        # 2. Analyze
        events, stats = detector.analyze(ownership, tracks, ball_track)
        print("Stats[10]:", dict(stats[10]))
        print("Stats[7]:", dict(stats[7]))
        
        self.assertGreater(stats[10]["touch_frames"], 3)
        self.assertGreater(stats[7]["touch_frames"], 3)
        self.assertEqual(stats[10]["passes_total"], 1)
        self.assertEqual(stats[10]["passes_complete"], 1)
        
    def test_xg_shot(self):
        detector = AdvancedEventDetector()
        tracks = []
        ball_track = []
        
        # Shot scenario: Player 9 in box, shoots to goal
        # Box is typically X < 16.5m (approx < 200px in default camera)
        
        for i in range(5):
            tracks.append({"boxes": [{"id": 9, "xyxy": [50, 300, 70, 320], "cls": 1}]})
            ball_track.append((60, 310)) # In box
            
        ownership = detector.calculate_ownership(tracks, ball_track)
        events, stats = detector.analyze(ownership, tracks, ball_track)
        
        print("Stats[9]:", dict(stats[9]))
        self.assertGreater(stats[9]["xg_foot_no_opponent"], 0.0)

if __name__ == "__main__":
    unittest.main()
