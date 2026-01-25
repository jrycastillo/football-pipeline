import unittest
from stats.post_processor import StatsEngine

class TestStatsEngine(unittest.TestCase):
    def test_pass_detection(self):
        engine = StatsEngine()
        
        # Mock Frames: 2 Players + Ball
        # Player 1 (ID 10) at (100, 100)
        # Player 2 (ID 20) at (500, 100)
        # Ball moves from P1 to P2 over 20 frames
        
        all_frames = []
        
        # Seq 1: P1 has ball (Frames 0-9)
        for i in range(10):
            frame = {
                "boxes": [
                    {"cls": 2, "id": 10, "xyxy": [90, 90, 110, 110], "conf": 0.9}, # P1
                    {"cls": 2, "id": 20, "xyxy": [490, 90, 510, 110], "conf": 0.9}, # P2
                    {"cls": 32, "id": 99, "xyxy": [95, 95, 105, 105], "conf": 0.9}, # Ball at P1
                ],
                "orig_shape": (1080, 1920)
            }
            all_frames.append(frame)
            
        # Seq 2: Ball in flight (Frames 10-14)
        for i in range(5):
            frame = {
                "boxes": [
                    {"cls": 2, "id": 10, "xyxy": [90, 90, 110, 110], "conf": 0.9},
                    {"cls": 2, "id": 20, "xyxy": [490, 90, 510, 110], "conf": 0.9},
                    {"cls": 32, "id": 99, "xyxy": [250, 95, 260, 105], "conf": 0.9}, # Ball mid-air
                ],
                "orig_shape": (1080, 1920)
            }
            all_frames.append(frame)


        # Seq 3: P2 has ball (Frames 15-24)
        for i in range(10):
            frame = {
                "boxes": [
                    {"cls": 2, "id": 10, "xyxy": [90, 90, 110, 110], "conf": 0.9},
                    {"cls": 2, "id": 20, "xyxy": [490, 90, 510, 110], "conf": 0.9},
                    {"cls": 32, "id": 99, "xyxy": [495, 95, 505, 105], "conf": 0.9}, # Ball at P2
                ],
                "orig_shape": (1080, 1920)
            }
            all_frames.append(frame)
            
        # Run Engine
        stats = engine.process_events(all_frames)
        
        print(f"Stats Result: {stats}")
        
        # Verify Pass: P1 should have 'passes' > 0?
        # Logic says: "Possession changed from prev -> own"
        # Prev=10, Own=20.
        # So event: from 10 to 20.
        # Stats[10]["passes"] += 1
        
        self.assertIn(10, stats)
        self.assertGreater(stats[10]["passes"], 0)
        print("Pass Detection test passed!")

if __name__ == "__main__":
    unittest.main()
