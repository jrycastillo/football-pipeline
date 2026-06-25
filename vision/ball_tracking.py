import numpy as np
import yaml

# Load Config
try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    CLASS_BALL = CONFIG["classes"].get("ball", 0)
except:
    CLASS_BALL = 0

def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

class BallTracker:
    def __init__(self):
        self.tracks = {} # frame_idx -> (x, y)
        self.raw_detections = {} # frame_idx -> box
        self.max_gap = 30 # Round 2 fix: 50→30 (50 frames at VID_STRIDE=5 = 10s, too long for linear interp)

    def update(self, frame_idx, boxes):
        """
        Update ball tracker with detections from a frame.
        """
        # Filter for ball class (0=Custom, 32=COCO)
        balls = [b for b in boxes if b["cls"] in [0, 32]]
        
        if not balls:
            return
            
        # Select best ball (highest confidence)
        # TODO: Could add proximity logic to previous ball position
        best_ball = max(balls, key=lambda x: x.get("conf", 0.0))
        
        self.raw_detections[frame_idx] = best_ball
        self.tracks[frame_idx] = bbox_center(best_ball["xyxy"])

    def interpolate(self, total_frames):
        """
        Fill gaps in ball tracking using linear interpolation.
        Returns a list of (x, y) or None for each frame.
        """
        final_track = [None] * total_frames
        
        # Fill knowns
        for idx, pos in self.tracks.items():
            if idx < total_frames:
                final_track[idx] = pos
                
        # Interpolate
        # Use pandas-like ffill/bfill logic or simple loop
        # For strict physics, maybe Kalman, but Linear is fine for stats
        
        last_idx = -1
        for i in range(total_frames):
            if final_track[i] is not None:
                if last_idx != -1 and (i - last_idx) <= self.max_gap:
                    # Interpolate
                    start_pos = final_track[last_idx]
                    end_pos = final_track[i]
                    steps = i - last_idx
                    dx = (end_pos[0] - start_pos[0]) / steps
                    dy = (end_pos[1] - start_pos[1]) / steps
                    
                    for j in range(1, steps):
                        final_track[last_idx + j] = (start_pos[0] + dx * j, start_pos[1] + dy * j)
                last_idx = i
                
        return final_track
