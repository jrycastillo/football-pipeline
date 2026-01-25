
import sys
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from collections import defaultdict, namedtuple
import argparse

# --- MOCK ARGS FOR V4 CONFIG ---
class Args:
    def __init__(self):
        # User V4 Config
        self.track_high_thresh = 0.50
        self.track_low_thresh = 0.15
        self.new_track_thresh = 0.90
        self.match_thresh = 0.60
        self.track_buffer = 240
        self.fuse_score = True
        
        # Defaults
        self.tracker_type = "bytetrack"
        self.with_reid = False
        self.mot20 = False
        self.gmc_method = "sparseOptFlow"

def main():
    print("🚀 Starting Tracking Metrics Analysis (V4 Config)...")
    
    # 1. Setup
    args = Args()
    model_path = "/home/ubuntu/videoforprocessing_link/football_analysis_v2/data/models/best_player_detect.pt"
    video_path = "clipped_ikorudo_tornadoes.mp4"
    
    print(f"   Model: {model_path}")
    print(f"   Video: {video_path}")
    print(f"   Params: High={args.track_high_thresh}, New={args.new_track_thresh}, Match={args.match_thresh}, Buffer={args.track_buffer}")

    model = YOLO(model_path)
    tracker = BYTETracker(args, frame_rate=25)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        sys.exit(1)
        
    # 2. Processing Loop
    track_lifetimes = defaultdict(list) # tid -> [frame_indices]
    frame_active_counts = {} # frame_idx -> count
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        # Inference (low thresh to let ByteTrack filter)
        # Using 0.1 to pass everything > 0.1 to ByteTrack (since low_thresh is 0.15)
        results = model(frame, verbose=False, classes=[1, 2], conf=0.1)[0]
        
        # Track
        # Track
        # ByteTrack expects the Boxes object (results.boxes), not the Results object
        if results.boxes is not None and len(results) > 0:
            results = results.cpu()
            tracks = tracker.update(results.boxes, img=frame)
        else:
            tracks = []
        
        # Store Data
        active_ids = []
        for t in tracks:
            if hasattr(t, 'track_id'):
                tid = int(t.track_id)
            else:
                # Assume numpy/list [x1, y1, x2, y2, id, conf, cls]
                # t might be a row
                try:
                    if hasattr(t, 'tolist'): t = t.tolist()
                    tid = int(t[4])
                except Exception as e:
                    # Fallback or skip
                    continue
                    
            active_ids.append(tid)
            track_lifetimes[tid].append(frame_idx)
            
        frame_active_counts[frame_idx] = len(active_ids)
        
        if frame_idx % 100 == 0:
            print(f"   Processed Frame {frame_idx} (Active: {len(active_ids)})")
            
    cap.release()
    print(f"✅ Processing Complete. Total Frames: {frame_idx}")
    
    # 3. Compute Metrics (Steady State: Frame 200+)
    START_ANALYSIS = 200
    if frame_idx <= START_ANALYSIS:
        print("⚠️ Video too short for steady-state analysis.")
        return

    # A. New IDs per 1k Frames
    # Count IDs whose START frame is >= START_ANALYSIS
    new_ids = [tid for tid, frames in track_lifetimes.items() if frames[0] >= START_ANALYSIS]
    count_new_ids = len(new_ids)
    analyzed_duration = frame_idx - START_ANALYSIS
    
    rate_per_1k = (count_new_ids / analyzed_duration) * 1000
    
    # B. Short Tracks (Percentage of ALL tracks)
    # We analyze ALL tracks for lifetime stats, or just steady state?
    # Usually lifetime is global property. Let's filter for tracks that STARTED in steady state window to be fair (or all).
    # Let's use ALL tracks for robust stats, but ignore those cut off at end? 
    # Simplest: Short tracks metric on ALL tracks observed.
    
    short_5 = 0
    short_10 = 0
    total_tracks = len(track_lifetimes)
    
    for frames in track_lifetimes.values():
        duration = len(frames)
        # Handle edge case: tracks starting at end of video shouldn't be penalized?
        # A 2-frame track at the very last frame is ambiguous.
        # But for burstiness, usually short tracks happen anywhere.
        if duration < 5: short_5 += 1
        if duration < 10: short_10 += 1
        
    pct_short_5 = (short_5 / total_tracks) * 100 if total_tracks > 0 else 0
    pct_short_10 = (short_10 / total_tracks) * 100 if total_tracks > 0 else 0
    
    # C. Active Percentiles (Steady State)
    steady_counts = [c for f, c in frame_active_counts.items() if f >= START_ANALYSIS]
    if not steady_counts: steady_counts = [0]
    
    p10 = np.percentile(steady_counts, 10)
    p50 = np.percentile(steady_counts, 50)
    p90 = np.percentile(steady_counts, 90)
    max_active = max(steady_counts)
    
    print("\n" + "="*40)
    print("📊 TRACKING METRICS REPORT (V4 Config)")
    print("="*40)
    print(f"Analysis Window: Frame {START_ANALYSIS} to {frame_idx} ({analyzed_duration} frames)")
    print(f"Total Unique IDs: {total_tracks}")
    print("-" * 20)
    print(f"1. New IDs / 1k Frames:   {rate_per_1k:.2f}  (Target: ≤ 20)")
    print(f"2. Short Tracks (< 5f):   {pct_short_5:.1f}%")
    print(f"3. Short Tracks (< 10f):  {pct_short_10:.1f}%")
    print(f"4. Active Tracks (P10/50/90): {p10:.1f} / {p50:.1f} / {p90:.1f} (Max: {max_active})")
    print("="*40 + "\n")

    # Pass/Fail Check
    if rate_per_1k <= 20:
        print("✅ PASS: NewIDs/1k is within acceptable range.")
    else:
        print(f"❌ FAIL: NewIDs/1k ({rate_per_1k:.1f}) exceeds limit (20).")

if __name__ == "__main__":
    main()
