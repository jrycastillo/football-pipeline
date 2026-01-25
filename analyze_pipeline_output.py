import json
import numpy as np
import sys
import os

def analyze_metrics(json_path, start_analysis_frame=200):
    print(f"🚀 Analyzing Pipeline Output: {json_path}")
    
    with open(json_path, 'r') as f:
        all_frames = json.load(f)
        
    total_frames = len(all_frames)
    print(f"Total Frames: {total_frames}")
    
    if total_frames <= start_analysis_frame:
        print("⚠️ Video too short for steady-state analysis.")
        return

    # 1. Build Track Lifetimes
    track_lifetimes = {} # tid -> [f_start, f_end, count]
    frame_active_counts = {}
    
    for i, frame in enumerate(all_frames):
        frame_idx = frame.get("frame_index", i) # Use index if key missing
        boxes = frame.get("boxes", [])
        
        active_ids = set()
        for box in boxes:
            tid = box.get("id")
            if tid is None: continue
            
            active_ids.add(tid)
            
            if tid not in track_lifetimes:
                track_lifetimes[tid] = [frame_idx, frame_idx, 1]
            else:
                track_lifetimes[tid][1] = frame_idx # Update end
                track_lifetimes[tid][2] += 1        # Update count
                
        frame_active_counts[frame_idx] = len(active_ids)

    # 2. Steady State Analysis
    analyzed_duration = total_frames - start_analysis_frame
    
    # A. New IDs / 1k Frames (Count IDs starting >= start_analysis_frame)
    new_ids_count = 0
    for tid, stats in track_lifetimes.items():
        start_frame = stats[0]
        if start_frame >= start_analysis_frame:
            new_ids_count += 1
            
    rate_per_1k = (new_ids_count / analyzed_duration) * 1000
    
    # B. Short Tracks
    short_5 = 0
    short_10 = 0
    total_tracks = len(track_lifetimes)
    
    for tid, stats in track_lifetimes.items():
        duration = stats[2] # Count of frames present
        # Alternatively: duration = stats[1] - stats[0] + 1 (span)
        # "Short track" usually refers to span or active count. Let's use count.
        if duration < 5: short_5 += 1
        if duration < 10: short_10 += 1
        
    pct_short_5 = (short_5 / total_tracks) * 100 if total_tracks > 0 else 0
    pct_short_10 = (short_10 / total_tracks) * 100 if total_tracks > 0 else 0
    
    # C. Active Percentiles
    steady_counts = [c for f, c in frame_active_counts.items() if f >= start_analysis_frame]
    if not steady_counts: steady_counts = [0]
    
    p10 = np.percentile(steady_counts, 10)
    p50 = np.percentile(steady_counts, 50)
    p90 = np.percentile(steady_counts, 90)
    max_active = max(steady_counts)
    
    print("\n" + "="*40)
    print("📊 PIPELINE METRICS REPORT")
    print("="*40)
    print(f"Analysis Window: Frame {start_analysis_frame} to {total_frames} ({analyzed_duration} frames)")
    print(f"Total Unique IDs: {total_tracks}")
    print("-" * 20)
    print(f"1. New IDs / 1k Frames:   {rate_per_1k:.2f}  (Target: ≤ 20)")
    print(f"2. Short Tracks (< 5f):   {pct_short_5:.1f}%")
    print(f"3. Short Tracks (< 10f):  {pct_short_10:.1f}%")
    print(f"4. Active Tracks (P10/50/90): {p10:.1f} / {p50:.1f} / {p90:.1f} (Max: {max_active})")
    print("="*40 + "\n")
    
    if rate_per_1k <= 20:
        print("✅ PASS")
    else:
        print("❌ FAIL")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_pipeline_output.py <path_to_debug_all_frames.json>")
        # Default fallback
        path = "output/debug_all_frames.json"
        if os.path.exists(path):
            analyze_metrics(path)
        else:
            print(f"File not found: {path}")
    else:
        analyze_metrics(sys.argv[1])
