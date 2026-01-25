import json
import os
import sys
import subprocess

def get_best_clips(events_file):
    with open(events_file, 'r') as f:
        events = json.load(f)
        
    # 1. Best Shot/Goal (Highest xG)
    best_xg = 0
    best_shot_frame = None
    
    # 2. Best Save
    best_save_frame = None
    
    # 3. Passing Chain (Consecutive passes)
    # Heuristic: Find window with most 'pass' events
    # Simplified: Just grab a cluster
    # Or just grab the first 'goal' or high xG shot
    
    for e in events:
        if e["type"] == "shot" or e["type"] == "goal":
            xg = e.get("xg", 0)
            if xg > best_xg:
                best_xg = xg
                best_shot_frame = e["frame"]
                
        if e["type"] == "save":
            best_save_frame = e["frame"]

    # Passing Chain: Find max passes in 10s window (250 frames)
    # Sort events by frame
    sorted_events = sorted([e for e in events if e["type"] == "pass"], key=lambda x: x["frame"])
    max_passes = 0
    best_chain_frame = None
    
    if sorted_events:
        for i in range(len(sorted_events)):
            count = 0
            start_f = sorted_events[i]["frame"]
            for j in range(i, len(sorted_events)):
                if sorted_events[j]["frame"] - start_f < 250: # 10s
                    count += 1
                else:
                    break
            if count > max_passes:
                max_passes = count
                best_chain_frame = start_f

    return best_shot_frame, best_save_frame, best_chain_frame

def render(video_path, out_dir, frame_idx, label):
    if frame_idx is None:
        print(f"[render] No event found for {label}")
        return
        
    start_sec = max(0, (frame_idx / 25.0) - 5.0)
    end_sec = (frame_idx / 25.0) + 5.0
    duration = end_sec - start_sec
    
    out_name = os.path.join(out_dir, f"validation_{label}.mp4")
    
    # Use ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        out_name
    ]
    
    print(f"[render] Generating {out_name} (Time: {start_sec:.1f}s)")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 render_clips.py <video_path> <events_json>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    events_json = sys.argv[2]
    out_dir = os.path.dirname(events_json)
    
    shot_f, save_f, chain_f = get_best_clips(events_json)
    
    render(video_path, out_dir, shot_f, "goal")
    render(video_path, out_dir, save_f, "save")
    render(video_path, out_dir, chain_f, "pass")
