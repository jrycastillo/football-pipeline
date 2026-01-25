import json
import os
import sys
import argparse

# Add current directory to path
sys.path.append(os.getcwd())

from vision.visualization import generate_debug_video

def make_video():
    parser = argparse.ArgumentParser(description="Generate Annotated Video")
    parser.add_argument("--video_path", default="/home/ubuntu/videoforprocessing_link/clipped_ikorudo_tornadoes.mp4")
    parser.add_argument("--tracking_data", default="./output/tracking_data.json")
    parser.add_argument("--stats_data", default="./output/players_stats.json")
    parser.add_argument("--output_path", default="./output/annotated_enhanced.mp4")
    
    args = parser.parse_args()

    VIDEO_PATH = args.video_path
    TRACKING_PATH = args.tracking_data
    STATS_PATH = args.stats_data
    OUTPUT_VIDEO = args.output_path

    print("Loading data...")
    if not os.path.exists(TRACKING_PATH):
        print(f"Error: {TRACKING_PATH} not found.")
        return
        
    with open(TRACKING_PATH) as f:
        frames_data = json.load(f)
    
    if not os.path.exists(STATS_PATH):
        print(f"Error: {STATS_PATH} not found.")
        return

    with open(STATS_PATH) as f:
        stats = json.load(f)
        
    # Convert stats to jersey_map (id -> {number: ...})
    # Stats format: {pid: {jersey_number: ..., ...}}
    jersey_map = {}
    for pid, data in stats.items():
        # pid might be string in json
        try:
            pid_int = int(pid)
            jersey_map[pid_int] = {"number": data.get("jersey_number", "?")}
        except:
            pass # handle non-int ids if any

    print(f"Loaded {len(frames_data)} frames of tracking data.")
    print(f"Loaded stats for {len(jersey_map)} players.")
    
    generate_debug_video(VIDEO_PATH, frames_data, jersey_map, OUTPUT_VIDEO)

if __name__ == "__main__":
    make_video()
