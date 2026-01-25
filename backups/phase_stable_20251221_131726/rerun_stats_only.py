import json
import os
import sys
from stats.post_processor import StatsEngine

def rerun_stats(output_dir):
    print(f"Loading tracking data from {output_dir}...")
    track_path = os.path.join(output_dir, "tracking_data.json")
    with open(track_path, "r") as f:
        data = json.load(f)
        
    if isinstance(data, list):
        all_frames = data
    else:
        all_frames = data.get("history", [])
        
    print(f"Loaded {len(all_frames)} frames.")
    
    # Mock ID Manager
    class MockIdentityManager:
        def __init__(self):
            self.jersey_registry = {}
            self.track_colors = {}
            self.player_colors = {}
        def is_jersey_number(self, pid):
            try:
                return int(pid) < 100
            except:
                return False
        def get_player_color(self, pid):
            return "Unknown"

    mock_id_manager = MockIdentityManager()
    
    # Initialize Stats Engine
    engine = StatsEngine()
    
    # Process
    print("Processing events...")
    formatted_stats, events = engine.process_events(all_frames, id_manager=mock_id_manager)
    
    # Save Raw
    raw_path = os.path.join(output_dir, "raw_players_stats.json")
    with open(raw_path, "w") as f:
        json.dump(formatted_stats, f, indent=2)
    print(f"Saved {raw_path}")
    
    # Save Clean
    clean_path = os.path.join(output_dir, "players_stats.json")
    cleaned = {}
    for pid, s in formatted_stats.items():
        if "Unknown" not in str(pid) and "Unknown" not in str(s["jersey_number"]):
            cleaned[pid] = s
            
    with open(clean_path, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"Saved {clean_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 rerun_stats_only.py <output_dir>")
        sys.exit(1)
    rerun_stats(sys.argv[1])
