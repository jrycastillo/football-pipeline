import json
import csv
import sys
import os

def flatten_player(key, data):
    # Flatten the nested structure
    # Key is ID or "Unknown_ID"
    row = {
        "id": key,
        "name": data.get("player_name", ""),
        "jersey": data.get("jersey_number", ""),
        "team": data.get("team", ""),
        "position": data.get("position", ""),
        "role": data.get("role", "")
    }
    
    stats = data.get("stats", {})
    for k, v in stats.items():
        row[k] = v
        
    return row

def convert(json_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    rows = []
    for k, v in data.items():
        rows.append(flatten_player(k, v))
        
    if not rows:
        print("No data found.")
        return

    # Determine headers dynamically
    headers = list(rows[0].keys())
    
    # Ensure specific order if desired, or just sort
    # Let's put ID, Name, Team, Jersey first
    priority = ["id", "name", "jersey", "team", "position", "time_on_ball_s", "xg_foot_no_opponent", "goals", "passes", "shots_saved_total"]
    
    # Simple sort based on priority
    def head_sort(h):
        if h in priority:
            return priority.index(h)
        return 100
        
    headers.sort(key=head_sort)

    out_path = json_path.replace(".json", ".csv")
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"CSV saved to {out_path}")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 json_to_csv.py <path_to_json>")
        sys.exit(1)
        
    convert(sys.argv[1])
