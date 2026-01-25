import json
import os

path = "output/test_ikorudo_refinement/players_stats.json"
if os.path.exists(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    fixed_count = 0
    for pid, v in data.items():
        if v.get("team") == "Unknown":
            v["team"] = "White" # Force Fallback as per Phase 81
            fixed_count += 1
            
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Fixed {fixed_count} Unknown teams.")
else:
    print("File not found.")
