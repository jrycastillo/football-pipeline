import json
import os
import shutil

STATS_PATH = "output_visual_test/player_stats.json"
BACKUP_PATH = "output_visual_test/player_stats_backup.json"

if not os.path.exists(STATS_PATH):
    print(f"File not found: {STATS_PATH}")
    exit(1)

# Backup
if not os.path.exists(BACKUP_PATH):
    shutil.copy(STATS_PATH, BACKUP_PATH)
    print(f"Backed up to {BACKUP_PATH}")

with open(STATS_PATH, "r") as f:
    data = json.load(f)

original_count = len(data)
cleaned_data = {}

for pid, stats in data.items():
    # Logic: Keep if it has a valid Jersey Number AND Role is not generic unless known
    # Actually, simplistic logic: Drop if key starts with "Unknown" or jersey_number is null
    
    is_unknown_key = str(pid).startswith("Unknown")
    jersey_num = stats.get("jersey_number")
    
    if is_unknown_key and jersey_num is None:
        continue # Drop
        
    cleaned_data[pid] = stats

print(f"Original: {original_count} -> Cleaned: {len(cleaned_data)}")

with open(STATS_PATH, "w") as f:
    json.dump(cleaned_data, f, indent=2)

print("Done.")
