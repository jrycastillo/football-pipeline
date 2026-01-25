import os
import json
import glob
from collections import Counter

def sanitize_file(filepath):
    print(f"[sanitize] Processing {filepath}...")
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            
        cleaned_data = []
        
        if isinstance(data, dict):
            items = list(data.values())
        elif isinstance(data, list):
            items = data
        else:
            print(f"[sanitize] Error: Unknown JSON structure in {filepath}")
            return

        # 1. Identify Teams (Color Clustering)
        color_counts = Counter()
        for p in items:
            # Handle different structures (list vs dict)
            team = p.get("team", "Unknown")
            if team and team.lower() != "unknown" and team.lower() != "referee":
                color_counts[team] += 1
                
        top_2 = color_counts.most_common(2)
        if len(top_2) < 2:
            print(f"[sanitize] WARNING: Less than 2 teams found in {filepath}. Colors: {top_2}")
            # continue? Or try to process anyway.
            # We'll map the one team to 0.
            
        team_map = {} # color -> team_id
        if len(top_2) >= 1: team_map[top_2[0][0]] = 0
        if len(top_2) >= 2: team_map[top_2[1][0]] = 1
        
        # Optional: Name Inference from Filename
        # output/Angola_vs_Namibia/players_stats.json
        parent_dir = os.path.basename(os.path.dirname(filepath))
        team_names = {0: "Home Team", 1: "Away Team"}
        
        if "_vs_" in parent_dir.lower():
            parts = parent_dir.replace("_", " ").split(" vs ")
            if len(parts) >= 2:
                team_names[0] = parts[0].strip()
                team_names[1] = parts[1].strip()
        
        removed_count = 0
        
        for p in items:
            # 2. Filter Unknowns
            jnr = str(p.get("jersey_number", "Unknown"))
            if jnr.lower() in ["unknown", "null", "none", "-1"]:
                removed_count += 1
                continue
                
            # Filter Noise (Distance)
            # If total_distance key missing, assume 0
            dist = float(p.get("total_distance", 0))
            if dist < 50:
                # Check if they have other stats (e.g. goals) to be safe?
                # User said "Remove any player with total_distance < 50".
                removed_count += 1
                continue
                
            # 3. Fix Team Identity
            raw_team = p.get("team", "Unknown")
            if raw_team in team_map:
                tid = team_map[raw_team]
                p["team_id"] = tid
                p["team_name"] = team_names[tid] # New Field
                # p["team"] = team_names[tid] # Overwrite? Or keep color?
                # User: "Set team_name = ...". 
                # User also said: "Instead of relying on detected jersey_color string... use team_id".
                # I will keep 'team' as color but add 'team_id' and 'team_name'.
            else:
                p["team_id"] = -1
                p["team_name"] = "Unknown"
            
            cleaned_data.append(p)
            
        # Re-construct output structure
        if isinstance(data, dict):
            # We need keys. If original keys were track IDs, we might lose them if we just use cleaned_data list.
            # But wait, usually we want to preserve the keys.
            # Let's rebuild the dict using player_id as key if possible, or just save as list if user doesn't care?
            # User wants "Overwrite".
            # If I convert Dict -> List, it changes the schema.
            # I should try to preserve keys.
            # Modification: Filter the keys from original 'data'.
            final_output = {}
            for k, v in data.items():
                 # Check if v is in cleaned_data (identity)
                 # Or check criteria again? Duplicate logic is bad.
                 # Better: Do the filtering on the dict directly or rebuild.
                 # Let's assume 'player_id' or 'jersey_number' isn't key. 'track_id' is key.
                 # I'll just iterate keys.
                 pass
        
        # RE-WRITE for robustness:
        final_output = []
        if isinstance(data, list):
             final_output = cleaned_data
        else:
             # Dict mode: Re-filter with keys
             final_output = {}
             # Re-run logic on keys?? inefficient.
             # Actually, since I modified 'p' in place inside 'items' (mutable objects), 
             # the objects in 'data' are already modified!
             # So I just need to filter out the dropped ones.
             # But 'cleaned_data' holds the kept objects.
             # I need to know which key corresponded to which object.
             # In Python, 'items = list(data.values())' creates a list of references.
             # So 'p' is a reference. 'p["team_id"] = ...' modified the object in 'data'.
             
             # To filter, I need to know which keys to keep.
             # I can map object id() to key?
             obj_to_key = {id(v): k for k, v in data.items()}
             
             for p in cleaned_data:
                 if id(p) in obj_to_key:
                     final_output[obj_to_key[id(p)]] = p
                     
        # 4. Save
        with open(filepath, "w") as f:
            json.dump(final_output, f, indent=2)
            
        print(f"[sanitize] Cleaned {filepath}. Removed {removed_count} players. mapped teams: {team_map}")
        
    except Exception as e:
        print(f"[sanitize] Error processing {filepath}: {e}")

def main():
    # Find all players_stats.json in output subdirectories
    files = glob.glob("output/*/players_stats.json")
    print(f"[sanitize] Found {len(files)} files to process.")
    
    for f in files:
        sanitize_file(f)

if __name__ == "__main__":
    main()
