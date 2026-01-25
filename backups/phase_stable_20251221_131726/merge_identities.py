import json
import os
import sys
from collections import defaultdict

def merge_identities(input_path="output/raw_tracks.json", output_path="output/players_stats.json"):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, "r") as f:
        raw_tracks = json.load(f)

    print(f"Loaded {len(raw_tracks)} raw tracks.")

    # 1. Group by Entity Key (Team + Jersey)
    groups = defaultdict(list)
    for track in raw_tracks:
        jersey = track.get("jersey_number")
        if jersey is None: continue
        
        team = track.get("team", "Unknown")
        key = f"{team}_{jersey}"
        groups[key].append(track)

    final_players = {}

    # 2. Resolve & Merge
    for key, tracks in groups.items():
        # Sort by start frame
        tracks.sort(key=lambda x: x["frames"][0] if x["frames"] else 0)
        
        valid_tracks = []
        
        # Temporal Conflict Check (Greedy)
        # TODO: Add sophisticated confidence-based resolution.
        # Current Logic: If overlap, keep the longer track (or first one).
        
        timeline = [] # (start, end, track_idx)
        
        for tr in tracks:
            if not tr["frames"]: continue
            start = tr["frames"][0]
            end = tr["frames"][-1]
            
            # Check overlap against accepted timeline
            is_overlap = False
            for (Ts, Te, _) in timeline:
                # Overlap if (Start < Te) and (End > Ts)
                if start < Te and end > Ts:
                    is_overlap = True
                    break
            
            if not is_overlap:
                timeline.append((start, end, tr))
                valid_tracks.append(tr)
            else:
                 print(f"Conflict Resolved: Dropped Track {tr['track_id']} (Overlap) for {key}")

        # Aggregation
        if not valid_tracks: continue
        
        # Base entity on first track
        base = valid_tracks[0]
        merged_stats = base["stats"].copy()
        merged_frames = base["frames"]
        
        # Sum stats from others
        for other in valid_tracks[1:]:
            # Sum numeric stats
            for k, v in other["stats"].items():
                if isinstance(v, (int, float)):
                    merged_stats[k] = merged_stats.get(k, 0) + v
            
            # Merge frames (trajectory)
            merged_frames.extend(other["frames"])
            
        merged_frames.sort()
        
        final_players[str(base["jersey_number"])] = {
            "jersey_number": base["jersey_number"],
            "team": base["team"],
            "position": "Player",
            "role": "Player", # Logic to detect GK?
            "stats": merged_stats,
            # "fragments": [t["track_id"] for t in valid_tracks] # Debug info
        }

    # Save
    with open(output_path, "w") as f:
        json.dump(final_players, f, indent=2)
    
    print(f"Merged {len(raw_tracks)} tracks into {len(final_players)} unique players.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    merge_identities()
