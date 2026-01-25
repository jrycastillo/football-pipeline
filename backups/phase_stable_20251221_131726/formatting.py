import json
import csv
from collections import defaultdict

def format_and_save(events, player_stats, team_stats, shots, xg_player, team_map, team_labels, jersey_map, output_path, csv_path, all_tracked_ids=None):
    players_flat = []
    
    # Merge all stats
    all_pids = set(player_stats.keys()) | set(xg_player.keys())
    if all_tracked_ids:
        all_pids = all_pids | set(all_tracked_ids)
    
    for pid in all_pids:
        stats = player_stats[pid]
        xg = xg_player[pid]
        
        tid = team_map.get(pid)
        team_name = team_labels.get(tid, "unknown")
        
        # Jersey
        j_info = jersey_map.get(pid, {})
        j_num = j_info.get("number", "Unknown")
        
        row = {
            "team": team_name,
            "player_id": int(pid),
            "jersey_number": j_num,
            
            "passes": stats["passes_total"],
            "accurate_passes_%": (100.0 * stats["passes_completed"] / stats["passes_total"]) if stats["passes_total"] > 0 else 0.0,
            
            "challenges": stats.get("challenges", 0), # Need to implement challenge logic in logic.py if missing
            "challenges_won": stats.get("challenges_won", 0),
            
            "tackles": stats["tackles_total"],
            
            "xg_total": xg.get("xg_total", 0.0),
            "goals": stats.get("goals_final", 0)
        }
        players_flat.append(row)
        
    # Save JSON
    payload = {
        "players_flat": players_flat,
        "team_labels": team_labels,
        "events": events
    }
    
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
        
    # Save CSV
    if players_flat:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(players_flat[0].keys()))
            writer.writeheader()
            writer.writerows(players_flat)
            
    return payload
