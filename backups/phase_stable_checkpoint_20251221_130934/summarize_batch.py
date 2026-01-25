import os
import json
import math
from collections import defaultdict, Counter

OUTPUT_DIR = "output"

def analyze_match_folder(match_dir):
    stats_path = os.path.join(match_dir, "players_stats.json")
    if not os.path.exists(stats_path):
        print(f"  [debug] Not found: {stats_path}")
        return None
        
    with open(stats_path, 'r') as f:
        data = json.load(f)
        
    # 1. Dynamic Team Identification
    # Collect all team colors (excluding 'Unknown')
    color_counts = Counter()
    for pid, p in data.items():
        team = p.get("team", "Unknown")
        if team and team != "Unknown":
            color_counts[team] += 1
            
    # Top 2 Colors
    top_2 = color_counts.most_common(2)
    team_labels = {}
    if len(top_2) >= 2:
        c1, count1 = top_2[0]
        c2, count2 = top_2[1]
        team_labels[c1] = f"Team {c1}"
        team_labels[c2] = f"Team {c2}"
    elif len(top_2) == 1:
        c1, _ = top_2[0]
        team_labels[c1] = f"Team {c1}"
        
    # 2. Find Key Players
    top_scorer = (None, -1.0)
    top_passer = (None, -1)
    
    for pid, p in data.items():
        stats = p.get("stats", {})
        team_raw = p.get("team", "Unknown")
        team_display = team_labels.get(team_raw, team_raw) # Map "Yellow" -> "Team Yellow"
        
        # Calculate xG Total
        xg = stats.get("xg_foot_no_opponent", 0) + stats.get("xg_foot_opponent_present", 0)
        
        if xg > top_scorer[1]:
            top_scorer = (p, xg)
            
        passes = stats.get("passes", 0)
        if passes > top_passer[1]:
            top_passer = (p, passes)
            
    return {
        "name": os.path.basename(match_dir).replace("_", " "),
        "top_scorer": top_scorer,
        "top_passer": top_passer,
        "teams": list(team_labels.values())
    }

def main():
    print(f"{'MATCH':<50} | {'TOP SCORER':<40} | {'TEAMS':<30}")
    print("-" * 120)
    
    matches = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    matches.sort()
    
    for m in matches:
        print(f"Checking {m}...")
        res = analyze_match_folder(os.path.join(OUTPUT_DIR, m))
        if res:
            # Format Scorer
            # Format Scorer
            p, score = res["top_scorer"]
            scorer_str = "None"
            if p:
                t_raw = p.get("team", "Unknown")
                # Re-map team color if possible
                t_final = t_raw
                # We need the team map from the analysis step, but I put it inside the function.
                # Let's just grab the color from the player dict, effectively handled by context.
                # Actually, I should have returned the Label Map or applied it.
                # Let's just print "Player X (Team Color)"
                scorer_str = f"{p['player_name']} ({score:.2f} xG) - {t_final}"
            
            teams_str = ", ".join(res["teams"])
            print(f"{res['name']:<50} | {scorer_str:<40} | {teams_str:<30}")

if __name__ == "__main__":
    main()
