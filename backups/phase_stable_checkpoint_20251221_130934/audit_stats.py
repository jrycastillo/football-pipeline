import json
import sys
import os

STATS_FILE = "output/players_stats.json"

def load_stats():
    if not os.path.exists(STATS_FILE):
        print(f"[audit] Error: {STATS_FILE} not found.")
        return None
    with open(STATS_FILE, 'r') as f:
        return json.load(f)

def audit(data):
    print("\n[VERIFICATION LEADERBOARD]")
    
    # 1. Identify Roles
    top_scorer = (None, -1.0)
    top_passer = (None, -1)
    top_dribbler = (None, -1)
    gk_found = None
    
    for pid, p in data.items():
        stats = p.get("stats", {})
        
        # xG Sum
        xg = stats.get("xg_foot_no_opponent", 0) + stats.get("xg_foot_opponent_present", 0) + \
             stats.get("xg_header_no_opponent", 0) + stats.get("xg_header_opponent_present", 0)
        
        if xg > top_scorer[1]:
            top_scorer = (p, xg)
            
        # Passes
        passes = stats.get("passes", 0)
        if passes > top_passer[1]:
            top_passer = (p, passes)
            
        # Dribbles
        dribbles = stats.get("dribbles", 0)
        if dribbles > top_dribbler[1]:
            top_dribbler = (p, dribbles)
            
        # GK
        # Check explicit position OR high saves
        if p.get("position") == "GK" or stats.get("shots_saved_total", 0) > 0:
            if gk_found is None or stats.get("shots_saved_total", 0) > gk_found[1]:
                gk_found = (p, stats.get("shots_saved_total", 0))

    # Print Leaderboard
    if top_scorer[0]:
        p = top_scorer[0]
        print(f"Top Scorer (xG): {p['player_name']} ({top_scorer[1]:.2f}) - Team {p['team']}")
        
    if top_passer[0]:
        p = top_passer[0]
        print(f"Top Passer:      {p['player_name']} ({top_passer[1]})   - Team {p['team']}")
        
    if top_dribbler[0]:
        p = top_dribbler[0]
        print(f"Top Dribbler:    {p['player_name']} ({top_dribbler[1]}) - Team {p['team']}")
        
    if gk_found:
        p = gk_found[0]
        print(f"Goalkeeper:      {p['player_name']} ({gk_found[1]} Saves) - Validated")
    else:
        print("Goalkeeper:      None Found (FAILURE)")
        
    print("\n[LOGIC SANITY REPORT]")
    # 2. Logic Checks
    
    # GK Check
    if gk_found:
        p = gk_found[0]
        saves = gk_found[1]
        dribbles = p["stats"].get("dribbles", 0)
        print(f"GK Check: {p['player_name']} -> Saves={saves}, Dribbles={dribbles}")
        if saves == 0 and dribbles > 5:
            print("  -> FAILURE: GK behaving like Outfield Player.")
        else:
            print("  -> PASS")
    else:
        print("GK Check: Skipped (No GK)")
        
    # Striker Check
    if top_scorer[0]:
        p = top_scorer[0]
        num = p.get("jersey_number")
        print(f"Striker Check: Top Scorer is #{num}")
        if num in [7, 9, 10, 11, 19, 20, 18]: # Extended heuristic
            print("  -> PASS (Common Forward Number)")
        elif num is None:
            print("  -> WARNING: Top Scorer has No Jersey Number")
        else:
            print("  -> NOTE: Unconventional Number")

    # Ghost Check
    ghosts = []
    for pid, p in data.items():
        if p.get("jersey_number") is None and p["stats"].get("time_on_ball_s", 0) > 30:
            ghosts.append(f"{p['player_name']} ({p['stats']['time_on_ball_s']}s)")
            
    if ghosts:
        print(f"Ghost Check: Found {len(ghosts)} active ghosts.")
        for g in ghosts[:3]: print(f"  -> {g}")
    else:
        print("Ghost Check: PASS (No active ghosts known)")

if __name__ == "__main__":
    data = load_stats()
    if data:
        audit(data)
