import math
from collections import defaultdict
from .geometry import get_distance, bbox_center

def _inc(d, k, v=1):
    d[k] = d.get(k, 0) + v

def detect_ownership(frames, team_map, max_owner_px=300, challenge_px=100):
    ownership = []
    player_stats = defaultdict(lambda: defaultdict(int))
    team_stats = defaultdict(lambda: defaultdict(int))
    
    for t, f in enumerate(frames):
        players = [b for b in f["boxes"] if b["cls"] in (2, 1) and b["id"] is not None] # 2=player, 1=gk
        balls = [b for b in f["boxes"] if b["cls"] == 0] # 0=ball
        owner = None
        
        if players and balls:
            best_pid = None
            min_dist = 1e9
            best_ball_idx = -1
            
            for b_idx, b in enumerate(balls):
                b_xy = bbox_center(b["xyxy"])
                # Find nearest player
                p_best, d_best = None, 1e9
                for p in players:
                    c = bbox_center(p["xyxy"])
                    d = get_distance(b_xy, c)
                    if d < d_best: p_best, d_best = p, d
                
                if d_best < min_dist:
                    min_dist = d_best
                    best_pid = p_best["id"] if p_best else None
                    best_ball_idx = b_idx
            
            if best_pid is not None and min_dist <= max_owner_px:
                owner = best_pid
                
            # Challenges
            if best_ball_idx != -1:
                b_xy = bbox_center(balls[best_ball_idx]["xyxy"])
                nearby = [p["id"] for p in players if get_distance(b_xy, bbox_center(p["xyxy"])) <= challenge_px]
                teams = {team_map.get(pid) for pid in nearby if team_map.get(pid) is not None}
                if len(teams) >= 2:
                    for pid in nearby:
                        _inc(player_stats[pid], "challenges")
                        tid = team_map.get(pid)
                        if tid is not None: _inc(team_stats[tid], "challenges")

        ownership.append(owner)
        
    # Smoothing
    smoothed = list(ownership)
    last = None
    gap = 0
    for i in range(len(smoothed)):
        if smoothed[i] is not None:
            last = smoothed[i]
            gap = 0
        elif last is not None and gap < 5:
            smoothed[i] = last
            gap += 1
            
    return smoothed, player_stats, team_stats

def detect_passes(frames, ownership, team_map, tackle_px=100):
    events = []
    prev = None
    
    for t, own in enumerate(ownership):
        if own is None: continue
        if prev is None: prev = own; continue
        
        if own != prev:
            ft = team_map.get(prev)
            tt = team_map.get(own)
            e = {
                "type": "pass_attempt", 
                "t": t, 
                "from": prev, 
                "to": own, 
                "from_team": ft, 
                "to_team": tt, 
                "completed": ft == tt
            }
            events.append(e)
            
            # Tackle detection (if teams differ)
            if ft is not None and tt is not None and ft != tt:
                f = frames[min(t, len(frames)-1)]
                # Get positions
                c_from = None
                c_to = None
                for b in f["boxes"]:
                    if b["id"] == prev: c_from = bbox_center(b["xyxy"])
                    if b["id"] == own: c_to = bbox_center(b["xyxy"])
                
                if c_from and c_to:
                    d = get_distance(c_from, c_to)
                    if d <= tackle_px:
                        events.append({
                            "type": "tackle",
                            "t": t,
                            "by": own,
                            "on": prev,
                            "by_team": tt
                        })
        prev = own
    return events

def detect_dribbles(frames, ownership, team_map, dribble_min_px=25, max_gap=25):
    events = []
    start = 0
    for t in range(1, len(ownership) + 1):
        if t == len(ownership) or ownership[t] != ownership[t-1]:
            pid = ownership[t-1]
            end = t - 1
            if pid is not None and end - start + 1 >= 3:
                # Check distance moved
                f0 = frames[start]
                f1 = frames[end]
                c0 = None
                c1 = None
                for b in f0["boxes"]:
                    if b["id"] == pid: c0 = bbox_center(b["xyxy"])
                for b in f1["boxes"]:
                    if b["id"] == pid: c1 = bbox_center(b["xyxy"])
                
                if c0 and c1:
                    dist = get_distance(c0, c1)
                    if dist >= dribble_min_px and (end - start) <= max_gap:
                        events.append({
                            "type": "dribble",
                            "t_start": start,
                            "t_end": end,
                            "pid": pid,
                            "dist": dist
                        })
            start = t
    return events
