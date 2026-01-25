import math
import numpy as np
import yaml
from collections import defaultdict

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CLASS = CONFIG["classes"]
HEURISTICS = CONFIG["heuristics"]

def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _inc(d, k, v=1):
    d[k] = d.get(k, 0) + v

def _center_at(f, pid):
    for b in f["boxes"]:
        if b["id"] == pid: return bbox_center(b["xyxy"])
    return None

def calculate_ownership(frames, team_map):
    ownership = []
    max_owner_px = HEURISTICS["MAX_OWNER_PX"]
    
    for f in frames:
        players = [b for b in f["boxes"] if b["cls"] in (CLASS["player"], CLASS["goalkeeper"]) and b["id"] is not None]
        balls = [b for b in f["boxes"] if b["cls"] == CLASS["ball"]]
        owner = None
        
        if players and balls:
            best_pid = None
            min_dist = 1e9
            
            # Find closest player to ANY ball
            for b in balls:
                b_xy = bbox_center(b["xyxy"])
                for p in players:
                    c = bbox_center(p["xyxy"])
                    d = math.hypot(b_xy[0] - c[0], b_xy[1] - c[1])
                    if d < min_dist:
                        min_dist = d
                        best_pid = p["id"]
            
            if best_pid is not None and min_dist <= max_owner_px:
                owner = best_pid
                
        ownership.append(owner)
        
    # Smooth ownership
    # 1. Forward fill (gap < 5)
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
            
    # 2. Remove short segments (< 3)
    final_ownership = []
    if smoothed:
        curr = smoothed[0]
        count = 1
        segments = []
        for pid in smoothed[1:]:
            if pid == curr: count += 1
            else:
                segments.append({"pid": curr, "count": count})
                curr = pid; count = 1
        segments.append({"pid": curr, "count": count})
        
        for i in range(1, len(segments)):
            if segments[i]["count"] < 3 and segments[i-1]["pid"] is not None:
                segments[i]["pid"] = segments[i-1]["pid"]
                
        for seg in segments:
            final_ownership.extend([seg["pid"]] * seg["count"])
            
    return final_ownership if final_ownership else smoothed

def detect_passes_and_events(frames, ownership, team_map):
    events = []
    player_stats = defaultdict(lambda: defaultdict(int))
    team_stats = defaultdict(lambda: defaultdict(int))
    
    tackle_px = HEURISTICS["TACKLE_PX"]
    
    prev = None
    for t, own in enumerate(ownership):
        if own is None: continue
        
        # Touch frames
        _inc(player_stats[own], "touch_frames")
        tid = team_map.get(own)
        if tid is not None: _inc(team_stats[tid], "touch_frames")
        
        if prev is None: prev = own; continue
        
        if own != prev:
            ft = team_map.get(prev)
            tt = team_map.get(own)
            completed = (ft == tt) if (ft is not None and tt is not None) else False
            
            e = {
                "type": "pass_attempt", 
                "t": t, 
                "from": prev, 
                "to": own, 
                "from_team": ft, 
                "to_team": tt, 
                "completed": completed
            }
            events.append(e)
            
            # Tackle Detection
            if ft is not None and tt is not None and ft != tt:
                f = frames[min(t, len(frames)-1)]
                c_from = _center_at(f, prev)
                c_to = _center_at(f, own)
                if c_from and c_to:
                    d = math.hypot(c_from[0] - c_to[0], c_from[1] - c_to[1])
                    if d <= tackle_px:
                        events.append({
                            "type": "tackle", 
                            "t": t, 
                            "by": own, 
                            "on": prev, 
                            "by_team": tt
                        })
                        _inc(player_stats[own], "challenges_won")
                        if tt is not None: _inc(team_stats[tt], "challenges_won")
        prev = own
        
    # Aggregate Stats
    for e in events:
        if e["type"] == "pass_attempt":
            _inc(player_stats[e["from"]], "passes_total")
            if e["completed"]: _inc(player_stats[e["from"]], "passes_completed")
            
            if e["from_team"] is not None:
                _inc(team_stats[e["from_team"]], "passes_total")
                if e["completed"]: _inc(team_stats[e["from_team"]], "passes_completed")
                
        elif e["type"] == "tackle":
            _inc(player_stats[e["by"]], "tackles_total")
            if e["by_team"] is not None: _inc(team_stats[e["by_team"]], "tackles_total")
            
    return events, player_stats, team_stats

def detect_shots_and_xg(frames, ownership, team_map):
    # Simplified port of legacy logic
    h, w = frames[0]["orig_shape"]
    fps = HEURISTICS["FPS"]
    speed_thr = HEURISTICS["SPEED_PX_THR_FRAC"] * w
    near_goal_dist = HEURISTICS["NEAR_GOAL_FRAC"] * w
    
    ball = [None] * len(frames)
    for t, f in enumerate(frames):
        balls = [b for b in f["boxes"] if b["cls"] == CLASS["ball"]]
        ball[t] = bbox_center(balls[0]["xyxy"]) if balls else None
        
    v = [0.0] * len(frames)
    for t in range(1, len(frames)):
        if ball[t] and ball[t-1]:
            v[t] = math.hypot(ball[t][0] - ball[t-1][0], ball[t][1] - ball[t-1][1])
            
    shots = []
    xg_player = defaultdict(lambda: defaultdict(float))
    
    start = 0
    for t in range(1, len(ownership) + 1):
        if t == len(ownership) or ownership[t] != ownership[t-1]:
            pid = ownership[t-1]
            end = t - 1
            if pid is not None:
                lo = max(start, 1)
                hi = min(end + 3, len(frames) - 1)
                if hi >= lo:
                    t_star = max(range(lo, hi + 1), key=lambda k: v[k])
                    if ball[t_star] and v[t_star] >= speed_thr:
                        # Check distance to goal
                        # Goal centers
                        gl = np.array([0.12 * w, 0.50 * h])
                        gr = np.array([0.88 * w, 0.50 * h])
                        pt = np.array(ball[t_star])
                        dist = min(np.linalg.norm(pt - gl), np.linalg.norm(pt - gr))
                        
                        if dist <= near_goal_dist:
                            # It's a shot candidate
                            # Calculate xG (Simplified)
                            dist_norm = max(0.0, min(1.0, dist / (HEURISTICS["XG_DIST_NORM_W"] * w)))
                            xg = 1.0 / (1.0 + math.exp(-(-3.2 + 4.8 * (1.0 - dist_norm))))
                            xg = float(max(0.02, min(0.75, xg)))
                            
                            shots.append({
                                "type": "shot",
                                "t": t_star,
                                "pid": pid,
                                "xg": xg,
                                "is_goal": False # TODO: Goal detection logic
                            })
                            
                            xg_player[pid]["xg_total"] += xg
            start = t
            
    return shots, xg_player
