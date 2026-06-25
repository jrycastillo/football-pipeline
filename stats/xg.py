import math
import numpy as np
from collections import defaultdict
from .geometry import get_distance, bbox_center

CLASS_BALL = 0
CLASS_GK = 1
CLASS_PLAYER = 2

def _sigmoid(x):
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def _goal_mouth_centers(w, h):
    return (np.array([0.12 * w, 0.50 * h]), np.array([0.88 * w, 0.50 * h]))

def _angle_to_goal(pt, w, h):
    lp, rp = _goal_mouth_centers(w, h)
    v1 = lp - np.array(pt)
    v2 = rp - np.array(pt)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 1.0
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return math.acos(cosang) / math.pi

def _distance_to_nearest_goal(pt, w, h):
    gl, gr = _goal_mouth_centers(w, h)
    return min(np.linalg.norm(np.array(pt) - gl), np.linalg.norm(np.array(pt) - gr))

def _estimate_xg(dist_px, angle_norm, header=False, under_pressure=False, w=1920):
    dist_norm = max(0.0, min(1.0, dist_px / max(1.0, 0.6 * w)))
    base = _sigmoid(-3.2 + 4.8 * (1.0 - dist_norm) + 1.8 * angle_norm)
    if header: base *= 0.80
    if under_pressure: base *= 0.75
    return float(max(0.02, min(0.75, base)))

def detect_shots_and_xg(frames, ownership, team_map, fps=25, speed_px_thr_frac=0.015, near_goal_frac=0.50, opp_thr_px=90):
    if not frames: return [], {}, {}
    h, w = frames[0]["orig_shape"]
    ball = [None] * len(frames)
    
    for t, f in enumerate(frames):
        balls = [b for b in f["boxes"] if b["cls"] == CLASS_BALL]
        ball[t] = bbox_center(balls[0]["xyxy"]) if balls else None
        
    v = [0.0] * len(frames)
    for t in range(1, len(frames)):
        if ball[t] and ball[t-1]:
            v[t] = math.hypot(ball[t][0] - ball[t-1][0], ball[t][1] - ball[t-1][1])
            
    speed_thr = speed_px_thr_frac * w
    shots = []
    
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
                    if ball[t_star] is not None and v[t_star] >= speed_thr:
                        dist = _distance_to_nearest_goal(ball[t_star], w, h)
                        if dist <= near_goal_frac * w:
                            ang = _angle_to_goal(ball[t_star], w, h)
                            
                            # Pressure check
                            under_pressure = False
                            my_team = team_map.get(pid)
                            me_c = None
                            for b in frames[t_star]["boxes"]:
                                if b["id"] == pid: me_c = bbox_center(b["xyxy"])
                            
                            if me_c:
                                for b in frames[t_star]["boxes"]:
                                    if b["cls"] in (CLASS_PLAYER, CLASS_GK) and b["id"] != pid:
                                        if team_map.get(b["id"]) != my_team:
                                            if get_distance(me_c, bbox_center(b["xyxy"])) <= opp_thr_px:
                                                under_pressure = True
                                                break
                            
                            # Header check (heuristic)
                            is_header = False
                            # ... (simplified header check if needed, or assume foot)
                            
                            xg = _estimate_xg(dist, ang, header=is_header, under_pressure=under_pressure, w=w)
                            
                            shots.append({
                                "type": "shot",
                                "t": t_star,
                                "pid": pid,
                                "xg": xg,
                                "team": my_team
                            })
            start = t
            
    return shots
