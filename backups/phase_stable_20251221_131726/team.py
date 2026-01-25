import cv2
import numpy as np
import math
from collections import defaultdict, Counter
import yaml

# Load Config
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()
CLASS = CONFIG["classes"]

def _hue_to_basic_color(h):
    if h is None: return "unknown"
    if h < 10 or h >= 170: return "red"
    elif h < 25: return "orange"
    elif h < 35: return "gold"
    elif h < 55: return "yellow"
    elif h < 85: return "green"
    elif h < 105: return "cyan"
    elif h < 135: return "blue"
    elif h < 155: return "purple"
    else: return "magenta"

def _describe_color(h, s, v):
    if s < 0.25:
        if v > 0.75: return "white"
        elif v < 0.35: return "black"
        else: return "gray"
    base = _hue_to_basic_color(h)
    if v > 0.80: return f"light_{base}"
    elif v < 0.35: return f"dark_{base}"
    else: return base

def _center_crop(img, fraction=0.5):
    if img is None or img.size == 0: return img
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    dy, dx = int(h * fraction / 2), int(w * fraction / 2)
    return img[cy-dy:cy+dy, cx-dx:cx+dx]

def assign_teams_to_ids_from_frames(frames, sample_frames=1000, class_players=(CLASS["player"],)):
    id_colors = {} 
    total_frames = len(frames)
    if total_frames == 0: return {}, {}
    
    step = max(1, total_frames // sample_frames)
    
    for i in range(0, total_frames, step):
        frame_data = frames[i]
        
        # Use crops if available
        if "crops" in frame_data and frame_data["crops"]:
            for crop_data in frame_data["crops"]:
                box_idx = crop_data["box_idx"]
                if box_idx < len(frame_data["boxes"]):
                    b = frame_data["boxes"][box_idx]
                    pid = b["id"]
                    cls = b["cls"]
                    
                    if pid is None: continue
                    pid = int(pid)
                    if pid < 0: continue
                    if int(cls) not in class_players: continue
                    
                    crop = crop_data["img"]
                    if crop is None or crop.size == 0: continue
                    
                    color_crop = _center_crop(crop, fraction=0.4)
                    hsv = cv2.cvtColor(color_crop, cv2.COLOR_BGR2HSV)
                    h = np.median(hsv[:,:,0])
                    s = np.median(hsv[:,:,1]) / 255.0
                    v = np.median(hsv[:,:,2]) / 255.0
                    
                    if pid not in id_colors: id_colors[pid] = []
                    id_colors[pid].append([h, s, v])

    hues_by_id = defaultdict(list)
    sv_by_id = defaultdict(list)
    for pid, colors in id_colors.items():
        for h, s, v in colors:
            hues_by_id[pid].append(h)
            sv_by_id[pid].append((s, v))

    if len(hues_by_id) < 2:
        print("[teams] Not enough player hues.")
        return {}, {}
        
    id_list = sorted(hues_by_id.keys())
    per_id_features = []
    per_id_hsv = {}
    
    for pid in id_list:
        h_med = float(np.median(hues_by_id[pid]))
        s_med = float(np.median([sv[0] for sv in sv_by_id[pid]])) if sv_by_id[pid] else 0.5
        v_med = float(np.median([sv[1] for sv in sv_by_id[pid]])) if sv_by_id[pid] else 0.5
        per_id_hsv[pid] = (h_med, s_med, v_med)
        h_rad = h_med * (np.pi / 90.0)
        per_id_features.append([math.cos(h_rad), math.sin(h_rad), s_med, v_med])
        
    per_id_features = np.array(per_id_features, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(per_id_features, 2, None, criteria, 10, flags)
    
    assign = {pid: int(labels[i][0]) for i, pid in enumerate(id_list)}
    counts = Counter(assign.values())
    if counts[0] < counts[1]:
        assign = {pid: (0 if lab == 1 else 1) for pid, lab in assign.items()}
        
    cluster_hsv = {0: [], 1: []}
    for pid, team_id in assign.items():
        cluster_hsv[team_id].append(per_id_hsv[pid])
        
    team_labels = {}
    for team_id in (0, 1):
        if not cluster_hsv[team_id]:
            team_labels[team_id] = f"team_{team_id}"
            continue
        median_h = np.median([c[0] for c in cluster_hsv[team_id]])
        median_s = np.median([c[1] for c in cluster_hsv[team_id]])
        median_v = np.median([c[2] for c in cluster_hsv[team_id]])
        p10_s = np.percentile([c[1] for c in cluster_hsv[team_id]], 10)
        
        if p10_s < 0.25 and median_v > 0.70:
            label = "white"
        else:
            label = _describe_color(median_h, median_s, median_v)
        team_labels[team_id] = label
        
    print(f"[teams] assignment: {len(assign)} ids; labels={team_labels}")
    return assign, team_labels

def resolve_team_conflicts(identity_map, team_map):
    """
    Task 2.2: Resolve conflicts.
    If two players on Team A are identified as "#10", the one with higher confidence keeps it.
    """
    # Group by (team, number)
    groups = defaultdict(list)
    for pid, info in identity_map.items():
        tid = team_map.get(pid)
        if tid is None: continue
        
        num = info["number"]
        if num is None: continue
        
        groups[(tid, num)].append((pid, info["conf"]))
        
    # Resolve
    final_map = identity_map.copy()
    for key, candidates in groups.items():
        if len(candidates) > 1:
            # Sort by confidence desc
            candidates.sort(key=lambda x: x[1], reverse=True)
            winner_pid = candidates[0][0]
            
            # Losers get number=None (or maybe keep it but mark as conflict? User said "higher confidence keeps it")
            for pid, conf in candidates[1:]:
                # We modify the entry in final_map
                # But wait, final_map[pid] is a dict, we should copy it or modify in place?
                # It's a dict of dicts.
                final_map[pid]["number"] = None # Revoke number
                print(f"[team] Conflict resolved: Team {key[0]} #{key[1]} -> PID {winner_pid} (conf {candidates[0][1]}), PID {pid} revoked.")
                
    return final_map
