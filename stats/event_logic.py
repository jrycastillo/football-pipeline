import math
import numpy as np
import yaml
from collections import defaultdict
from vision.camera import Camera

# Load Config
try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    HEURISTICS = CONFIG.get("heuristics", {})
    FPS = HEURISTICS.get("FPS", 25)
    VID_STRIDE = HEURISTICS.get("VID_STRIDE", 1)
except:
    HEURISTICS = {}
    FPS = 25
    VID_STRIDE = 1

# Effective FPS accounts for frame skipping: with VID_STRIDE=3 at 25fps,
# each entry in all_frames/ball_track is 3/25=0.12s apart, not 1/25=0.04s
EFF_FPS = FPS / VID_STRIDE

# Constants (Meters) - Phase 190/195: Relaxed thresholds
DIST_TOUCH = 5.0  # Increased for better possession detection
DIST_DRIBBLE_OPP = 3.0  # Phase 195: Increased from 2.0 to 3.0m for more dribble detection
DIST_PASS_MIN = 3.0  # R18.2: raised 1.0 -> 3.0m. Sub-3m ownership flips are mapping noise in crowded areas, not real passes (engine fired ~49 transitions/min at 1m).
TIME_DRIBBLE_RETAIN = 1.5  # Phase 195: Reduced from 2.0s to 1.5s for quicker dribble success
SHOT_SPEED_THRESHOLD = 16.0  # Round 19: 16 m/s (58 km/h) — filters passes/clearances, keeps real shots

def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

class AdvancedEventDetector:
    def __init__(self, frame_width=None, frame_height=None):
        self.camera = Camera(frame_width=frame_width, frame_height=frame_height)
        
    def calculate_ownership(self, player_tracks, ball_track):
        """
        Map possession: Who had the ball when? (Using Meters)
        """
        ownership = [None] * len(ball_track)
        
        for t, ball_pos in enumerate(ball_track):
            if ball_pos is None: continue
            if t >= len(player_tracks): break
            
            frame_data = player_tracks[t]
            boxes = frame_data.get("boxes", [])
            
            best_pid = None
            min_dist_m = 100.0
            
            for b in boxes:
                if b.get("id") is None: continue
                # if b.get("cls") not in [1, 2]: continue # Player/GK
                
                c = bbox_center(b["xyxy"])
                dist_m = self.camera.calculate_distance(c, ball_pos)
                
                if dist_m < min_dist_m:
                    min_dist_m = dist_m
                    best_pid = b["id"]
            
            if best_pid is not None and min_dist_m <= DIST_TOUCH:
                ownership[t] = best_pid
                
        # Smooth
        return self._smooth_ownership(ownership)

    def _smooth_ownership(self, ownership):
        smoothed = list(ownership)
        last = None
        gap = 0
        for i in range(len(smoothed)):
            if smoothed[i] is not None:
                last = smoothed[i]
                gap = 0
            elif last is not None and gap < int(2.5 * EFF_FPS):  # 2.5s gap fill — ball undetected during passes/flights (industry: 2-3s)
                smoothed[i] = last
                gap += 1
        return smoothed

    def analyze(self, ownership, player_tracks, ball_track, team_map=None, raw_ball_frames=None):
        """
        Detects: Dribbles, Passes, Crosses, Tackles, Interceptions, Goals, XG.
        team_map: dict {pid (str): "TeamName"}
        raw_ball_frames: set of frame indices with actual ball detections (not interpolated)
        """
        if raw_ball_frames is None:
            raw_ball_frames = set()  # Fallback: treat all as raw (legacy behavior)
        stats = defaultdict(lambda: defaultdict(int))
        events = []
        
        # xG & Shot Logic Constants (Meters)
        GOAL_X = 105.0 # Goal Line
        GOAL_CENTER_Y = 34.0
        GOAL_WIDTH_HALF = 3.66 # 7.32 / 2
        
        if not ball_track: return [], stats

        # --- DISTANCE CALCULATION ---
        # Iterate per player track across time
        # player_tracks is list of frames -> list of boxes
        
        # We need to map: PID -> List of positions (x,y)
        # Or accumulate on the fly.
        
        prev_pos = {} # pid -> (x_m, y_m)

        # Round 11 fix: Speed limit for distance calculation
        # Max human sprint ~12 m/s; with VID_STRIDE=3 at 25fps each
        # frame interval is 0.12s, so max movement = 12*0.12 = 1.44m
        max_dist_per_frame = 12.0 * VID_STRIDE / FPS

        for t, frame_data in enumerate(player_tracks):
            boxes = frame_data.get("boxes", [])
            # Round 11 fix: Deduplicate PIDs within a frame to prevent prev_pos oscillation.
            # When finalize_bindings maps multiple ByteTrack fragments to the same jersey,
            # overlapping tracks produce 2+ boxes with the same PID in one frame.
            # Processing both causes prev_pos to ping-pong between physical locations,
            # so every frame-to-frame delta exceeds the speed limit → near-zero distance.
            # Fix: only process the FIRST (largest) box per PID per frame.
            seen_pids = set()
            for b in boxes:
                pid = b.get("id")
                if pid is None: continue
                if pid in seen_pids:
                    continue  # Skip duplicate PID in same frame
                seen_pids.add(pid)

                c = bbox_center(b["xyxy"])
                world_c = self.camera.project_point(c[0], c[1]) # Returns (x_m, y_m)

                if pid in prev_pos:
                    px, py = prev_pos[pid]
                    dx = world_c[0] - px
                    dy = world_c[1] - py
                    dist = math.hypot(dx, dy)

                    if dist < max_dist_per_frame:
                        stats[pid]["distance_m"] += dist

                # Phase 83: Track Class ID (Ball=0, GK=1, Player=2, Ref=3)
                cls = b.get("cls", 2)
                if "class_counts" not in stats[pid]:
                     stats[pid]["class_counts"] = defaultdict(int)
                stats[pid]["class_counts"][cls] += 1

                prev_pos[pid] = world_c

        # Helper: Calculate xG — P1 fix: logistic regression instead of exponential
        # Old formula (0.75 * exp(-0.15*dist) * angle*1.5) was inflated at close range
        # (always 0.99 under 3m). Logistic model calibrated to approximate StatsBomb values:
        #   5m → ~0.39, 11m → ~0.14, 20m → ~0.04, 30m → ~0.01
        def calculate_xg(start_pos, header=False, under_pressure=False, goal_x=None):
            if goal_x is None:
                dist_to_right = abs(start_pos[0] - GOAL_X)
                dist_to_left = abs(start_pos[0] - 0.0)
                goal_x = GOAL_X if dist_to_right < dist_to_left else 0.0

            # X = distance from goal line, C = lateral offset from center
            X = abs(start_pos[0] - goal_x)
            C = abs(start_pos[1] - GOAL_CENTER_Y)
            if X < 0.5: X = 0.5
            angle_rad = math.atan2(7.32 * X, X * X + C * C - GOAL_WIDTH_HALF ** 2)
            if angle_rad < 0: angle_rad += math.pi

            # Soccermatics open-source model (Wyscout data, 105x68m pitch).
            # Penalty (X=11,C=0)->0.77, 6yd box (X=5.5,C=0)->0.58, box edge (X=16.5,C=0)->0.12
            log_odds = (0.5103
                        + 0.6338 * angle_rad
                        - 0.2798 * math.hypot(X, C)
                        + 0.1243 * X
                        - 0.0300 * C
                        + 0.0014 * X * X
                        + 0.0041 * C * C
                        - 0.1251 * angle_rad * X)
            xg = 1.0 / (1.0 + math.exp(-log_odds))
            if header: xg *= 0.6
            if under_pressure: xg *= 0.75
            return min(0.95, max(0.01, xg))

        # 0. Spatial Residency Check (for GK ID)
        # Iterate all player tracks to count frames in box
        for t, frame_data in enumerate(player_tracks):
             for box in frame_data["boxes"]:
                 pid = box["id"]
                 if pid is not None:
                      # Project to meters
                      c = ((box["xyxy"][0] + box["xyxy"][2])/2, (box["xyxy"][1] + box["xyxy"][3])/2)
                      m = self.camera.project_point(c[0], c[1])
                      
                      # Check Box 1 (Left) & 2 (Right)
                      # Box: X < 16.5 or X > 105-16.5
                      # Y in [13.84, 54.16]
                      in_box = False
                      if (m[0] < 16.5 or m[0] > (105.0 - 16.5)) and (13.84 < m[1] < 54.16):
                          in_box = True
                          
                      if in_box:
                          stats[pid]["frames_in_box"] += 1
                          
                      # Accumulate for Centroid (Phase 82)
                      stats[pid]["sum_x"] += m[0]
                      stats[pid]["sum_y"] += m[1]
                      stats[pid]["pos_count"] += 1
                      
        # Calculate Averages & Dominant Class
        for pid, s in stats.items():
            if s["pos_count"] > 0:
                s["avg_x"] = s["sum_x"] / s["pos_count"]
                s["avg_y"] = s["sum_y"] / s["pos_count"]
            
            # Phase 83: Determine Dominant Class
            if "class_counts" in s:
                 # Get class with max counts
                 dom_cls = max(s["class_counts"].items(), key=lambda x: x[1])[0]
                 s["dominant_class"] = dom_cls
            else:
                 s["dominant_class"] = 2 # Default Player

        # Round 16: Infer team attack directions for goal attribution validation
        # Team with lower avg_x attacks right goal (105m), higher avg_x attacks left (0m)
        self._team_attack_direction = {}  # team_name -> goal_x
        if team_map:
            team_sum_x = defaultdict(float)
            team_count = defaultdict(int)
            for pid, s in stats.items():
                if s.get("pos_count", 0) > 10:
                    team_name = team_map.get(str(pid))
                    if team_name and team_name != "Unknown":
                        team_sum_x[team_name] += s.get("avg_x", 52.5)
                        team_count[team_name] += 1
            team_avg_x = {}
            for tn in team_sum_x:
                if team_count[tn] > 0:
                    team_avg_x[tn] = team_sum_x[tn] / team_count[tn]
            if len(team_avg_x) >= 2:
                sorted_teams = sorted(team_avg_x.items(), key=lambda x: x[1])
                self._team_attack_direction[sorted_teams[0][0]] = 105.0
                self._team_attack_direction[sorted_teams[1][0]] = 0.0
                print(f"[GoalDir] {sorted_teams[0][0]} attacks RIGHT (avg_x={sorted_teams[0][1]:.1f}), "
                      f"{sorted_teams[1][0]} attacks LEFT (avg_x={sorted_teams[1][1]:.1f})")

        # 1. Possession & Dribbling
        # P0 fix: Track tackle frames from section 1 to prevent double-counting in section 3
        # P2 fix: Require ball carrier movement for dribble detection
        dribble_debug_count = 0
        _tackle_frames_s1 = set()  # frames where tackles were credited in this section
        _last_tackle_s1 = {}  # pid -> last frame, cooldown for section 1 tackles
        last_dribble_frame = {}  # pid -> last frame counted as dribble
        dribble_lookback = max(1, int(EFF_FPS * 0.5))  # 0.5 second lookback for movement check
        for t, pid in enumerate(ownership):
            if pid is not None:
                stats[pid]["touch_frames"] += 1

                # Check Dribble (Opponent within range)
                opp_id = self._is_opponent_near(t, pid, player_tracks, dist_m=DIST_DRIBBLE_OPP)
                if opp_id is not None:
                    # P2 fix: Require ball carrier to have moved (standing still ≠ dribble)
                    moved = False
                    if t >= dribble_lookback and ball_track[t] and ball_track[t - dribble_lookback]:
                        move_dist = self.camera.calculate_distance(
                            ball_track[t], ball_track[t - dribble_lookback])
                        if move_dist > 1.5:  # R18.1: raised 1.0 -> 1.5m in 0.5s
                            moved = True
                    elif t < dribble_lookback:
                        moved = True  # Not enough history, benefit of doubt

                    if not moved:
                        continue  # Standing still with opponent near → not a dribble

                    # R18.1: Dribble cooldown raised 1s -> 3s — a take-on is a discrete
                    # event, not something that recurs every second of a carry
                    last_frame = last_dribble_frame.get(pid, -999)
                    if t - last_frame > EFF_FPS * 3:
                        last_dribble_frame[pid] = t
                        stats[pid]["dribbles"] += 1
                        dribble_debug_count += 1

                        # Credit Challenge to Opponent (Phase 85)
                        stats[opp_id]["challenges_total"] += 1

                        # Check Success (Retain for 1.5s)
                        if self._retains_possession(t, pid, ownership, duration_s=TIME_DRIBBLE_RETAIN):
                             stats[pid]["dribbles_successful"] += 1
                        else:
                            # Dribble Failed -> Challenge Won by Opponent
                            stats[opp_id]["challenges_won_total"] += 1
                            last_s1 = _last_tackle_s1.get(opp_id, -999)
                            if t - last_s1 > int(EFF_FPS * 60):
                                stats[opp_id]["tackles"] += 1
                                stats[opp_id]["tackles_successful"] += 1
                                _last_tackle_s1[opp_id] = t
                                _tackle_frames_s1.add(t)  # P0: track for dedup

        # 2. Passing (Change of Ownership)
        # Segment ownership
        segments = []
        if ownership:
            curr = ownership[0]
            start = 0
            for i, pid in enumerate(ownership):
                if pid != curr:
                    segments.append({"pid": curr, "start": start, "end": i-1})
                    curr = pid
                    start = i
            segments.append({"pid": curr, "start": start, "end": len(ownership)-1})

        # FIX: Filter out None segments so passes through gaps (A → None → B) are counted
        # Previously: A → None → B was skipped because both A→None and None→B had a None endpoint
        # Now: We compare non-None segments directly with a max gap check
        non_none_segments = [s for s in segments if s["pid"] is not None]
        _pass_debug = {"transitions": 0, "gap_filtered": 0, "no_ball": 0, "too_short": 0, "tackle_filtered": 0, "counted": 0}
        # Round 2 fix: Interception debounce — max 1 per player per 3-second window
        _last_interception_frame = {}  # pid -> last frame an interception was credited

        # Minimum ownership duration before a possession counts as a pass origin.
        # Industry (Tryolabs): 0.4s mode window at 25fps. At stride=3, EFF_FPS~8.3,
        # so 3 frames ≈ 0.36s — prevents single-frame noise blips from registering as passes.
        MIN_OWN_FRAMES = max(3, int(EFF_FPS * 0.36))  # ~3 frames at stride=3

        for i in range(len(non_none_segments) - 1):
            seg_a = non_none_segments[i]
            seg_b = non_none_segments[i+1]

            p_a = seg_a["pid"]
            p_b = seg_b["pid"]

            if p_a == p_b: continue

            # Round 14: Skip if passer held ball for less than 0.3s (noise)
            seg_a_duration = seg_a["end"] - seg_a["start"]
            if seg_a_duration < MIN_OWN_FRAMES:
                continue

            _pass_debug["transitions"] += 1

            # Max gap check: Don't count as pass if gap > 3 seconds (ball out of play)
            gap_frames = seg_b["start"] - seg_a["end"]
            if gap_frames > EFF_FPS * 3:
                _pass_debug["gap_filtered"] += 1
                continue

            # P0 fix: Check if this is a tackle/dispossession, not a pass
            # If p_a and p_b were within close proximity at the transition,
            # the ball was physically won, not deliberately passed
            # Threshold: 2.0m (arm's length) — 3.0m was too aggressive, filtered short passes
            transition_frame = seg_a["end"]
            if transition_frame < len(player_tracks):
                p_a_box = None
                p_b_box = None
                for b in player_tracks[transition_frame].get("boxes", []):
                    if b.get("id") == p_a: p_a_box = b
                    if b.get("id") == p_b: p_b_box = b
                if p_a_box and p_b_box:
                    p_a_c = bbox_center(p_a_box["xyxy"])
                    p_b_c = bbox_center(p_b_box["xyxy"])
                    prox = self.camera.calculate_distance(p_a_c, p_b_c)
                    if prox < 0.8:  # Only filter body-contact range (<0.8m); 2m was over-filtering short passes
                        _pass_debug["tackle_filtered"] += 1
                        continue

            # Verify distance
            # FIX: Search nearby frames if ball position is None at exact endpoint
            # Ball is often undetected during passes (in flight) so we search backwards/forwards
            start_pos = ball_track[seg_a["end"]]
            end_pos = ball_track[seg_b["start"]]

            if not start_pos:
                for offset in range(1, 8):
                    idx = seg_a["end"] - offset
                    if idx >= seg_a["start"] and idx >= 0 and ball_track[idx]:
                        start_pos = ball_track[idx]
                        break

            if not end_pos:
                for offset in range(1, 8):
                    idx = seg_b["start"] + offset
                    if idx <= seg_b["end"] and idx < len(ball_track) and ball_track[idx]:
                        end_pos = ball_track[idx]
                        break

            if not (start_pos and end_pos):
                _pass_debug["no_ball"] += 1
                continue

            if start_pos and end_pos:
                dist = self.camera.calculate_distance(start_pos, end_pos)
                if dist <= DIST_PASS_MIN:
                    _pass_debug["too_short"] += 1
                if dist > DIST_PASS_MIN:
                    _pass_debug["counted"] += 1
                    stats[p_a]["passes_total"] += 1
                    
                    # RELAXED SUCCESS CHECK
                    # If p_b is SAME TEAM as p_a -> Complete
                    # If p_b is OPPONENT -> Interception/Incomplete
                    
                    is_complete = True # Default legacy
                    
                    if team_map:
                        team_a = team_map.get(str(p_a))
                        team_b = team_map.get(str(p_b))
                        
                        if team_a and team_b and team_a != "Unknown" and team_b != "Unknown":
                            if team_a == team_b:
                                is_complete = True
                            else:
                                is_complete = False # Interception by Opponent
                        else:
                            # If unknown team, fallback to "distance" or "tackle"? 
                            # Assume complete to avoid 0 stats if clustering fails.
                            is_complete = True
                    
                    if is_complete:
                        stats[p_a]["passes_complete"] += 1
                        events.append({"type": "pass", "from": p_a, "to": p_b, "frame": seg_a["end"]})

                    # 3. Check Crosses (Side Channel to Box)
                    # Side Channel: |y - 34| > 25 -> y < 9 or y > 59
                    # Box: x > 88.5 or x < 16.5, and |y - 34| < 20.15
                    start_m = self.camera.project_point(start_pos[0], start_pos[1])
                    end_m = self.camera.project_point(end_pos[0], end_pos[1])

                    in_side_channel = abs(start_m[1] - 34.0) > 25.0
                    into_box_right = end_m[0] > 88.5 and abs(end_m[1] - 34.0) < 20.15
                    into_box_left = end_m[0] < 16.5 and abs(end_m[1] - 34.0) < 20.15

                    if in_side_channel and (into_box_right or into_box_left):
                        stats[p_a]["crosses_total"] += 1
                        if is_complete:
                            stats[p_a]["crosses_complete"] += 1
                        events.append({"type": "cross", "from": p_a, "to": p_b, "frame": seg_a["end"]})

                    # --- ADVANCED STATS (Phase 86) ---
                    # 1. Packing (Opponents bypassed)
                    # packing_value = self._calculate_packing(seg_a["end"], seg_b["start"], team_a, tracks, ball_tracks)
                    # NOTE: _calculate_packing needs tracks. Passing placeholder for now or lightweight implementation.
                    # Simplified: Just measure X-gain? No, user wants packing.
                    # Implementation detail: iterate all tracks at frame 'start', count opps in X-range.
                    pass_dist = self.camera.calculate_distance(start_pos, end_pos)
                    
                    # 2. Pass Range Classification
                    if pass_dist <= 16:
                        stats[p_a]["short_passes"] += 1
                        if is_complete: stats[p_a]["short_passes_accurate"] += 1
                    elif pass_dist <= 30:
                        stats[p_a]["medium_passes"] += 1
                        if is_complete: stats[p_a]["medium_passes_accurate"] += 1
                    else:
                        stats[p_a]["long_passes"] += 1
                        if is_complete: stats[p_a]["long_passes_accurate"] += 1 # "accurate_long_passes"
                        
                    # 3. Expected Assist (xA)
                    # Will be populated if this pass leads to a shot (next event)
                    # Stored in `last_pass_info` for Shot logic to consume.
                    self.last_pass_info = {"player": p_a, "to": p_b, "frame": seg_b["start"], "xg_assigned": False}

                    # --- INTERCEPTION LOGIC & ADVANCED DEFENSE ---
                    if not is_complete:
                        if team_map and team_a and team_b and team_a != team_b:
                            int_frame = seg_b["start"]
                            last_int = _last_interception_frame.get(p_b, -999)
                            # Industry (Opta/StatsBomb): interception requires ball was in flight
                            # (directed toward someone else, then cut off). R18.1: minimum travel
                            # raised 1.5m -> 3.0m — short transitions in crowded areas are ownership
                            # mapping noise, not cut-out passes.
                            ball_was_in_flight = dist > 3.0
                            # R18.1: debounce 15s -> 30s per player
                            if int_frame - last_int > int(EFF_FPS * 30) and ball_was_in_flight:
                                _last_interception_frame[p_b] = int_frame
                                stats[p_b]["interceptions"] += 1
                                stats[p_b]["ball_interceptions_total"] += 1

                                # Ball Recovery in Opp Half
                                # Round 3 fix: Project to meters before comparing to 52.5m
                                # Previously compared pixel X (0-1920) to 52.5 meters,
                                # causing ALL recoveries to classify as "opp half"
                                end_m = self.camera.project_point(end_pos[0], end_pos[1])
                                if end_m[0] > 52.5:
                                     stats[p_b]["ball_recoveries_opp_half"] += 1
                                elif end_m[0] < 52.5:
                                     stats[p_b]["ball_recoveries_own_half"] += 1

                                events.append({"type": "interception", "by": p_b, "frame": int_frame})



                    # 4. Check Shots / Key Passes (End in Box)
                    # If end_pos is in box and NO next possession or Goal?
                    # "Shot" logic usually implies high velocity towards goal.
                    # Simplified: If Pass ends in Box and is NOT complete (or complete to shooter?), maybe shot?
                    # Better: Analzye Ball Trajectory for Shots (Speed > 15m/s towards goal)
                    
        print(f"[PassDebug] Ownership transitions: {_pass_debug['transitions']}, "
              f"gap_filtered: {_pass_debug['gap_filtered']}, tackle_filtered: {_pass_debug['tackle_filtered']}, "
              f"no_ball: {_pass_debug['no_ball']}, too_short: {_pass_debug['too_short']}, "
              f"counted: {_pass_debug['counted']}")

        # 5. Shot Detection (Trajectory Analysis)
        # Round 2 fix: Only compute velocity on raw ball detections (not interpolated)
        # Interpolated positions create false velocity spikes at gap boundaries
        frames_count = len(ball_track)
        _shot_debug = {"raw_pairs": 0, "interp_skipped": 0}
        for i in range(2, frames_count):
            if ball_track[i] and ball_track[i-2]:
                # Round 2 fix: Skip if either frame is interpolated
                if raw_ball_frames and (i not in raw_ball_frames or (i-2) not in raw_ball_frames):
                    _shot_debug["interp_skipped"] += 1
                    continue
                _shot_debug["raw_pairs"] += 1
                p1 = ball_track[i-2]
                p2 = ball_track[i]
                
                # Distance in meters (using Camera.calculate_distance logic, but we have world coords if projected)
                # Wait, ball_track is currently PIXELS.
                # Project to Meters!
                # We need self.camera
                
                m1 = self.camera.project_point(p1[0], p1[1])
                m2 = self.camera.project_point(p2[0], p2[1])
                
                dist_m = math.hypot(m2[0]-m1[0], m2[1]-m1[1])
                speed_mps = dist_m / (2.0 / EFF_FPS)  # 2 processed frames apart

                # Shot Threshold: Phase 195 lowered to 8 m/s
                # Fix: Detect shots toward BOTH goals (not just right)
                moving_right = m2[0] > m1[0]
                moving_left = m2[0] < m1[0]

                if speed_mps > SHOT_SPEED_THRESHOLD and (moving_right or moving_left):
                    # Determine target goal based on direction
                    if moving_right:
                        goal_x = 105.0  # Right goal
                        goal_center_y = 34.0
                    else:
                        goal_x = 0.0    # Left goal
                        goal_center_y = 34.0

                    # R19: Ball must be in attacking third (within 35m of target goal)
                    dist_to_goal = abs(m2[0] - goal_x)
                    if dist_to_goal > 35.0:
                        continue  # Ball too far from goal to be a real shot

                    # Check if inside goal coordinates
                    # Simple linear projection
                    if abs(m2[0] - m1[0]) > 0.1:
                        slope = (m2[1] - m1[1]) / (m2[0] - m1[0])
                        y_at_goal = m2[1] + slope * (goal_x - m2[0])

                        if 30.34 < y_at_goal < 37.66:
                            # Potential Shot on Target
                            # Attribute to last possessor
                            # Find who had ball last
                            shooter = ownership[i] if i < len(ownership) else None
                            if not shooter and i >= 5:
                                shooter = ownership[i-5]  # Look back
                            # FIX: Skip if shooter is a GK (cls_id=1) - goal kicks/punts shouldn't count as shots
                            if shooter and stats.get(shooter, {}).get("dominant_class") == 1:
                                shooter = None
                            if shooter:
                                # Round 19: Shot debounce 5s (was 2s) to reduce over-count
                                shot_debounce = max(10, int(EFF_FPS * 5))
                                recent = [e for e in events if e["type"] == "shot" and abs(e["frame"] - i) < shot_debounce]
                                if not recent:
                                    # Check if under pressure
                                    under_pressure = self._is_opponent_near(i, shooter, player_tracks, dist_m=3.0) is not None
                                    xg = calculate_xg(m1, goal_x=goal_x, under_pressure=under_pressure)
                                    stats[shooter]["shots_on_target"] += 1
                                    # Categorize xG by pressure
                                    if under_pressure:
                                        stats[shooter]["xg_foot_opponent_present"] += xg
                                    else:
                                        stats[shooter]["xg_foot_no_opponent"] += xg
                                    # Shots categorization - use distance to target goal
                                    shot_dist = math.hypot(m2[0] - goal_x, m2[1] - goal_center_y)
                                    if shot_dist <= 5:
                                        stats[shooter]["close_range_shots"] += 1
                                    elif shot_dist <= 16:
                                        stats[shooter]["mid_range_shots"] += 1
                                    else:
                                        stats[shooter]["long_range_shots"] += 1

                                    # Assign xA to previous passer
                                    if hasattr(self, 'last_pass_info') and self.last_pass_info:
                                        # Check if pass was recent (within 5 seconds?)
                                        # Heuristic: If pass receiver == shooter
                                        if self.last_pass_info["to"] == shooter and not self.last_pass_info["xg_assigned"]:
                                            assister = self.last_pass_info["player"]
                                            stats[assister]["expected_assists"] += xg
                                            self.last_pass_info["xg_assigned"] = True

                                    events.append({
                                        "type": "shot",
                                        "player": shooter,
                                        "frame": i,
                                        "xg": round(xg, 2),
                                        "speed": round(speed_mps, 1),
                                        "direction": "right" if moving_right else "left"
                                    })

                                    # --- BLOCKED SHOT LOGIC ---
                                    # Check if any opponent is on the shot vector (Ball -> Goal)
                                    blocked = False
                                    for opp_id in player_tracks[i].get("ids", []):
                                         if opp_id == shooter: continue
                                         pass

                                    # --- GOAL DETECTION ---
                                    # Two methods: (1) trajectory extrapolation to goal line,
                                    # (2) ball reaching near goal zone and disappearing.
                                    # Linear homography can't reliably project the far goal line,
                                    # so we use the shot vector to predict where the ball crosses.
                                    goal_confirmed = False
                                    GOAL_Y_MIN = 30.34   # goal post Y in meters (34 - 7.32/2)
                                    GOAL_Y_MAX = 37.66   # goal post Y in meters (34 + 7.32/2)
                                    target_goal_x = goal_x  # 105.0 or 0.0 from shot direction

                                    # Method 1: Extrapolate shot trajectory to goal line.
                                    # Gate: shot must originate from within 30m of target goal
                                    # AND xG must be meaningful (>0.02) — eliminates long-range
                                    # shots that happen to project near the center of the goal.
                                    dist_shooter_to_goal = abs(m1[0] - target_goal_x)
                                    xg_for_gate = calculate_xg(m1, goal_x=target_goal_x, under_pressure=False)
                                    if (dist_shooter_to_goal <= 30.0 and xg_for_gate > 0.02
                                            and abs(m2[0] - m1[0]) > 0.05):
                                        slope = (m2[1] - m1[1]) / (m2[0] - m1[0])
                                        y_at_goal = m2[1] + slope * (target_goal_x - m2[0])
                                        if GOAL_Y_MIN <= y_at_goal <= GOAL_Y_MAX:
                                            # Ball is heading into the goal — confirm it stays
                                            # on course for at least 2 more frames (not deflected)
                                            lookahead = max(3, int(0.5 * EFF_FPS))
                                            on_course = 0
                                            for k in range(i+1, min(i+lookahead+1, len(ball_track))):
                                                if ball_track[k]:
                                                    mk2 = self.camera.project_point(ball_track[k][0], ball_track[k][1])
                                                    y_proj = mk2[1] + slope * (target_goal_x - mk2[0])
                                                    if GOAL_Y_MIN - 0.5 <= y_proj <= GOAL_Y_MAX + 0.5:
                                                        on_course += 1
                                            if on_course >= 2:
                                                goal_confirmed = True

                                    # Method 2: Ball enters goal zone (within 5m of goal line, Y on target)
                                    # and then disappears (no detection for 1+ second) — ball in net
                                    if not goal_confirmed:
                                        goal_lookahead = max(10, int(2.0 * EFF_FPS))
                                        last_ball_in_zone = None
                                        for k in range(i, min(i + goal_lookahead, len(ball_track))):
                                            if ball_track[k]:
                                                mk = self.camera.project_point(ball_track[k][0], ball_track[k][1])
                                                near_goal = (mk[0] > 98.0) if moving_right else (mk[0] < 7.0)
                                                if near_goal and (GOAL_Y_MIN - 1.5 <= mk[1] <= GOAL_Y_MAX + 1.5):
                                                    last_ball_in_zone = k
                                        if last_ball_in_zone is not None:
                                            gap_start = last_ball_in_zone + 1
                                            gap_end = min(last_ball_in_zone + int(EFF_FPS * 1.5), len(ball_track))
                                            missing = sum(1 for k in range(gap_start, gap_end) if not ball_track[k])
                                            if missing >= int(EFF_FPS * 0.8):
                                                goal_confirmed = True

                                    if goal_confirmed:
                                        # Round 16: Validate shooter's team attacks this goal
                                        correct_shooter = shooter
                                        if team_map and self._team_attack_direction:
                                            shooter_team = team_map.get(str(shooter))
                                            expected_goal = self._team_attack_direction.get(shooter_team)
                                            actual_goal = 105.0 if moving_right else 0.0
                                            if expected_goal is not None and expected_goal != actual_goal:
                                                # Mismatch: find nearest player from attacking team
                                                atk_teams = [t for t, g in self._team_attack_direction.items() if g == actual_goal]
                                                if atk_teams and i < len(player_tracks):
                                                    best_alt, best_d = None, 999
                                                    for b in player_tracks[i].get("boxes", []):
                                                        alt_pid = b.get("id")
                                                        if alt_pid is None: continue
                                                        if team_map.get(str(alt_pid)) == atk_teams[0]:
                                                            c = bbox_center(b["xyxy"])
                                                            if ball_track[i]:
                                                                d = self.camera.calculate_distance(c, ball_track[i])
                                                                if d < best_d:
                                                                    best_d = d
                                                                    best_alt = alt_pid
                                                    if best_alt and best_d < 15.0:
                                                        print(f"[Goal] Corrected: {shooter} ({shooter_team}) -> "
                                                              f"{best_alt} ({atk_teams[0]})")
                                                        correct_shooter = best_alt
                                        # R19: Goal debounce — 30s per-team cooldown
                                        goal_team = team_map.get(str(correct_shooter), "unknown") if team_map else "unknown"
                                        goal_debounce_frames = int(30.0 * EFF_FPS)
                                        recent_team_goal = any(
                                            e for e in events
                                            if e["type"] == "goal"
                                            and abs(e["frame"] - i) < goal_debounce_frames
                                            and team_map and team_map.get(str(e["player"])) == goal_team
                                        )
                                        if not recent_team_goal:
                                            stats[correct_shooter]["goals"] += 1
                                            stats[correct_shooter]["goals_total"] += 1
                                            events.append({"type": "goal", "player": correct_shooter, "frame": i, "assist": None})
                                            print(f"[Goal] Confirmed: Player #{correct_shooter} (team={goal_team}) at frame {i}")
                                        else:
                                            print(f"[Goal] Debounced: Player #{correct_shooter} (team={goal_team}) at frame {i} — duplicate within 30s")

                                    # print(f"SHOT! Player {shooter} | Speed {speed_mps:.1f} m/s | xG {xg:.2f}")

        print(f"[ShotDebug] Raw pairs evaluated: {_shot_debug['raw_pairs']}, "
              f"interpolated skipped: {_shot_debug['interp_skipped']}")

        # 6. GK Save Detection (Post-Hoc Analysis of Trajectories)
        # FIX: Use YOLO dominant_class (cls_id=1) instead of frames_in_box heuristic
        # This prevents field players (strikers/defenders in the box) from getting save stats
        possible_gks = []
        for pid, s in stats.items():
             if s.get("dominant_class") == 1:  # Only actual GKs (YOLO class 1)
                 possible_gks.append(pid)
        
        # Iterate high-speed ball segments again (same raw-only filter as shots)
        for i in range(2, frames_count - 5):
            if ball_track[i] and ball_track[i-2]:
                # Round 2 fix: Skip interpolated frames
                if raw_ball_frames and (i not in raw_ball_frames or (i-2) not in raw_ball_frames):
                    continue
                m1 = self.camera.project_point(ball_track[i-2][0], ball_track[i-2][1])
                m2 = self.camera.project_point(ball_track[i][0], ball_track[i][1])
                dist_m = math.hypot(m2[0]-m1[0], m2[1]-m1[1])
                speed_mps = dist_m / (2.0 / EFF_FPS)

                # If Shot Incoming (using same threshold as shot detection)
                if speed_mps > SHOT_SPEED_THRESHOLD and (m2[0] < 5.0 or m2[0] > 100.0): # Near Goal Ends
                     # Check next few frames for "Intervention"
                     # Intervention = Speed drop OR Direction change
                     # AND GK is close (<1m)
                     
                     for gk_id in possible_gks:
                         # Get GK pos at time i
                         # Iterate boxes in frame i
                         gk_pos = None
                         gk_box_y_min = 0
                         for b in player_tracks[i]["boxes"]:
                             if b["id"] == gk_id:
                                 gk_pos = self.camera.project_point(bbox_center(b["xyxy"])[0], bbox_center(b["xyxy"])[1])
                                 gk_box_y_min = b["xyxy"][1]
                                 break
                         
                         if gk_pos:
                             # Dist Ball to GK
                             d_gk = math.hypot(gk_pos[0]-m2[0], gk_pos[1]-m2[1])
                             if d_gk < 2.0: # 2 meter radius validation
                                 # Potential Interaction
                                 # Check what happens next (i+1 to i+3)
                                 # If speed drops < 5 m/s OR vector flips
                                 
                                 f_next = min(i+3, len(ball_track)-1)
                                 if ball_track[f_next]:
                                     m_next = self.camera.project_point(ball_track[f_next][0], ball_track[f_next][1])
                                     v_next_x = m_next[0] - m2[0]
                                     # v_prev_x = m2[0] - m1[0]
                                     
                                     # If direction flipped (Shot X+ -> Save X-)
                                     # or Speed Death
                                     dist_next = math.hypot(m_next[0]-m2[0], m_next[1]-m2[1])
                                     speed_next = dist_next / (3.0 / EFF_FPS)
                                     
                                     if speed_next < 5.0 or (np.sign(v_next_x) != np.sign(m2[0]-m1[0])):
                                          # SAVE DETECTED!
                                          # Use a debounce to avoid multi-counting same save (~1 second)
                                          save_debounce = max(20, int(EFF_FPS))
                                          recent_saves = [e for e in events if e["type"] == "save" and abs(e["frame"] - i) < save_debounce]
                                          if not recent_saves:
                                               stats[gk_id]["shots_saved_total"] += 1
                                               events.append({"type": "save", "player": gk_id, "frame": i})
                                               
                                               # Classify Range
                                               # Origin of shot? We need to trace back to last "kick"
                                               # Simple Lookback: 3 seconds?
                                               shot_origin = (52.5, 34.0) # Default mid
                                               # Look backwards for low speed or player touch
                                               for k in range(i, max(0, i-50), -1):
                                                   if ownership[k] is not None and ownership[k] != gk_id:
                                                        if ball_track[k]:
                                                             shot_origin = self.camera.project_point(ball_track[k][0], ball_track[k][1])
                                                        break
                                               
                                               shot_dist = math.hypot(shot_origin[0]-GOAL_X, shot_origin[1]-GOAL_CENTER_Y) 
                                               # (Actually dist to Goal Center)
                                               
                                               if shot_dist < 6.0: stats[gk_id]["close_range_saves"] += 1
                                               elif shot_dist < 17.0: stats[gk_id]["mid_range_saves"] += 1
                                               else: stats[gk_id]["long_range_saves"] += 1
                                               
                                               # Classify Type: Jumping
                                               # Check y_min change
                                               # Classify Type: Jumping (Phase 85)
                                               # Check GK Height/Aspect Ratio
                                               if gk_box_y_min > 0:
                                                    # If box top is unusually high (small Y) compared to standing?
                                                    # Easier: Check Aspect Ratio. Jumping = Stretched vertically?
                                                    # Or just "High Save" -> Shot height > 2m? 
                                                    # We don't have z-axis.
                                                    # Heuristic: If shot was "Long Range" (>17m), likely jumping/diving.
                                                    if shot_dist > 15.0:
                                                         stats[gk_id]["jumping_saves"] += 1

        # 3. Defensive (Tackles)
        # R18.1: Transition-based tackles DISABLED. Every cross-team ownership flip
        # fired this path; even with controlled-possession, proximity, and
        # ball-reaction gates it overcounted 3-8x (ownership mapping noise in
        # crowded areas). Tackles now come only from Section 1 (failed-dribble
        # challenges), which matches the Opta definition: dispossessing a player
        # who is in control of the ball.
        _ENABLE_TRANSITION_TACKLES = False
        _last_tackle_frame_s3 = {}  # pid -> last frame a tackle was credited in section 3
        for i in (range(len(segments) - 1) if _ENABLE_TRANSITION_TACKLES else range(0)):
             seg_a = segments[i]
             seg_b = segments[i+1]
             p_a = seg_a["pid"]
             p_b = seg_b["pid"]

             if p_a is not None and p_b is not None and p_a != p_b:
                 end_frame = seg_a["end"]
                 # P0 fix: Skip if already counted as tackle in section 1 (within 1s window)
                 already_counted = any(abs(end_frame - tf) < EFF_FPS for tf in _tackle_frames_s1)
                 if already_counted:
                     continue

                 # Industry (Opta): tackle requires CONTROLLED possession — passer must have
                 # held ball for at least 20 effective frames (~2.4s at stride=3) before being
                 # dispossessed. Filters loose ball recoveries and split-second deflections.
                 seg_a_dur = seg_a["end"] - seg_a["start"]
                 if seg_a_dur < 20:
                     continue

                 # Industry: tackle is always a cross-team event (defender dispossesses attacker)
                 if team_map:
                     team_a = team_map.get(str(p_a))
                     team_b = team_map.get(str(p_b))
                     if team_a and team_b and team_a != "Unknown" and team_b != "Unknown":
                         if team_a == team_b:
                             continue  # Same team — not a tackle

                 # Per-player cooldown: 45s — real tackles are rare (2-3 per player per 90 min)
                 last_tkl = _last_tackle_frame_s3.get(p_b, -999)
                 if end_frame - last_tkl < int(EFF_FPS * 45):
                     continue

                 # Require physical proximity (1.5m) AND ball must have been moving before
                 # the dispossession (spd_before > 1.0 m/s) — eliminates standing challenges.
                 # Direction change OR speed drop ≥50% counts as a valid tackle reaction.
                 ball_speed_drop = False
                 if ball_track and end_frame >= 2 and end_frame + 2 < len(ball_track):
                     b_before = ball_track[end_frame - 2]
                     b_at     = ball_track[end_frame]
                     b_after  = ball_track[end_frame + 2]
                     if b_before and b_at and b_after:
                         m_before = self.camera.project_point(b_before[0], b_before[1])
                         m_at     = self.camera.project_point(b_at[0], b_at[1])
                         m_after  = self.camera.project_point(b_after[0], b_after[1])
                         spd_before = math.hypot(m_at[0]-m_before[0], m_at[1]-m_before[1])
                         spd_after  = math.hypot(m_after[0]-m_at[0], m_after[1]-m_at[1])
                         if spd_before > 1.0 and (spd_after < spd_before * 0.5):
                             ball_speed_drop = True
                         if spd_before > 1.0:
                             dx1, dy1 = m_at[0]-m_before[0], m_at[1]-m_before[1]
                             dx2, dy2 = m_after[0]-m_at[0], m_after[1]-m_at[1]
                             if (dx1*dx2 + dy1*dy2) < 0:
                                 ball_speed_drop = True
                 if self._is_opponent_near(end_frame, p_a, player_tracks, dist_m=1.5) and ball_speed_drop:
                     _last_tackle_frame_s3[p_b] = end_frame
                     stats[p_b]["tackles"] += 1
                     stats[p_b]["tackles_successful"] += 1
                     events.append({"type": "tackle", "by": p_b, "on": p_a, "frame": end_frame})
                     
        # 4. Penalty Box Touch Tracking (xG REMOVED — Round 3 fix)
        # Round 3 fix: Section 4 was accumulating xG for every ownership segment ending
        # in the penalty box, regardless of whether a shot was taken. This caused:
        # - xG inflation: 9.77 total (vs realistic 2-4) at VID_STRIDE=3
        # - GK xG: Red #1 (GK) accumulated 1.50 xG from 0 shots (goal kicks/catches)
        # - Double-counting: Same xG counters used by Section 5 (actual shots)
        # xG now comes ONLY from Section 5 (velocity-verified shot detection).
        # Section 4 still tracks penalty box touches for "in_box_touches" stat.
        for seg in segments:
            pid = seg["pid"]
            if pid is None: continue

            end_f = seg["end"]
            ball_pos = ball_track[end_f]

            if ball_pos and self.camera.is_in_penalty_box(ball_pos):
                stats[pid]["in_box_touches"] += 1
                
        return events, stats

    def _is_opponent_near(self, frame_idx, pid, player_tracks, dist_m=2.0):
        if frame_idx >= len(player_tracks): return False
        boxes = player_tracks[frame_idx].get("boxes", [])
        
        my_box = next((b for b in boxes if b["id"] == pid), None)
        if not my_box: return None
        
        my_c = bbox_center(my_box["xyxy"])
        
        for b in boxes:
            if b["id"] == pid or b["id"] is None: continue
            
            c = bbox_center(b["xyxy"])
            d = self.camera.calculate_distance(my_c, c)
            if d < dist_m:
                return b["id"] # Return Opponent ID
        return None

        
    def _calculate_packing(self, start_idx, end_idx, passing_team, tracks, ball_tracks):
        """
        Calculate Packing Rate: Number of defenders bypassed by the pass.
        Logic: Defenders between ball start X and ball end X.
        """
        try:
            # Ball positions
            ball_start = ball_tracks[start_idx]
            ball_end = ball_tracks[end_idx]
            if not ball_start or not ball_end:
                return 0
            
            # Use centroid for ball
            bx_start = (ball_start[0] + ball_start[2]) / 2
            bx_end = (ball_end[0] + ball_end[2]) / 2
            
            # Check direction (Forward pass?)
            # Valid packing usually implies getting closer to goal.
            # We count simply "bypassed" in longitudinal direction.
            
            # Defenders at START of pass
            frame_tracks = tracks.get(start_idx, [])
            defenders = [t for t in frame_tracks if t['team'] != passing_team and t['role'] != 'advertisement']
            
            packed_count = 0
            min_x = min(bx_start, bx_end)
            max_x = max(bx_start, bx_end)
            
            # Heuristic: Only count if pass moves significantly (e.g. > 2m)
            if abs(bx_end - bx_start) < 2.0:
                return 0

            for d in defenders:
                dx, dy = d['bbox_pitch'] # Assuming transformed coordinates
                if min_x < dx < max_x:
                    packed_count += 1
            
            return packed_count
        except Exception as e:
            # print(f"Packing error: {e}")
            return 0

    def _calculate_defensive_density(self, frame_idx, shooter_team, tracks, ball_tracks):
        """
        Calculate Defensive Density for xG: Defenders in the cone between ball and goal.
        """
        try:
            ball = ball_tracks[frame_idx]
            if not ball:
                return 0, 0.0
            
            bx, by = (ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2
            
            # Determine Goal Target
            # If x > 52.5 (Length/2), attacking Right Goal (105, 34)
            # Else attacking Left Goal (0, 34)
            PITCH_L = 105.0
            PITCH_W = 68.0
            GOAL_Y_MIN = 30.34
            GOAL_Y_MAX = 37.66
            
            if bx > PITCH_L / 2:
                goal_x = PITCH_L
            else:
                goal_x = 0
                
            # Define Cone Triangle: (bx,by), (goal_x, GOAL_Y_MIN), (goal_x, GOAL_Y_MAX)
            # Simplification: Count defenders inside this triangle
            
            frame_tracks = tracks.get(frame_idx, [])
            defenders = [t for t in frame_tracks if t['team'] != shooter_team and t['role'] != 'advertisement']
            
            density = 0
            
            def point_in_triangle(px, py, p0x, p0y, p1x, p1y, p2x, p2y):
                # Barycentric coordinates
                area = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)
                s = 1 / (2 * area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py)
                t = 1 / (2 * area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py)
                return s > 0 and t > 0 and (1 - s - t) > 0

            for d in defenders:
                dx, dy = d['bbox_pitch']
                if point_in_triangle(dx, dy, bx, by, goal_x, GOAL_Y_MIN, goal_x, GOAL_Y_MAX):
                    density += 1
            
            # Heuristic xG: Base 0.30 - (0.05 * density) - (dist_factor...)
            # We just return density and a simple penalized xG
            base_xg = 0.30 # Average big chance
            xg_value = max(0.01, base_xg - (0.05 * density))
            
            return density, xg_value
            
        except Exception as e:
            # print(f"xG error: {e}")
            return 0, 0.0
        return False
        
    def _retains_possession(self, start_frame, pid, ownership, duration_s=2.0):
        frames = int(duration_s * EFF_FPS)
        end_frame = min(len(ownership), start_frame + frames)
        
        # Check if pid owns majority of frames in window
        count = 0
        for f in range(start_frame, end_frame):
            if ownership[f] == pid:
                count += 1
        
        return count > (frames * 0.5)

    def _dist_to_goal(self, ball_px):
        # Center Goal (105, 34) or (0, 34)
        try:
             bx, by = self.camera.project_point(ball_px[0], ball_px[1])
             g1 = (0, 34.0)
             g2 = (105.0, 34.0)
             d1 = np.sqrt((bx-g1[0])**2 + (by-g1[1])**2)
             d2 = np.sqrt((bx-g2[0])**2 + (by-g2[1])**2)
             return min(d1, d2)
        except: return 50.0

    def _check_opp_cone(self, frame_idx, pid, player_tracks, ball_px):
        # Placeholder: Check if any opp is < 3m towards goal
        return self._is_opponent_near(frame_idx, pid, player_tracks, dist_m=3.0)
