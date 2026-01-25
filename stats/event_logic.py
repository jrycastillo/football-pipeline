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
except:
    HEURISTICS = {}
    FPS = 25

# Constants (Meters) - Phase 190/195: Relaxed thresholds
DIST_TOUCH = 5.0  # Increased for better possession detection
DIST_DRIBBLE_OPP = 3.0  # Phase 195: Increased from 2.0 to 3.0m for more dribble detection
DIST_PASS_MIN = 2.0  # Reduced to detect shorter passes
TIME_DRIBBLE_RETAIN = 1.5  # Phase 195: Reduced from 2.0s to 1.5s for quicker dribble success
SHOT_SPEED_THRESHOLD = 8.0  # Phase 195: Reduced from 15 m/s to 8 m/s (29 km/h)

def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

class AdvancedEventDetector:
    def __init__(self):
        self.camera = Camera() # Default Homography
        
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
            elif last is not None and gap < 15:  # Phase 190: Increased from 5 to 15 for better smoothing
                smoothed[i] = last
                gap += 1
        return smoothed

    def analyze(self, ownership, player_tracks, ball_track, team_map=None):
        """
        Detects: Dribbles, Passes, Crosses, Tackles, Interceptions, Goals, XG.
        team_map: dict {pid (str): "TeamName"}
        """
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
        
        prev_pos = {} # pid -> (x,y)
        
        for t, frame_data in enumerate(player_tracks):
            boxes = frame_data.get("boxes", [])
            for b in boxes:
                pid = b.get("id")
                if pid is None: continue
                
                c = bbox_center(b["xyxy"])
                world_c = self.camera.project_point(c[0], c[1]) # Returns (x_m, y_m)
                
                if pid in prev_pos:
                    px, py = prev_pos[pid]
                    dx = world_c[0] - px
                    dy = world_c[1] - py
                    dist = math.hypot(dx, dy)
                    
                    # Sanity check: Human speed limit (~10m/s -> 0.4m per frame @ 25fps)
                    # If jump is too big, ignore (teleport/tracking error)
                    if dist < 1.0: 
                        stats[pid]["distance_m"] += dist
                
                # Phase 83: Track Class ID (Ball=0, GK=1, Player=2, Ref=3)
                cls = b.get("cls", 2)
                if "class_counts" not in stats[pid]:
                     stats[pid]["class_counts"] = defaultdict(int)
                stats[pid]["class_counts"][cls] += 1
                
                prev_pos[pid] = world_c

        # Helper: Calculate xG (Updated to handle both goals)
        def calculate_xg(start_pos, header=False, under_pressure=False, goal_x=None):
            # If goal_x not specified, use nearest goal
            if goal_x is None:
                dist_to_right = abs(start_pos[0] - GOAL_X)
                dist_to_left = abs(start_pos[0] - 0.0)
                goal_x = GOAL_X if dist_to_right < dist_to_left else 0.0

            dist = math.hypot(goal_x - start_pos[0], GOAL_CENTER_Y - start_pos[1])
            if dist < 0.1: dist = 0.1
            angle = math.atan2(7.32, dist)
            xg = 0.75 * math.exp(-0.15 * dist) * (angle * 1.5)
            if header: xg *= 0.6
            if under_pressure: xg *= 0.75  # Pressure factor from xg.py
            return min(0.99, max(0.01, xg))

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

        # 1. Possession & Dribbling
        dribble_debug_count = 0
        for t, pid in enumerate(ownership):
            if pid is not None:
                stats[pid]["touch_frames"] += 1
                
                # Check Dribble (Opponent within range)
                opp_id = self._is_opponent_near(t, pid, player_tracks, dist_m=DIST_DRIBBLE_OPP)
                if opp_id is not None:
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
                        stats[opp_id]["tackles"] += 1  # Phase 195: Credit tackle on failed dribble
                        stats[opp_id]["tackles_successful"] += 1

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

        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i+1]
            
            p_a = seg_a["pid"]
            p_b = seg_b["pid"]
            
            if p_a is None or p_b is None: continue # Ball lost or gained from nowhere
            if p_a == p_b: continue
            
            # Pass Attempt A -> B (Endpoint Logic)
            # Old Logic: Required A and B to be adjacent in segments.
            # New Logic: A ... (Gap) ... B
            # We iterate adjacent segments, but the "Gap" is handled by ownership array being None?
            # segments are calculated based on NON-NONE ownership.
            # So if A has ball, then None, then B has ball.
            # 'segments' list skips None.
            # So seg[i] is A, seg[i+1] is B.
            # This ALREADY implements "Endpoint Logic" effectively, because we ignore the gap.
            # The issue is strictness of "Interception".
            
            # Verify distance
            start_pos = ball_track[seg_a["end"]]
            end_pos = ball_track[seg_b["start"]]

            if start_pos and end_pos:
                dist = self.camera.calculate_distance(start_pos, end_pos)
                if dist > DIST_PASS_MIN:
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
                        # ... (existing interception logic) ...
                        if team_map and team_a and team_b and team_a != team_b:
                            stats[p_b]["interceptions"] += 1
                            stats[p_b]["ball_interceptions_total"] += 1 # Sync name
                            
                            # Ball Recovery in Opp Half
                            # If p_b recovers ball and x > 52.5 (Assuming p_b attacking direction >?)
                            # Actually "Opponent's Half" depends on team side.
                            # Heuristic: If X > 52.5 (Right Half), recovery count? 
                            # We don't know team sides easily without manual input.
                            # Simplification: If loc > 52.5, assume it's attacking half implies high press?
                            # Or just count raw: "recoveries_high"
                            if end_pos[0] > 52.5: # Assuming recovered in right half
                                 stats[p_b]["ball_recoveries_opp_half"] += 1
                            elif end_pos[0] < 52.5:
                                 stats[p_b]["ball_recoveries_own_half"] += 1
                            
                            events.append({"type": "interception", "by": p_b, "frame": seg_b["start"]})



                    # 4. Check Shots / Key Passes (End in Box)
                    # If end_pos is in box and NO next possession or Goal?
                    # "Shot" logic usually implies high velocity towards goal.
                    # Simplified: If Pass ends in Box and is NOT complete (or complete to shooter?), maybe shot?
                    # Better: Analzye Ball Trajectory for Shots (Speed > 15m/s towards goal)
                    
        # 5. Shot Detection (Trajectory Analysis)
        # Iterate ball track for High Velocity > Goal
        frames_count = len(ball_track)
        for i in range(2, frames_count):
            if ball_track[i] and ball_track[i-2]:
                # Calcluate Velocity
                p1 = ball_track[i-2]
                p2 = ball_track[i]
                
                # Distance in meters (using Camera.calculate_distance logic, but we have world coords if projected)
                # Wait, ball_track is currently PIXELS.
                # Project to Meters!
                # We need self.camera
                
                m1 = self.camera.project_point(p1[0], p1[1])
                m2 = self.camera.project_point(p2[0], p2[1])
                
                dist_m = math.hypot(m2[0]-m1[0], m2[1]-m1[1])
                speed_mps = dist_m / (2.0 / FPS) # 2 frames @ 25fps = 0.08s
                
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
                            if shooter:
                                # Debounce: Don't count same shot multiple times
                                # Check if already counted in last 10 frames?
                                recent = [e for e in events if e["type"] == "shot" and abs(e["frame"] - i) < 10]
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

                                    # --- GOAL DETECTION (Phase 85) ---
                                    # Check if ball continues INTO net
                                    # Look ahead 10 frames
                                    goal_confirmed = False
                                    for k in range(i, min(i+10, len(ball_track))):
                                        if ball_track[k]:
                                            mk = self.camera.project_point(ball_track[k][0], ball_track[k][1])
                                            # Check correct goal based on shot direction
                                            if moving_right and mk[0] > 105.0 and (30.34 < mk[1] < 37.66):
                                                goal_confirmed = True
                                                break
                                            elif moving_left and mk[0] < 0.0 and (30.34 < mk[1] < 37.66):
                                                goal_confirmed = True
                                                break

                                    if goal_confirmed:
                                        stats[shooter]["goals"] += 1
                                        stats[shooter]["goals_total"] += 1 # Sync name
                                        events.append({"type": "goal", "player": shooter, "frame": i, "assist": None})

                                    # print(f"SHOT! Player {shooter} | Speed {speed_mps:.1f} m/s | xG {xg:.2f}")

        # 6. GK Save Detection (Post-Hoc Analysis of Trajectories)
        # Look for Ball near Goal -> High Speed -> Sudden Stop/Deflection near "GK"
        # We need to know WHO IS GK first.
        # Heuristic: Player with most 'frames_in_box' (>90%) is GK (per team?)
        # Let's find candidate GKs from stats
        possible_gks = []
        for pid, s in stats.items():
             if s["frames_in_box"] > 500: # Minimum frames
                 possible_gks.append(pid)
        
        # Iterate high-speed ball segments again
        for i in range(2, frames_count - 5):
            if ball_track[i] and ball_track[i-2]:
                m1 = self.camera.project_point(ball_track[i-2][0], ball_track[i-2][1])
                m2 = self.camera.project_point(ball_track[i][0], ball_track[i][1])
                dist_m = math.hypot(m2[0]-m1[0], m2[1]-m1[1])
                speed_mps = dist_m / (2.0 / FPS)
                
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
                                     speed_next = dist_next / (3.0 / FPS)
                                     
                                     if speed_next < 5.0 or (np.sign(v_next_x) != np.sign(m2[0]-m1[0])):
                                          # SAVE DETECTED!
                                          # Use a debounce to avoid multi-counting same save
                                          recent_saves = [e for e in events if e["type"] == "save" and abs(e["frame"] - i) < 20]
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
        # Detect change of possession where Opponent was near
        for i in range(len(segments) - 1):
             seg_a = segments[i]
             seg_b = segments[i+1]
             p_a = seg_a["pid"]
             p_b = seg_b["pid"]
             
             if p_a is not None and p_b is not None and p_a != p_b:
                 # Change occurred. Was B near A at the end of A's stint?
                 end_frame = seg_a["end"]
                 if self._is_opponent_near(end_frame, p_a, player_tracks, dist_m=DIST_TOUCH): # Close encounter
                     # A lost ball to B.
                     # Credit B with Tackle
                     stats[p_b]["tackles"] += 1
                     stats[p_b]["tackles_successful"] += 1
                     events.append({"type": "tackle", "by": p_b, "on": p_a, "frame": end_frame})
                     
        # 4. Shooting & xG
        # Heuristic: Ball moves towards Goal fast, no receiver
        # Ideally needs Velocity vector.
        # Simple Logic: Last touch in Box -> xG
        
        for seg in segments:
            pid = seg["pid"]
            if pid is None: continue
            
            end_f = seg["end"]
            ball_pos = ball_track[end_f]
            
            if ball_pos and self.camera.is_in_penalty_box(ball_pos):
                # Potential Shot
                # Calculate xG
                angle = self.camera.get_shot_cone_angle(ball_pos)
                dist_g = self._dist_to_goal(ball_pos)
                
                # Simple Model: xG = 0.1 * (10/dist) * (angle/45)
                if dist_g > 0:
                    xg = 0.1 * (10.0 / dist_g) * (angle / 45.0)
                else:
                    xg = 0.5
                xg = min(0.99, xg)
                
                # Opponent Cone?
                opp_present = self._check_opp_cone(end_f, pid, player_tracks, ball_pos)
                
                if opp_present:
                    stats[pid]["xg_foot_opponent_present"] += xg
                else:
                    stats[pid]["xg_foot_no_opponent"] += xg
                    
                # Count as shot? Not every touch in box is a shot.
                # Only if ball leaves player and goes near goal.
                # Tracking this without velocity is hard.
                # We will log xG accumulation for touches in High Danger Zone.
                
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
        frames = int(duration_s * FPS)
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
