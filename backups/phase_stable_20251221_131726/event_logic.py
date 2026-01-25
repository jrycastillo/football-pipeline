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

# Constants (Meters)
DIST_TOUCH = 1.5
DIST_DRIBBLE_OPP = 2.0
DIST_PASS_MIN = 3.0
TIME_DRIBBLE_RETAIN = 2.0 # Seconds

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
            elif last is not None and gap < 5:
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

        # Helper: Calculate xG (Existing)
        def calculate_xg(start_pos, header=False):
             # ... (Existing logic) ...
             dx = GOAL_X - start_pos[0]
             dy = max(0, abs(start_pos[1] - GOAL_CENTER_Y) - GOAL_WIDTH_HALF)
             dist = math.hypot(GOAL_X - start_pos[0], GOAL_CENTER_Y - start_pos[1])
             if dist < 0.1: dist = 0.1
             angle = math.atan2(7.32, dist)
             xg = 0.75 * math.exp(-0.15 * dist) * (angle * 1.5)
             if header: xg *= 0.6
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
        for t, pid in enumerate(ownership):
            if pid is not None:
                stats[pid]["touch_frames"] += 1
                
                # Check Dribble (Opponent < 2m)
                if self._is_opponent_near(t, pid, player_tracks, dist_m=DIST_DRIBBLE_OPP):
                    stats[pid]["dribbles"] += 1
                    # Check Success (Retain for 2s)
                    if self._retains_possession(t, pid, ownership, duration_s=TIME_DRIBBLE_RETAIN):
                         stats[pid]["dribbles_successful"] += 1
                         
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
                    # Box: x > 88.5 and |y - 34| < 20.15
                    
                    is_cross = False
                    if (start_pos[1] < 9 or start_pos[1] > 59) and (end_pos[0] > 88.5 and 13.85 < end_pos[1] < 54.15):
                        is_cross = True
                        stats[p_a]["crosses"] += 1
                        if is_complete:
                             stats[p_a]["crosses_accurate"] += 1
                        events.append({"type": "cross", "from": p_a, "to": p_b, "frame": seg_a["end"]})

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
                
                # Shot Threshold: > 15 m/s (~54 km/h) and direction towards Goal (X=105)
                if speed_mps > 15.0 and m2[0] > m1[0]: # Moving towards Right Goal
                    # Check if inside goal coordinates at X=105
                    # Simple linear projection
                    if abs(m2[0] - m1[0]) > 0.1:
                        slope = (m2[1] - m1[1]) / (m2[0] - m1[0])
                        y_at_goal = m2[1] + slope * (105.0 - m2[0])
                        
                        if 30.34 < y_at_goal < 37.66:
                            # Potential Shot on Target
                            # Attribute to last possessor
                            # Find who had ball last
                            shooter = ownership[i] or ownership[i-5] # Look back
                            if shooter:
                                # Debounce: Don't count same shot multiple times
                                # (Omitted for brevity, assuming minimal false positives or dedupe later)
                                # Check if already counted in last 10 frames?
                                recent = [e for e in events if e["type"] == "shot" and abs(e["frame"] - i) < 10]
                                if not recent:
                                    xg = calculate_xg(m1)
                                    stats[shooter]["shots_on_target"] += 1
                                    stats[shooter]["xg_foot_opponent_present"] += xg # Accumulate xG
                                    events.append({
                                        "type": "shot", 
                                        "player": shooter, 
                                        "frame": i, 
                                        "xg": round(xg, 2),
                                        "speed": round(speed_mps, 1)
                                    })
                                    # print(f"SHOT! Player {shooter} | Speed {speed_mps:.1f} m/s | xG {xg:.2f}")

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
                
                # If Shot Incoming (>15 m/s towards goal)
                if speed_mps > 15.0 and (m2[0] < 5.0 or m2[0] > 100.0): # Near Goal Ends
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
                                               prev_y_min = gk_box_y_min # Simplified, need history
                                               # Heuristic: just assume 'jumping' if strictly high in image? 
                                               # Or check if y_min decreased (moved up) significantly compared to i-5
                                               stats[gk_id]["jumping_saves"] += 0 # Placeholder for complex logic

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
        if not my_box: return False
        
        my_c = bbox_center(my_box["xyxy"])
        
        for b in boxes:
            if b["id"] == pid or b["id"] is None: continue
            
            c = bbox_center(b["xyxy"])
            d = self.camera.calculate_distance(my_c, c)
            if d < dist_m:
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
