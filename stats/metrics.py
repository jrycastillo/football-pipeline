from collections import defaultdict
import math
from .event_logic import AdvancedEventDetector

class StatsEngine:
    def __init__(self):
        self.detector = AdvancedEventDetector()
        
    def process_events(self, all_frames, id_manager=None):
        """
        Process full video history to generate stats.
        """
        player_tracks = all_frames
        
        # Extract Ball Track
        from vision.ball_tracking import BallTracker
        ball_tracker = BallTracker()
        for t, f in enumerate(all_frames):
             ball_tracker.update(t, f["boxes"])
        ball_track = ball_tracker.interpolate(len(all_frames))
        
        balls_found = sum(1 for b in ball_track if b is not None)
        print(f"[StatsEngine] Found/Interpolated ball in {balls_found}/{len(all_frames)} frames.")

        if id_manager:
            self._cluster_teams(id_manager)

        # 1. Map Possession
        ownership = self.detector.calculate_ownership(player_tracks, ball_track)
        
        # 2. Detect Events & Get Stats
        team_map_ref = self.team_map if hasattr(self, "team_map") else None
        events, raw_stats = self.detector.analyze(ownership, player_tracks, ball_track, team_map=team_map_ref)
        
        # Phase 192/216: Remap raw_stats from track IDs to jersey numbers
        # This ensures passes/events are attributed to jersey numbers, not track IDs
        # Phase 216: Also builds reverse lookup from jersey_registry for soft-registered jerseys
        if id_manager:
            remapped_stats = {}
            remap_count = 0
            
            # Build reverse lookup: track_id -> jersey_number from jersey_registry
            track_to_jersey = {}
            if hasattr(id_manager, 'active_bindings'):
                track_to_jersey.update(id_manager.active_bindings)
            if hasattr(id_manager, 'jersey_registry'):
                for jersey_num, info in id_manager.jersey_registry.items():
                    if isinstance(info, dict) and 'track_id' in info:
                        tid = info['track_id']
                        if tid not in track_to_jersey:  # active_bindings takes priority
                            track_to_jersey[tid] = jersey_num
            
            for track_id, stats_dict in raw_stats.items():
                # Check if this track ID maps to a jersey number
                jersey_num = track_to_jersey.get(track_id)
                target_id = jersey_num if (jersey_num is not None and jersey_num != track_id) else track_id
                if jersey_num is not None and jersey_num != track_id:
                    remap_count += 1
                
                # Initialize target if needed
                if target_id not in remapped_stats:
                    remapped_stats[target_id] = defaultdict(int)
                
                # Merge stats - handle both int values and nested dicts
                for key, value in stats_dict.items():
                    if isinstance(value, (int, float)):
                        remapped_stats[target_id][key] += value
                    else:
                        # If value is a dict or other type, just assign (don't accumulate)
                        remapped_stats[target_id][key] = value
            raw_stats = remapped_stats
            print(f"[Phase 216] Remapped {remap_count} track IDs to jersey numbers (active_bindings + jersey_registry)")
        
        # 3. Final Formatting
        formatted_stats = {}
        
        # Get all distinct IDs from stats + jersey registry
        all_ids = set(raw_stats.keys())
        if id_manager:
            all_ids.update(id_manager.jersey_registry.keys())

        # Pre-calculate Minutes Played for Noise Filtering (Task 3)
        id_frame_counts = defaultdict(int)
        for f in all_frames:
             for b in f["boxes"]:
                 if b["id"] is not None:
                     id_frame_counts[b["id"]] += 1

        # Phase 216 Fix: Remap frame counts to jersey numbers
        if id_manager:
            track_to_jersey = {}
            if hasattr(id_manager, 'active_bindings'):
                track_to_jersey.update(id_manager.active_bindings)
            if hasattr(id_manager, 'jersey_registry'):
                for jersey_num, info in id_manager.jersey_registry.items():
                    if isinstance(info, dict) and 'track_id' in info:
                        tid = info['track_id']
                        if tid not in track_to_jersey:
                            track_to_jersey[tid] = jersey_num
            
            # Aggregate frame counts for mapped entities
            for tid, count in list(id_frame_counts.items()):
                if tid in track_to_jersey:
                    jnum = track_to_jersey[tid]
                    # Add to jersey count
                    id_frame_counts[jnum] += count
                    # Note: We don't remove the track ID count, as raw_stats might still have it? 
                    # Actually raw_stats was remapped destructivelyish (remapped_stats).
                    # But it's safer to keep both or ensure keys match.


        for id_key in all_ids:
            if id_key is None: continue
            
            # --- FILTER NOISE ---
            # Calculate total time presence
            # If known jersey, we trust it more, but let's check actual track presence or usage?
            # Issue: id_key might be '10' (Jersey) or '1235' (Track).
            # If it's a Jersey ID, we need to count frames where binding -> Jersey.
            # But here we are iterating KEYS.
            # Simpler approach: check if 'time_on_ball' > X? No, defenders don't touch ball.
            # Use raw frame count if possible.
            # If id_key is int (Jersey), it means we found it or it was in registry.
            # If id_key is track, we use id_frame_counts.
            
            # Robust Frame Count Lookup
            # Note: Stats engine doesn't track frame count per ID in 'raw_stats', only events.
            # So we use our pre-calc.
            # But wait, 'all_frames' boxes have 'id' updated to Jersey Number if bound/replaced in orchestrator?
            # Orchestrator does: "box['id'] = jersey_num".
            # So id_frame_counts should contain Jersey Numbers for bound frames.
            
            total_frames = id_frame_counts.get(id_key, 0)
            if isinstance(id_key, str) and id_key.isdigit(): # Handle stringified ints
                 total_frames = id_frame_counts.get(int(id_key), 0) + id_frame_counts.get(id_key, 0)
            
            minutes_played = (total_frames / 25.0) / 60.0 # Just for reference
            seconds_played = (total_frames / 25.0)
            
            # Strict Filter: < 3.0 Seconds -> DELETE (Task 3)
            # User request: "Delete ANY player object that has total_time_on_pitch < 3.0 seconds."
            # EXCEPTION: If they scored a goal, KEEP THEM!
            goals_detected = raw_stats.get(id_key, {}).get("goals", 0)
            if seconds_played < 3.0 and goals_detected == 0:
                # print(f"Skipping {id_key} (played {seconds_played:.2f}s)")
                continue

                    # Identify Player
            is_known = False
            
            # Check 1: Is it a valid Jersey Number in Registry?
            if id_manager and id_manager.is_jersey_number(id_key):
                is_known = True

            # Check 2: (CRITICAL FIX) Did this ID actually appear as a BOUND Jersey?
            # If id_key is "1" (Track ID), and "1" is in Registry (Goalie), is_known=True.
            # But if Track 1 was never swapped/bound to Jersey 1, then "1" is just a Track.
            # We assume Orchestrator swaps IDs. So if "1" is a Jersey, it must appeal in active_bindings.values().
            # BUT: active_bindings only holds CURRENT frame bindings.
            # We need historical knowledge.
            # However, StatsEngine processes `all_frames` which HAS the swapped IDs.
            # If "1" appears in `raw_stats`, it means some frame had ID 1.
            # If Orchestrator swapped it, it's a Jersey.
            # If Orchestrator didn't (Unbound Track 1), it's a Track.
            # COLLISION: We cannot distinguish "Jersey 1" from "Unbound Track 1" if both are present in `raw_stats[1]`.
            # HEURISTIC: If id_key is small (< 100) and in Registry, we prioritize Jersey interpretation.
            # BUT User hates "ID 1 -> Jersey 1" if it's actually Track 1.
            # Let's use the 'active_bindings' check as a proxy for "Was ever bound"?
            # No, active_bindings is transient.
            
            # Revised approach: Only trust as Jersey if it's explicitly in the registry AND we trust the source.
            # For now, we stick to is_known, but we fix the FALLBACKS.
            
            if is_known:
                final_key = str(id_key)
                jersey_num = int(id_key)
                
                # Use Clustered Team Map if available
                if hasattr(self, "team_map") and final_key in self.team_map:
                    team_name = self.team_map[final_key]
                else:
                    team_name = id_manager.get_player_color(jersey_num) if id_manager else "Unknown"
                    if team_name == "Unknown": team_name = "Unknown"
                
                player_name = f"Player {jersey_num}"
            else:
                # Unknown Track
                final_key = f"Unknown_{id_key}"
                jersey_num = None # STRICT: Nullify Jersey Number
                team_name = "Unknown"
                
                # Task 56: Force Color for Unknowns
                if hasattr(self, "team_map") and final_key in self.team_map:
                    team_name = self.team_map[final_key]
                elif id_manager:
                    # Check track_colors
                    raw_key = id_key
                    if isinstance(id_key, str) and id_key.isdigit(): raw_key = int(id_key)
                    raw_color = id_manager.track_colors.get(raw_key, "Unknown")
                    if raw_color != "Unknown":
                        team_name = raw_color
                
                if team_name == "Unknown":
                     team_name = "White" 
                     
                # FIX: Initialize player_name for Unknowns to avoid leaking previous value
                player_name = f"Unknown Player {id_key}"
            
            # Get Metrics
            s = raw_stats.get(id_key, defaultdict(int)) 
            
            # Derived Metrics
            fps_val = 25.0
            time_on_ball = round(s["touch_frames"] / fps_val, 2)
            
            p_total = s["passes_total"]
            p_comp = s["passes_complete"]
            acc_pass = (p_comp / p_total) * 100.0 if p_total > 0 else 0.0
            
            formatted_stats[final_key] = {
                "player_name": player_name,
                "jersey_number": jersey_num,
                "team": team_name, 
                "position": "Player", 
                "role": "Player",
                # Phase 216: Confidence Metadata
                "observations": total_frames,
                "confidence_score": round(min(1.0, total_frames / 100.0), 2),  # Normalize to 0-1
                "soft_registered": (id_manager.jersey_registry.get(jersey_num, {}).get("soft", False) if isinstance(id_manager.jersey_registry.get(jersey_num), dict) else False) if id_manager and jersey_num else False,
                "stats": {
                    "total_distance": round(s["distance_m"], 2),
                    "time_on_ball_s": time_on_ball,
                    "touch_frames": s["touch_frames"],
                    
                    # Offensive (Target)
                    "goals_total": s["goals"],
                    "shots_on_target_total": s["shots_on_target"],
                    "shots_wide_total": 0, 
                    "penalty_total": 0,
                    
                    # Offensive (Delivering)
                    "crosses_total": s["crosses_total"],
                    "crosses_accurate_total": s["crosses_complete"],
                    "dribbles_total": s["dribbles"],
                    "dribbles_successful_total": s["dribbles_successful"],
                    
                    # Interaction
                    "passes_total": p_total,
                    "passes_accurate": p_comp,
                    "accurate_passes_percent": round(acc_pass, 1),
                    
                    # Defensive
                    "challenges_total": s["challenges_total"], # Fixed from dribbles
                    "challenges_won_total": s["challenges_won_total"],
                    "tackles_total": s["tackles"],
                    "tackles_successful_total": s["tackles_successful"],
                    "ball_interceptions_total": s["interceptions"], # Mapped
                    "fouls_total": s["fouls_total"],
                    
                    # xG
                    "xg_foot_no_opponent": round(s["xg_foot_no_opponent"], 2),
                    "xg_header_no_opponent": 0.0,
                    "xg_foot_opponent_present": round(s["xg_foot_opponent_present"], 2),
                    "xg_header_opponent_present": 0.0,
                    
                    # --- ADVANCED STATS (Phase 86) ---
                    # Offensive
                    "blocked_shots_by_opponent": s.get("blocked_shots", 0),
                    "shots_on_post_bar": 0, # Model Gap
                    "goals_standard_situation": 0, # Model Gap
                    "free_kick_scored": 0, # Model Gap
                    
                    # Delivering
                    "expected_assists": round(s.get("expected_assists", 0.0), 2),
                    "offsides_total": 0, # Complex Heuristic Gap
                    "packing_total": s.get("packing", 0),
                    
                    # Interaction
                    "ball_touches_total": s["touch_frames"], # Correct mapping
                    
                    # Defensive
                    "fouls_suffered": 0, # Inverse of committed (Model Gap)
                    "ball_recoveries_opp_half": s.get("ball_recoveries_opp_half", 0),
                    "played_offside": 0, # Gap
                    
                    # Categorization (Passes)
                    "foot_passes_open_play_total": 0, # Model Gap
                    "hand_passes_total": 0, # Model Gap
                    "short_passes_total": s.get("short_passes", 0),
                    "medium_passes_total": s.get("medium_passes", 0),
                    "long_passes_total": s.get("long_passes", 0),
                    "accurate_foot_passes_open_play_total": 0,
                    "hand_passes_accurate_total": 0,
                    "short_passes_accurate_total": s.get("short_passes_accurate", 0),
                    "medium_passes_accurate_total": s.get("medium_passes_accurate", 0),
                    # "long_passes_accurate_total" is "accurate_long_passes_total" (Existing)
                    "accurate_long_passes_total": s["accurate_long_passes"],    

                    # Categorization (Shots)
                    "close_range_shots_total": s.get("close_range_shots", 0),
                    "mid_range_shots_total": s.get("mid_range_shots", 0),
                    "long_range_shots_total": s.get("long_range_shots", 0),

                    # GK Stats
                    "shots_saved_total": s["shots_saved_total"],
                    "close_range_saved_total": s["close_range_saves"],
                    "mid_range_saved_total": s["mid_range_saves"],
                    "long_range_saved_total": s["long_range_saves"],
                    "jumping_saves_total": s["jumping_saves"],
                    "saves_without_jumping_total": s["shots_saved_total"] - s["jumping_saves"],
                    "penalties_saved_total": s["penalties_saved"],
                    "freekick_saved_total": s["freekick_saved"],
                    "corners_saved_total": s["corners_saved"],
                    "goals_standard_situation_conceded": 0, # Gap
                    "xg_per_shot_saved": round(s.get("xg_saved_sum", 0) / max(1, s["shots_saved_total"]), 2),

                    "goals_conceded": s["goals_conceded"]
                }
            }
            
            # --- STRICT POSITION & ROLE LOGIC (Phase 84 Fix) ---
            
            # Check for Color Outlier (Referee / GK Candidates)
            is_outlier = False
            if hasattr(self, "primary_teams") and len(self.primary_teams) >= 2:
                 t_clean = team_name.capitalize() if team_name else "Unknown"
                 if t_clean not in self.primary_teams and t_clean != "Unknown":
                     is_outlier = True
            
            percent_in_box = s.get("frames_in_box", 0) / max(1, s.get("pos_count", 1))

            # 1. REFEREE LOGIC
            # Outlier Color AND No Jersey Number (Qwen ignores refs)
            if is_outlier and jersey_num is None:
                 formatted_stats[final_key]["position"] = "Referee"
                 formatted_stats[final_key]["role"] = "Referee"
                 
            # 2. GOALKEEPER LOGIC
            # Must be Outlier AND >80% in Box (User requested >80%)
            elif is_outlier and percent_in_box > 0.80:
                 formatted_stats[final_key]["position"] = "GK"
                 formatted_stats[final_key]["role"] = "Goalkeeper"
            
            # Exception: Jersey #1 is always Goalkeeper? (Traditional Logic)
            if jersey_num == 1:
                 formatted_stats[final_key]["position"] = "GK"
                 formatted_stats[final_key]["role"] = "Goalkeeper"
            
            # FINAL FILTER (Task 106): Drop Unknown Players
            # user request: "I need all of the stats" -> DO NOT DELETE UNKNOWNS.
            current_role = formatted_stats[final_key]["role"]
            if current_role == "Player" and jersey_num is None:
                # Fallback: Ensure they at least have a distinct name
                if "Unknown" not in formatted_stats[final_key]["player_name"]:
                     formatted_stats[final_key]["player_name"] = f"Unknown Player {final_key}"
                pass # KEEP EVERYONE
            
        return formatted_stats, events

    def _cluster_teams(self, id_manager):
        """
        Force-assign every player to Team A or Team B based on Jersey Color.
        Merges similar colors and assigns Unknown players to nearest team.
        """
        # 1. Collect all color samples
        color_samples = [] # (r, g, b, pid)
        # IdentityManager stores player_colors as strings usually ("red", "white") or raw tuples?
        # Current implementation of `get_jersey_color` returns STRING.
        # If we only have strings, we just group by string.
        # "Red" -> Team A, "White" -> Team B.

        # But `IdentityManager.player_colors` stores what?
        # Check `vision/identity_manager.py`. line 10: "Maps Jersey Number -> Detected Color (e.g. 'Red')"
        # So we have strings.

        # Build jersey -> cls_id mapping to filter goalkeepers
        jersey_classes = {}
        if hasattr(id_manager, 'track_classes') and hasattr(id_manager, 'active_bindings'):
            for track_id, jersey_num in id_manager.active_bindings.items():
                cls_id = id_manager.track_classes.get(track_id)
                if cls_id is not None:
                    jersey_classes[jersey_num] = cls_id

        teams = defaultdict(list)
        for jersey, color in id_manager.player_colors.items():
            # Exclude goalkeepers (cls_id=1) from team clustering
            # Only use field players (cls_id=2) for team color determination
            cls_id = jersey_classes.get(jersey)
            if cls_id == 1:
                print(f"[Team] Skipping GK #{jersey} (color: {color}) from team clustering")
                continue
            teams[color].append(jersey)

        # === FIX #1: Merge similar colors ===
        # Colors like "Green" and "Lime" should be treated as the same team
        # NOTE: Cyan is kept separate as it's a common team color (not merged to Blue)
        color_merge_map = {
            "lime": "green",
            "teal": "green",
            "navy": "blue",
            "maroon": "red",
            "pink": "red",
            "silver": "white",
            "gray": "white",
            "gold": "yellow"
        }

        merged_teams = defaultdict(list)
        for color, jerseys in teams.items():
            if color == "Unknown":
                merged_teams["Unknown"].extend(jerseys)
            else:
                merged_color = color_merge_map.get(color.lower(), color.lower())
                merged_teams[merged_color.capitalize()].extend(jerseys)
        teams = dict(merged_teams)

        print(f"[Team] After merging similar colors: {dict((k, len(v)) for k, v in teams.items())}")

        # If we have mainly 2 colors, great.
        # If we have "Unknown", "Blue", "Red", "White"... we need to merge.
        # Heuristic: Top 2 most frequent colors are the teams.

        counts = {c: len(ids) for c, ids in teams.items() if c != "Unknown"}
        if len(counts) >= 2:
            top_2 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]
            team_a_color = top_2[0][0]
            team_b_color = top_2[1][0]
            
            # Update Identity Manager with EXPLICIT Team Names (The Color Itself)
            self.team_map = {}
            team_a_jerseys = [int(j) for j in teams[team_a_color]]
            team_b_jerseys = [int(j) for j in teams[team_b_color]]

            for j in teams[team_a_color]: self.team_map[str(j)] = team_a_color.capitalize() # "Red"
            for j in teams[team_b_color]: self.team_map[str(j)] = team_b_color.capitalize() # "White"

            # FIX: Also add track_id -> team mappings via active_bindings reverse lookup
            # This allows team validation to work for both jersey numbers AND track IDs
            if hasattr(id_manager, 'active_bindings'):
                for track_id, jersey_num in id_manager.active_bindings.items():
                    jersey_team = self.team_map.get(str(jersey_num))
                    if jersey_team and str(track_id) not in self.team_map:
                        self.team_map[str(track_id)] = jersey_team

            # Also add track_colors for unbound tracks
            if hasattr(id_manager, 'track_colors'):
                for track_id, color in id_manager.track_colors.items():
                    if str(track_id) not in self.team_map:
                        # Map color to team
                        color_lower = color.lower() if color else ""
                        if color_lower in [team_a_color.lower(), team_a_color]:
                            self.team_map[str(track_id)] = team_a_color.capitalize()
                        elif color_lower in [team_b_color.lower(), team_b_color]:
                            self.team_map[str(track_id)] = team_b_color.capitalize()

            print(f"[ReID Fix] team_map now has {len(self.team_map)} entries (jersey + track IDs)")

            # Helper for the rest
            self.primary_teams = {team_a_color.capitalize(), team_b_color.capitalize()}

            # === FIX #2: Force-assign Unknown players to nearest team ===
            # Calculate team averages (needed for both Unknown and GK assignment)
            avg_a = sum(team_a_jerseys) / len(team_a_jerseys) if team_a_jerseys else 15
            avg_b = sum(team_b_jerseys) / len(team_b_jerseys) if team_b_jerseys else 25

            print(f"[Team] Debug: 'Unknown' in teams = {'Unknown' in teams}")
            print(f"[Team] Debug: team_a_jerseys = {team_a_jerseys}")
            print(f"[Team] Debug: team_b_jerseys = {team_b_jerseys}")
            print(f"[Team] Team {team_a_color} avg jersey: {avg_a:.1f}")
            print(f"[Team] Team {team_b_color} avg jersey: {avg_b:.1f}")

            # FIX V5.2: Allow Unknown assignment even if one team has few/no players
            # Changed from 'and' to 'or' - only need at least ONE team to have players
            if "Unknown" in teams and (team_a_jerseys or team_b_jerseys):
                print(f"[Team] Starting Unknown player assignment for {len(teams['Unknown'])} players")

                for unknown_jersey in teams["Unknown"]:
                    try:
                        jersey_num = int(unknown_jersey)

                        # Handle case where one team might be empty
                        if team_a_jerseys and team_b_jerseys:
                            # Both teams exist - assign to closer team
                            dist_a = abs(jersey_num - avg_a)
                            dist_b = abs(jersey_num - avg_b)
                            if dist_a < dist_b:
                                assigned_team = team_a_color.capitalize()
                                confidence = "high" if dist_a < 10 else "medium"
                            else:
                                assigned_team = team_b_color.capitalize()
                                confidence = "high" if dist_b < 10 else "medium"
                        elif team_a_jerseys:
                            # Only team A exists - assign all Unknown to team B
                            assigned_team = team_b_color.capitalize()
                            confidence = "medium"
                        else:
                            # Only team B exists - assign all Unknown to team A
                            assigned_team = team_a_color.capitalize()
                            confidence = "medium"

                        self.team_map[str(unknown_jersey)] = assigned_team
                        print(f"[Team] Assigned Unknown #{unknown_jersey} → {assigned_team} ({confidence} confidence)")

                    except (ValueError, TypeError):
                        print(f"[Team] Warning: Could not parse jersey number for Unknown player: {unknown_jersey}")
            else:
                if "Unknown" in teams:
                    print(f"[Team] Warning: Unknown assignment skipped - team_a_jerseys={bool(team_a_jerseys)}, team_b_jerseys={bool(team_b_jerseys)}")
                else:
                    print(f"[Team] No Unknown players to assign")

            # === FIX #3: Assign Goalkeepers to Teams ===
            # GKs were excluded from team clustering, now assign them based on jersey number proximity
            gk_jerseys = [j for j, cls_id in jersey_classes.items() if cls_id == 1]
            if gk_jerseys and team_a_jerseys and team_b_jerseys:
                print(f"[Team] Assigning {len(gk_jerseys)} goalkeeper(s) to teams")
                for gk_jersey in gk_jerseys:
                    try:
                        jersey_num = int(gk_jersey)
                        dist_a = abs(jersey_num - avg_a)
                        dist_b = abs(jersey_num - avg_b)

                        # Assign GK to closer team by jersey number
                        if dist_a < dist_b:
                            assigned_team = team_a_color.capitalize()
                        else:
                            assigned_team = team_b_color.capitalize()

                        self.team_map[str(gk_jersey)] = assigned_team
                        gk_color = id_manager.player_colors.get(gk_jersey, "Unknown")
                        print(f"[Team] Assigned GK #{gk_jersey} (color: {gk_color}) → {assigned_team}")

                    except (ValueError, TypeError):
                        print(f"[Team] Warning: Could not parse GK jersey number: {gk_jersey}")

            # Default others to closest? Or Unknown.
        elif len(counts) == 1:
            # === SPECIAL CASE: Only 1 color detected ===
            # This means one team has clear colors, the other team is mostly Unknown
            # Assign all Unknown players to the opposite team
            print(f"[Team] Warning: Only 1 team color detected: {list(counts.keys())[0]}")

            team_a_color = list(counts.keys())[0]
            team_a_jerseys = [int(j) for j in teams[team_a_color]]

            # Build team_map for known team
            self.team_map = {}
            for j in teams[team_a_color]:
                self.team_map[str(j)] = team_a_color.capitalize()

            # Infer Team B color using common football kit pairings
            # This helps provide a meaningful team name instead of "TeamB"
            common_pairings = {
                "red": ["blue", "white", "cyan", "yellow"],
                "blue": ["red", "white", "yellow", "cyan"],
                "green": ["white", "red", "yellow"],
                "white": ["red", "blue", "green", "black"],
                "black": ["white", "red", "yellow"],
                "yellow": ["blue", "red", "black", "white"]
            }

            # Try to infer opposing team color
            team_a_lower = team_a_color.lower()
            if team_a_lower in common_pairings:
                # Use the most common pairing (first in list)
                team_b_color = common_pairings[team_a_lower][0].capitalize()
                print(f"[Team] Inferred opposing team color: {team_b_color} (common pairing with {team_a_color})")
            else:
                # Fallback to generic name
                team_b_color = "TeamB"
                print(f"[Team] Using generic team name: {team_b_color}")

            # Assign ALL Unknown players to Team B
            if "Unknown" in teams:
                print(f"[Team] Assigning {len(teams['Unknown'])} Unknown players to {team_b_color}")
                for unknown_jersey in teams["Unknown"]:
                    self.team_map[str(unknown_jersey)] = team_b_color

            # Assign Goalkeepers (single-team case)
            gk_jerseys = [j for j, cls_id in jersey_classes.items() if cls_id == 1]
            if gk_jerseys:
                print(f"[Team] Assigning {len(gk_jerseys)} goalkeeper(s) in single-team scenario")
                avg_a = sum(team_a_jerseys) / len(team_a_jerseys) if team_a_jerseys else 15
                for gk_jersey in gk_jerseys:
                    try:
                        jersey_num = int(gk_jersey)
                        # If GK jersey is close to Team A average, assign to Team A, else Team B
                        if abs(jersey_num - avg_a) < 10:
                            assigned_team = team_a_color.capitalize()
                        else:
                            assigned_team = team_b_color

                        self.team_map[str(gk_jersey)] = assigned_team
                        gk_color = id_manager.player_colors.get(gk_jersey, "Unknown")
                        print(f"[Team] Assigned GK #{gk_jersey} (color: {gk_color}) → {assigned_team}")
                    except (ValueError, TypeError):
                        print(f"[Team] Warning: Could not parse GK jersey number: {gk_jersey}")

            self.primary_teams = {team_a_color.capitalize(), team_b_color}
            print(f"[Team] Final teams: {team_a_color} ({len(team_a_jerseys)} players), {team_b_color} ({len(teams.get('Unknown', []))} players)")
        else:
            self.team_map = {}
