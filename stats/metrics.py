from collections import defaultdict
import math
from .event_logic import AdvancedEventDetector, EFF_FPS

class StatsEngine:
    def __init__(self, frame_width=None, frame_height=None):
        self.detector = AdvancedEventDetector(frame_width=frame_width, frame_height=frame_height)
        
    def _build_track_to_jersey(self, id_manager):
        """Build the canonical track_id -> jersey_number map.

        Combines locked bindings, the jersey registry, consistent-vote
        recoveries, and raw-read recoveries — the same resolution logic
        Phase 216 used inline. Extracted so it can run BEFORE the stats
        engine (Option A: resolve identities up front so each player is a
        single ID and events are never summed across fragments).
        """
        track_to_jersey = {}
        if not id_manager:
            return track_to_jersey

        if hasattr(id_manager, 'active_bindings'):
            track_to_jersey.update(id_manager.active_bindings)
        if hasattr(id_manager, 'jersey_registry'):
            for jersey_num, info in id_manager.jersey_registry.items():
                if isinstance(info, dict) and 'track_id' in info:
                    tid = info['track_id']
                    if tid not in track_to_jersey:
                        track_to_jersey[tid] = jersey_num

        # Consistent-vote recovery: fragments that read the right number but
        # never locked due to the global-uniqueness constraint.
        if hasattr(id_manager, 'vote_counts') and hasattr(id_manager, 'vote_tallies'):
            for tid, votes in id_manager.vote_counts.items():
                if tid in track_to_jersey or not votes:
                    continue
                best_num = max(votes, key=votes.get)
                best_tally = id_manager.vote_tallies.get(tid, {}).get(best_num, 0)
                best_score = votes[best_num]
                second_score = sorted(votes.values())[-2] if len(votes) > 1 else 0
                if best_tally >= 2 and (best_score - second_score) >= 0.3 and best_score >= 0.50:
                    track_to_jersey[tid] = best_num
                    if not hasattr(id_manager, 'vote_recovered_jerseys'):
                        id_manager.vote_recovered_jerseys = set()
                    id_manager.vote_recovered_jerseys.add(best_num)
        elif hasattr(id_manager, 'vote_counts'):
            for tid, votes in id_manager.vote_counts.items():
                if tid in track_to_jersey or not votes:
                    continue
                best_num = max(votes, key=votes.get)
                best_score = votes[best_num]
                second_score = sorted(votes.values())[-2] if len(votes) > 1 else 0
                if best_score >= 1.3 and (best_score - second_score) >= 0.5:
                    track_to_jersey[tid] = best_num
                    if not hasattr(id_manager, 'vote_recovered_jerseys'):
                        id_manager.vote_recovered_jerseys = set()
                    id_manager.vote_recovered_jerseys.add(best_num)

        # Raw-read recovery: low-confidence reads that never entered vote_counts.
        already_found = set(track_to_jersey.values())
        if hasattr(id_manager, 'raw_read_counts'):
            jersey_raw_counts = defaultdict(int)
            for tid, counts in id_manager.raw_read_counts.items():
                for jnum, cnt in counts.items():
                    jersey_raw_counts[jnum] += cnt
            for jnum, total_cnt in sorted(jersey_raw_counts.items(), key=lambda x: x[1], reverse=True):
                if jnum in already_found:
                    continue
                if total_cnt >= 20:
                    best_tid = max(
                        (tid for tid, counts in id_manager.raw_read_counts.items() if jnum in counts),
                        key=lambda t: id_manager.raw_read_counts[t].get(jnum, 0)
                    )
                    track_to_jersey[best_tid] = jnum
                    already_found.add(jnum)
                    if not hasattr(id_manager, 'vote_recovered_jerseys'):
                        id_manager.vote_recovered_jerseys = set()
                    id_manager.vote_recovered_jerseys.add(jnum)
                    print(f"[Phase 216++] Raw recovery: Jersey #{jnum} ({total_cnt} total raw reads)")

        return track_to_jersey

    def process_events(self, all_frames, id_manager=None, match_kits=None, siglip_teams=None):
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
        # Round 2 fix: Track which frames have real detections vs interpolated
        raw_ball_frames = set(ball_tracker.tracks.keys())

        balls_found = sum(1 for b in ball_track if b is not None)
        raw_count = len(raw_ball_frames)
        ball_pct = (balls_found / len(all_frames) * 100) if all_frames else 0
        print(f"[StatsEngine] Ball track: {balls_found}/{len(all_frames)} frames ({ball_pct:.1f}%) "
              f"[{raw_count} raw detections, {balls_found - raw_count} interpolated]")
        if ball_pct < 30:
            print(f"[StatsEngine] WARNING: Low ball detection rate ({ball_pct:.1f}%). "
                  f"Pass/shot stats will be unreliable. Check ball model quality.")

        if id_manager:
            # Build player_id -> dominant YOLO class from frame data
            # This is needed because id_manager.track_classes doesn't exist
            class_counts_per_id = defaultdict(lambda: defaultdict(int))
            for f in all_frames:
                for b in f["boxes"]:
                    pid = b.get("id")
                    cls = b.get("cls", 2)
                    if pid is not None:
                        class_counts_per_id[pid][cls] += 1

            # Determine dominant class per player ID
            player_dominant_classes = {}
            for pid, cls_counts in class_counts_per_id.items():
                dominant_cls = max(cls_counts.items(), key=lambda x: x[1])[0]
                player_dominant_classes[pid] = dominant_cls

            self._cluster_teams(id_manager, player_dominant_classes, match_kits=match_kits, siglip_teams=siglip_teams)

        # Option A: Resolve identities BEFORE the stats engine runs.
        # Remap every box's track ID to its final jersey number so each player
        # is a single ID for ownership/event detection. This eliminates the
        # fragment over-count at its source — events are detected once per
        # player instead of once per ByteTrack fragment, so Phase 216 no longer
        # has to sum (and over-sum) across overlapping fragments.
        _pre_resolved_jerseys = set()
        if id_manager:
            _ttj = self._build_track_to_jersey(id_manager)
            if _ttj:
                _remapped_boxes = 0
                for f in all_frames:
                    for b in f.get("boxes", []):
                        tid = b.get("id")
                        if tid is None:
                            continue
                        jnum = _ttj.get(tid)
                        if jnum is not None and jnum != tid:
                            b["id"] = jnum
                            _remapped_boxes += 1
                _pre_resolved_jerseys = set(_ttj.values())
                print(f"[Option A] Pre-resolved {len(_ttj)} track IDs to "
                      f"{len(_pre_resolved_jerseys)} jerseys "
                      f"({_remapped_boxes} box IDs remapped before stats)")

        # 1. Map Possession
        ownership = self.detector.calculate_ownership(player_tracks, ball_track)
        owned_frames = sum(1 for o in ownership if o is not None)
        own_pct = (owned_frames / len(ownership) * 100) if ownership else 0
        print(f"[StatsEngine] Ownership: {owned_frames}/{len(ownership)} frames ({own_pct:.1f}%) assigned to players")

        # 2. Detect Events & Get Stats
        team_map_ref = self.team_map if hasattr(self, "team_map") else None
        events, raw_stats = self.detector.analyze(ownership, player_tracks, ball_track,
                                                    team_map=team_map_ref,
                                                    raw_ball_frames=raw_ball_frames)
        
        # Phase 216: Remap raw_stats from track IDs to jersey numbers.
        # PICK PRIMARY TRACK per jersey (most frames) — do NOT sum across tracks.
        # Summing caused 10-22x stat inflation when multiple ByteTrack fragments
        # mapped to the same jersey.
        if id_manager:
            remapped_stats = {}
            remap_count = 0

            # Option A: box IDs were already remapped to jersey numbers before
            # the engine ran, so most stats come back keyed by jersey. The map
            # is rebuilt here only to catch any residual raw track IDs (e.g.
            # frames before a player's identity resolved). With Option A active,
            # each jersey typically has a single candidate and the merge below
            # no longer sums across overlapping fragments.
            track_to_jersey = self._build_track_to_jersey(id_manager)

            # Group tracks by target jersey number, keeping track of which
            # track has the most data (primary track)
            jersey_candidates = defaultdict(list)  # jersey_num -> [(track_id, stats_dict, weight)]
            unmapped = {}  # tracks with no jersey mapping

            # Set of jersey numbers known to id_manager — used to recognize
            # stats keys that are ALREADY jersey numbers (Option A pre-resolved
            # boxes produce jersey-keyed stats, so key == jersey, not a track ID).
            _known_jerseys = set(track_to_jersey.values()) | _pre_resolved_jerseys

            for track_id, stats_dict in raw_stats.items():
                jersey_num = track_to_jersey.get(track_id)
                # Option A: key is already a resolved jersey number
                if jersey_num is None and track_id in _known_jerseys:
                    jersey_num = track_id
                if jersey_num is not None:
                    weight = 0
                    for key in ("distance_m", "total_distance", "touch_frames", "ball_touches"):
                        if key in stats_dict:
                            weight += abs(stats_dict[key]) if isinstance(stats_dict[key], (int, float)) else 0
                    # Use (team, jersey) as key to support same jersey number on both teams.
                    # R18.2: team from the clustered team_map (the two discovered kits),
                    # not per-fragment raw color — raw colors include noise labels
                    # (Blue/Green) that aren't actual teams.
                    track_team = None
                    if team_map_ref:
                        track_team = team_map_ref.get(str(track_id)) or team_map_ref.get(str(jersey_num))
                    if not track_team or track_team == "Unknown":
                        track_team = stats_dict.get("team", id_manager.get_track_color(track_id) if id_manager else "Unknown")
                    candidate_key = (track_team, jersey_num)
                    jersey_candidates[candidate_key].append((track_id, stats_dict, weight))
                    remap_count += 1
                else:
                    unmapped[track_id] = stats_dict

            # R18.1: Consolidate same-jersey groups split by noisy per-fragment team
            # labels. Fragments of one player often carry different team labels
            # (Red/White/Unknown/None), which split the merge into separate small
            # groups — the unwrap step then kept only one group and silently dropped
            # the rest (e.g. #21 lost a 343-event group in favor of a 76-event one).
            # Pool all fragments per jersey under the weight-dominant real team.
            _by_jersey = defaultdict(list)
            for (team, jnum), cands in jersey_candidates.items():
                _by_jersey[jnum].append((team, cands))
            # R18.2: only the two clustered kit teams are valid labels — raw color
            # noise (Blue/Green on a Red/White match) must not win the team vote
            _valid_teams = set(team_map_ref.values()) - {"Unknown", None} if team_map_ref else None
            _consolidated = {}
            for jnum, groups in _by_jersey.items():
                team_weights = defaultdict(float)
                all_cands = []
                for team, cands in groups:
                    all_cands.extend(cands)
                    if team and team != "Unknown":
                        if _valid_teams and team not in _valid_teams:
                            continue
                        team_weights[team] += sum(c[2] for c in cands)
                best_team = max(team_weights, key=team_weights.get) if team_weights else "Unknown"
                if len(groups) > 1:
                    print(f"[Phase 216] Jersey #{jnum}: consolidated {len(groups)} team-label groups "
                          f"({len(all_cands)} fragments) under team '{best_team}'")
                _consolidated[(best_team, jnum)] = all_cands
            jersey_candidates = _consolidated

            # For each jersey, MERGE stats from all fragments:
            # - Event stats (discrete occurrences): SUM across all fragments
            # - Distance/touch: also SUM — ByteTrack fragments cover non-overlapping
            #   time periods so distance is additive (unlike events which could double-count
            #   if same event appears in multiple fragments, distance physically cannot)
            _EVENT_KEYS = {
                "distance_m", "touch_frames",  # accumulative but additive across non-overlapping fragments
                "tackles", "tackles_successful", "shots_on_target",
                "dribbles", "dribbles_successful",
                "passes_total", "passes_complete",
                "crosses_total", "crosses_complete",
                "interceptions", "ball_interceptions_total",
                "ball_recoveries_opp_half", "ball_recoveries_own_half",
                "challenges_total", "challenges_won_total",
                "goals", "goals_total", "goals_conceded",
                "shots_saved_total",
                "close_range_shots", "mid_range_shots", "long_range_shots",
                "short_passes", "medium_passes", "long_passes",
                "short_passes_accurate", "medium_passes_accurate", "long_passes_accurate",
                "close_range_saves", "mid_range_saves", "long_range_saves",
                "jumping_saves", "penalties_saved", "freekick_saved", "corners_saved",
                "in_box_touches",
                "xg_foot_no_opponent", "xg_foot_opponent_present", "expected_assists",
                "fouls_total",
            }

            # Round 13: Top-N merge — sum events from top N fragments by weight.
            # R18: Raised from 5 to 20 — this video has ~67 fragments per player vs ~10
            # on Hamburg/Bayern, so top-5 was only capturing ~7% of events per player.
            # High-risk events (goals, tackles) are still capped after merge.
            _MERGE_TOP_N = 20  # Sum events from primary + top 19 fragments
            for candidate_key, candidates in jersey_candidates.items():
                jersey_num = candidate_key[1] if isinstance(candidate_key, tuple) else candidate_key
                if len(candidates) == 1:
                    _, stats_dict, _ = candidates[0]
                    remapped_stats[candidate_key] = stats_dict
                else:
                    # Sort by weight (most data first = primary)
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    primary_tid, primary_stats, primary_w = candidates[0]

                    # Take top N fragments for event merging
                    merge_candidates = candidates[:_MERGE_TOP_N]
                    merged = defaultdict(int)
                    merged.update(primary_stats)
                    recovered_events = 0
                    for tid, stats_dict, w in merge_candidates[1:]:
                        for key in _EVENT_KEYS:
                            if key in stats_dict:
                                val = stats_dict[key]
                                if isinstance(val, (int, float)) and val > 0:
                                    merged[key] = merged.get(key, 0) + val
                                    recovered_events += 1
                    # R20: Goal rescue with dedup — scan remaining fragments for goals,
                    # but only rescue if top-N merge found 0 goals for this player.
                    # Cap at 2 goals per player max (hat-tricks are rare edge cases).
                    _CRITICAL_EVENT_KEYS = {"goals", "goals_total"}
                    existing_goals = max(merged.get("goals", 0), merged.get("goals_total", 0))
                    if existing_goals == 0:
                        # No goals in top-N — check remaining fragments
                        remaining = candidates[_MERGE_TOP_N:]
                        has_rescued = False
                        for tid, stats_dict, w in remaining:
                            if has_rescued:
                                break
                            for key in _CRITICAL_EVENT_KEYS:
                                if key in stats_dict:
                                    val = stats_dict[key]
                                    if isinstance(val, (int, float)) and val > 0:
                                        merged[key] = merged.get(key, 0) + min(val, 1)
                                        has_rescued = True
                        if has_rescued:
                            print(f"[Phase 216] Jersey #{jersey_num}: RESCUED 1 goal from minor fragments")
                    # Cap goals per player at 2 (covers most real scenarios)
                    _MAX_GOALS_PER_PLAYER = 2
                    for key in _CRITICAL_EVENT_KEYS:
                        if merged.get(key, 0) > _MAX_GOALS_PER_PLAYER:
                            print(f"[Phase 216] Jersey #{jersey_num}: capped {key} from {merged[key]} to {_MAX_GOALS_PER_PLAYER}")
                            merged[key] = _MAX_GOALS_PER_PLAYER

                    remapped_stats[candidate_key] = merged
                    extra = len(merge_candidates) - 1
                    skipped = len(candidates) - len(merge_candidates)
                    print(f"[Phase 216] Jersey #{jersey_num}: top-{len(merge_candidates)} merged "
                          f"(primary={primary_tid}, recovered {recovered_events} events from {extra} fragment(s), "
                          f"{skipped} minor fragments skipped)")

            # Far-outlier guard, scaled to observed match duration.
            # With Option A (identities resolved before stats), events are now
            # counted once per player, so real per-player variation is valid and
            # must NOT be flattened. These ceilings are deliberately ~2.5x the
            # realistic top-of-range so they fire ONLY on clearly-broken values
            # (e.g. a fragment-leak explosion of 40+ tackles), never on a genuine
            # busy player. A previous tighter version (tackles 6/90 -> cap 2 for a
            # 30-min clip) was clamping everyone to identical values and hiding
            # the real spread.
            _RATE_CAPS_PER_90 = {
                "tackles": 18, "tackles_successful": 18,
                "interceptions": 25, "ball_interceptions_total": 25,
                "dribbles": 30, "dribbles_successful": 24,
                "challenges_total": 45, "challenges_won_total": 35,
                "shots_on_target": 14,
                "passes_total": 160, "passes_complete": 150,
                "crosses_total": 25, "crosses_complete": 18,
                "distance_m": 14500.0,
            }
            match_minutes = (len(all_frames) / EFF_FPS) / 60.0 if all_frames else 90.0
            _dur_ratio = max(0.05, match_minutes / 90.0)
            for ck, merged_stats in remapped_stats.items():
                jnum_log = ck[1] if isinstance(ck, tuple) else ck
                for key, per90 in _RATE_CAPS_PER_90.items():
                    cap = per90 * _dur_ratio if key == "distance_m" else max(1, round(per90 * _dur_ratio))
                    val = merged_stats.get(key, 0)
                    if isinstance(val, (int, float)) and val > cap:
                        print(f"[Phase 216] Jersey #{jnum_log}: capped {key} {val} -> {cap} "
                              f"(rate cap, {match_minutes:.0f} min)")
                        merged_stats[key] = cap

            # Keep unmapped tracks as-is
            for track_id, stats_dict in unmapped.items():
                remapped_stats[track_id] = stats_dict

            raw_stats = remapped_stats
            print(f"[Phase 216] Remapped {remap_count} track IDs to jersey numbers (pick-primary, no summing)")
        
        # 3. Final Formatting
        formatted_stats = {}
        
        # Get all distinct IDs from stats + jersey registry.
        # Phase 216 remap produces tuple keys (team, jersey). Unwrap them so
        # the formatting loop can look up stats correctly.
        tuple_stats = {k: v for k, v in raw_stats.items() if isinstance(k, tuple)}
        plain_stats = {k: v for k, v in raw_stats.items() if not isinstance(k, tuple)}
        # R18.1: Jersey-mapped stats always win over an unmapped raw track that
        # happens to share the same numeric ID (e.g. unmapped track 21 colliding
        # with jersey #21). Previously the unmapped track won and the jersey's
        # entire merged stats were silently dropped.
        for (team, jnum), v in tuple_stats.items():
            v["team"] = team
            plain_stats[jnum] = v
        raw_stats = plain_stats

        all_ids = set(raw_stats.keys())
        if id_manager:
            all_ids.update(id_manager.jersey_registry.keys())

        # Pre-calculate Minutes Played for Noise Filtering (Task 3)
        id_frame_counts = defaultdict(int)
        for f in all_frames:
             for b in f["boxes"]:
                 if b["id"] is not None:
                     id_frame_counts[b["id"]] += 1

        # Phase 216 Fix: Remap frame counts to jersey numbers (pick max, not sum)
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

            # Pick the max frame count across tracks for each jersey (not sum)
            jersey_max_frames = defaultdict(int)
            for tid, count in list(id_frame_counts.items()):
                if tid in track_to_jersey:
                    jnum = track_to_jersey[tid]
                    jersey_max_frames[jnum] = max(jersey_max_frames[jnum], count)
            for jnum, max_count in jersey_max_frames.items():
                id_frame_counts[jnum] = max_count


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
            
            minutes_played = (total_frames / EFF_FPS) / 60.0
            seconds_played = (total_frames / EFF_FPS)
            
            # Strict Filter: < 1.5 Seconds -> DELETE
            # Exception: vote-recovered jerseys are exempt — they are real players
            # with limited camera time recovered via Phase 216/216++
            vote_recovered_jerseys = getattr(id_manager, 'vote_recovered_jerseys', set())
            goals_detected = raw_stats.get(id_key, {}).get("goals", 0)
            is_vote_recovered = id_key in vote_recovered_jerseys or (isinstance(id_key, str) and id_key.isdigit() and int(id_key) in vote_recovered_jerseys)
            if not is_vote_recovered and seconds_played < 1.5 and goals_detected == 0:
                continue

            # Round 16: Secondary ghost filter — moderate presence but zero activity
            # Catches detection artifacts that persist 2-24s but contribute nothing
            # Exception: vote-recovered jerseys are real players, not ghosts
            if not is_vote_recovered and seconds_played > 2.0 and total_frames < 200 and goals_detected == 0:
                s_check = raw_stats.get(id_key, defaultdict(int))
                has_activity = (
                    s_check.get("passes_total", 0) > 0 or
                    s_check.get("tackles", 0) > 0 or
                    s_check.get("shots_on_target", 0) > 0 or
                    s_check.get("touch_frames", 0) > 0 or
                    s_check.get("dribbles", 0) > 0 or
                    s_check.get("interceptions", 0) > 0
                )
                dominant_cls = s_check.get("dominant_class", 2)
                if not has_activity and dominant_cls != 1:
                    continue

            # Identify Player
            is_known = False

            # Vote-recovered jerseys are always treated as known real players
            if is_vote_recovered:
                is_known = True

            # Check 1: Is it a valid Jersey Number in Registry?
            if not is_known and id_manager and id_manager.is_jersey_number(id_key):
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

                # Use team stored on stat dict from Phase 216 remap first
                team_from_stats = raw_stats.get(id_key, {}).get("team") if isinstance(raw_stats.get(id_key), dict) else None
                # Use Clustered Team Map if available
                if hasattr(self, "team_map") and final_key in self.team_map:
                    team_name = self.team_map[final_key]
                elif team_from_stats and team_from_stats not in ("Unknown", None):
                    team_name = team_from_stats
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
            time_on_ball = round(s["touch_frames"] / EFF_FPS, 2)
            
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
                    # FIX: Wire foot_passes_open_play to actual pass counts (all detected passes are open play)
                    "foot_passes_open_play_total": p_total,
                    "hand_passes_total": 0, # Model Gap - requires hand contact detection
                    "short_passes_total": s.get("short_passes", 0),
                    "medium_passes_total": s.get("medium_passes", 0),
                    "long_passes_total": s.get("long_passes", 0),
                    "accurate_foot_passes_open_play_total": p_comp,
                    "hand_passes_accurate_total": 0,
                    "short_passes_accurate_total": s.get("short_passes_accurate", 0),
                    "medium_passes_accurate_total": s.get("medium_passes_accurate", 0),
                    # FIX: Key mismatch - event_logic uses "long_passes_accurate" not "accurate_long_passes"
                    "accurate_long_passes_total": s.get("long_passes_accurate", 0),    

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

        # Round 15: Ghost filter removed — was deleting White team players
        # and worsening team imbalance (14G/5W). Keep all players in output.

        return formatted_stats, events

    def _cluster_teams(self, id_manager, player_dominant_classes=None, match_kits=None, siglip_teams=None):
        """
        Force-assign every player to Team A or Team B based on Jersey Color.
        Merges similar colors and assigns Unknown players to nearest team.

        player_dominant_classes: dict {player_id -> dominant YOLO class} built from all_frames
        """
        # Build jersey -> cls_id mapping to filter goalkeepers
        # Uses dominant class from actual frame data (not id_manager.track_classes which doesn't exist)
        jersey_classes = {}
        if player_dominant_classes:
            for pid, cls_id in player_dominant_classes.items():
                jersey_classes[pid] = cls_id
            gk_count = sum(1 for c in jersey_classes.values() if c == 1)
            print(f"[Team] Found {gk_count} goalkeeper(s) from YOLO class detection: "
                  f"{[pid for pid, c in jersey_classes.items() if c == 1]}")

        # === SigLIP-based team clustering (Round 7) ===
        # If SigLIP visual clusters are available, use them as the PRIMARY team source.
        # This replaces HSV color-based clustering which fails under lighting variation.
        if siglip_teams and len(siglip_teams) >= 4:
            print(f"[Team] Using SigLIP visual clustering ({len(siglip_teams)} tracks)")

            # 1. Map track_ids → jersey numbers, group by cluster label
            cluster_jerseys = defaultdict(list)  # {0: [jersey1, jersey2], 1: [jersey3, ...]}
            for track_id, cluster_label in siglip_teams.items():
                jersey_num = id_manager.active_bindings.get(track_id)
                if jersey_num is None:
                    continue
                # Skip goalkeepers
                if jersey_classes.get(jersey_num) == 1 or jersey_classes.get(str(jersey_num)) == 1:
                    continue
                cluster_jerseys[cluster_label].append(str(jersey_num))

            if len(cluster_jerseys) >= 2:
                # 2. Name each cluster by the dominant HSV color of its members
                cluster_colors = {}
                for label, jerseys in cluster_jerseys.items():
                    color_counts = defaultdict(int)
                    for j in jerseys:
                        color = id_manager.player_colors.get(j, "Unknown")
                        if color != "Unknown":
                            color_counts[color] += 1
                    if color_counts:
                        cluster_colors[label] = max(color_counts, key=color_counts.get)
                    else:
                        cluster_colors[label] = f"Team{label}"

                # If both clusters got the same color name, disambiguate
                labels = sorted(cluster_jerseys.keys())
                team_a_label, team_b_label = labels[0], labels[1]
                team_a_color = cluster_colors.get(team_a_label, "TeamA")
                team_b_color = cluster_colors.get(team_b_label, "TeamB")
                if team_a_color == team_b_color:
                    team_a_color = f"{team_a_color}_A"
                    team_b_color = f"{team_b_color}_B"

                print(f"[Team] SigLIP clusters: {team_a_color} ({len(cluster_jerseys[team_a_label])} players), "
                      f"{team_b_color} ({len(cluster_jerseys[team_b_label])} players)")

                # 3. Build team_map
                self.team_map = {}
                team_a_jerseys = [int(j) for j in cluster_jerseys[team_a_label]]
                team_b_jerseys = [int(j) for j in cluster_jerseys[team_b_label]]
                for j in cluster_jerseys[team_a_label]:
                    self.team_map[str(j)] = team_a_color.capitalize()
                for j in cluster_jerseys[team_b_label]:
                    self.team_map[str(j)] = team_b_color.capitalize()

                # 4. Assign players NOT in SigLIP (unlocked tracks, GKs) via jersey proximity
                avg_a = sum(team_a_jerseys) / len(team_a_jerseys) if team_a_jerseys else 15
                avg_b = sum(team_b_jerseys) / len(team_b_jerseys) if team_b_jerseys else 25

                # Assign GKs
                for jersey_str, cls_id in jersey_classes.items():
                    if cls_id == 1 and str(jersey_str) not in self.team_map:
                        try:
                            jnum = int(jersey_str)
                            dist_a = abs(jnum - avg_a)
                            dist_b = abs(jnum - avg_b)
                            assigned = team_a_color if dist_a < dist_b else team_b_color
                            self.team_map[str(jersey_str)] = assigned.capitalize()
                            print(f"[Team] SigLIP: GK #{jersey_str} → {assigned} (jersey proximity)")
                        except (ValueError, TypeError):
                            pass

                # Assign remaining jerseys from player_colors not in SigLIP
                for jersey, color in id_manager.player_colors.items():
                    if str(jersey) not in self.team_map:
                        try:
                            jnum = int(jersey)
                            dist_a = abs(jnum - avg_a)
                            dist_b = abs(jnum - avg_b)
                            assigned = team_a_color if dist_a < dist_b else team_b_color
                            self.team_map[str(jersey)] = assigned.capitalize()
                            print(f"[Team] SigLIP: Unmatched #{jersey} → {assigned} (jersey proximity)")
                        except (ValueError, TypeError):
                            pass

                # Add track_id mappings for unbound tracks
                if hasattr(id_manager, 'active_bindings'):
                    for track_id, jersey_num in id_manager.active_bindings.items():
                        jersey_team = self.team_map.get(str(jersey_num))
                        if jersey_team and str(track_id) not in self.team_map:
                            self.team_map[str(track_id)] = jersey_team

                self.primary_teams = {team_a_color.capitalize(), team_b_color.capitalize()}
                print(f"[Team] SigLIP team_map: {len(self.team_map)} entries")
                return  # Skip HSV-based clustering entirely
            else:
                print(f"[Team] SigLIP: Only {len(cluster_jerseys)} clusters found, falling back to HSV")

        # === HSV-based clustering (fallback) ===
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
            "cyan": "blue",
            "maroon": "red",
            "pink": "red",
            "orange": "red", # Often GK or vibrant kit
            "silver": "white",
            "gray": "white",
            "gold": "yellow",
            "purple": "blue" # Mapping purple to blue bucket for now
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
            # --- Round 6: Kit-guided team selection ---
            # If KitCoordinator discovered exactly 2 player kit colors, prefer those
            # over naive count-based top-2 selection.  This prevents scenarios like
            # V3 where Green(6) is orphaned when Red(21)+Blue(6) are picked as top-2,
            # even though the real teams are Red and Green.
            kit_guided = False
            if match_kits and len(match_kits.get("players", [])) == 2:
                kit_a_raw, kit_b_raw = match_kits["players"]
                kit_a_norm = color_merge_map.get(kit_a_raw.lower(), kit_a_raw.lower()).capitalize()
                kit_b_norm = color_merge_map.get(kit_b_raw.lower(), kit_b_raw.lower()).capitalize()
                if kit_a_norm in counts and kit_b_norm in counts:
                    team_a_color = kit_a_norm
                    team_b_color = kit_b_norm
                    kit_guided = True
                    print(f"[Team] Kit-guided selection: {team_a_color} ({counts[team_a_color]}), "
                          f"{team_b_color} ({counts[team_b_color]}) "
                          f"(from match_kits: {kit_a_raw}/{kit_b_raw})")
                elif kit_a_norm in counts or kit_b_norm in counts:
                    # Round 18: One kit color present, other missing from player_colors.
                    # This happens when dark jerseys (Black/Navy) get misclassified as
                    # Yellow/Green in individual tracks but KitCoordinator correctly
                    # identifies the true kit color from aggregate observations.
                    # Trust the kit discovery: present color is one team, all other
                    # players belong to the missing kit color team.
                    present = kit_a_norm if kit_a_norm in counts else kit_b_norm
                    missing = kit_b_norm if present == kit_a_norm else kit_a_norm
                    team_a_color = present
                    team_b_color = missing
                    # Create empty entry for the missing color so downstream code works
                    teams[missing] = []
                    kit_guided = True
                    print(f"[Team] Kit-guided selection (partial): {present} ({counts[present]}), "
                          f"{missing} (0 locked, from match_kits: {kit_a_raw}/{kit_b_raw})")

            if not kit_guided:
                top_2 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]
                team_a_color = top_2[0][0]
                team_b_color = top_2[1][0]
                print(f"[Team] Count-based selection: {team_a_color} ({counts[team_a_color]}), "
                      f"{team_b_color} ({counts[team_b_color]})")
            
            # Update Identity Manager with EXPLICIT Team Names (The Color Itself)
            self.team_map = {}
            team_a_jerseys = [int(j) for j in teams[team_a_color]]
            team_b_jerseys = [int(j) for j in teams[team_b_color]]

            for j in teams[team_a_color]: self.team_map[str(j)] = team_a_color.capitalize() # "Red"
            for j in teams[team_b_color]: self.team_map[str(j)] = team_b_color.capitalize() # "White"

            # Pre-compute team jersey averages (needed by FIX #4 tie-break and FIX #2/#3)
            avg_a = sum(team_a_jerseys) / len(team_a_jerseys) if team_a_jerseys else 15
            avg_b = sum(team_b_jerseys) / len(team_b_jerseys) if team_b_jerseys else 25

            # === FIX #4: Merge orphan color groups into nearest primary team ===
            # After picking top-2, any remaining color groups (3rd, 4th, etc.) are
            # force-assigned to the nearest primary team using HSV hue distance.
            # This prevents stray GK kit colors or misclassified shades from
            # appearing as separate teams in the final output.
            _HSV_HUE_REF = {
                "red": 0, "orange": 15, "yellow": 30, "gold": 30,
                "green": 60, "lime": 60, "teal": 75, "cyan": 90,
                "blue": 120, "navy": 120, "purple": 140, "magenta": 155,
                "pink": 170, "maroon": 0,
                "white": -1, "black": -1, "gray": -1, "silver": -1,
            }

            def _hue_dist(c1, c2):
                """Circular hue distance (0-90 scale). Achromatic colors get max distance."""
                h1 = _HSV_HUE_REF.get(c1.lower(), -1)
                h2 = _HSV_HUE_REF.get(c2.lower(), -1)
                if h1 < 0 or h2 < 0:
                    return 180  # achromatic → max distance, let jersey-number fallback decide
                return min(abs(h1 - h2), 180 - abs(h1 - h2))

            # Detect if either team is achromatic (White/Black/Gray)
            _team_a_achromatic = _HSV_HUE_REF.get(team_a_color.lower(), -1) < 0
            _team_b_achromatic = _HSV_HUE_REF.get(team_b_color.lower(), -1) < 0

            for orphan_color, orphan_jerseys in teams.items():
                if orphan_color in (team_a_color, team_b_color, "Unknown"):
                    continue
                orphan_hue = _HSV_HUE_REF.get(orphan_color.lower(), -1)

                # Special case: one team is achromatic (White/Black).
                # Hue distance to achromatic is always 180 (max), so orphans
                # would never be assigned there. But green-tinted white jerseys
                # get classified as Green/Blue — they should go to the White team.
                # Rule: if orphan is NOT close to the chromatic team (>30°), assign
                # to the achromatic team (it's likely a misclassified white/black).
                if _team_a_achromatic and not _team_b_achromatic:
                    # Team A is achromatic (e.g. White), Team B is chromatic (e.g. Red)
                    dist_to_chromatic = _hue_dist(orphan_color, team_b_color)
                    if orphan_hue >= 0 and dist_to_chromatic > 30:
                        nearest = team_a_color  # Assign to White
                    else:
                        nearest = team_b_color  # Close to chromatic team
                elif _team_b_achromatic and not _team_a_achromatic:
                    # Team B is achromatic, Team A is chromatic
                    dist_to_chromatic = _hue_dist(orphan_color, team_a_color)
                    if orphan_hue >= 0 and dist_to_chromatic > 30:
                        nearest = team_b_color  # Assign to White/Black
                    else:
                        nearest = team_a_color  # Close to chromatic team
                else:
                    # Both teams achromatic (White + Black) or both chromatic
                    dist_a = _hue_dist(orphan_color, team_a_color)
                    dist_b = _hue_dist(orphan_color, team_b_color)
                    if dist_a == dist_b:
                        # Balance-aware tie-break — assign to the SMALLER team
                        size_a = len(team_a_jerseys)
                        size_b = len(team_b_jerseys)
                        nearest = team_b_color if size_a > size_b else team_a_color
                    else:
                        nearest = team_a_color if dist_a < dist_b else team_b_color
                for j in orphan_jerseys:
                    self.team_map[str(j)] = nearest.capitalize()
                print(f"[Team] FIX#4: Merged orphan color '{orphan_color}' ({len(orphan_jerseys)} players) → {nearest}")

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
                        # Map color to team — also check orphan colors via merge
                        color_lower = color.lower() if color else ""
                        merged_color = color_merge_map.get(color_lower, color_lower)
                        if merged_color in [team_a_color.lower()]:
                            self.team_map[str(track_id)] = team_a_color.capitalize()
                        elif merged_color in [team_b_color.lower()]:
                            self.team_map[str(track_id)] = team_b_color.capitalize()
                        else:
                            # Orphan track color — use achromatic-aware hue distance
                            orphan_h = _HSV_HUE_REF.get(color_lower, -1)
                            if _team_a_achromatic and not _team_b_achromatic:
                                d_chrom = _hue_dist(color_lower, team_b_color)
                                if orphan_h >= 0 and d_chrom > 30:
                                    self.team_map[str(track_id)] = team_a_color.capitalize()
                                else:
                                    self.team_map[str(track_id)] = team_b_color.capitalize()
                            elif _team_b_achromatic and not _team_a_achromatic:
                                d_chrom = _hue_dist(color_lower, team_a_color)
                                if orphan_h >= 0 and d_chrom > 30:
                                    self.team_map[str(track_id)] = team_b_color.capitalize()
                                else:
                                    self.team_map[str(track_id)] = team_a_color.capitalize()
                            else:
                                d_a = _hue_dist(color_lower, team_a_color)
                                d_b = _hue_dist(color_lower, team_b_color)
                                if d_a <= d_b:
                                    self.team_map[str(track_id)] = team_a_color.capitalize()
                                else:
                                    self.team_map[str(track_id)] = team_b_color.capitalize()

            print(f"[ReID Fix] team_map now has {len(self.team_map)} entries (jersey + track IDs)")

            # Round 15: Team rebalancing when ratio > 2:1
            # MPS color classifier misclassifies White as Green, causing 14G/5W.
            # Move lowest-observation players from larger team to smaller team
            # until ratio is <= 1.5:1 or teams are equal.
            size_a = len(team_a_jerseys)
            size_b = len(team_b_jerseys)
            if size_a > 0 and size_b > 0:
                ratio = max(size_a, size_b) / min(size_a, size_b)
                if ratio > 2.0:
                    print(f"[Team] WARNING: Team size imbalance detected! "
                          f"{team_a_color}={size_a}, {team_b_color}={size_b} (ratio {ratio:.1f}:1). "
                          f"Attempting rebalance...")

                    # Identify which team is larger
                    if size_a > size_b:
                        big_color, small_color = team_a_color, team_b_color
                        big_jerseys, small_jerseys = team_a_jerseys, team_b_jerseys
                    else:
                        big_color, small_color = team_b_color, team_a_color
                        big_jerseys, small_jerseys = team_b_jerseys, team_a_jerseys

                    # Get observation counts for players in the bigger team
                    # Players with fewest observations are most likely misclassified
                    jersey_obs = {}
                    if hasattr(id_manager, 'jersey_registry'):
                        for j in big_jerseys:
                            reg = id_manager.jersey_registry.get(j, {})
                            jersey_obs[j] = reg.get("observations", 0) if isinstance(reg, dict) else 0
                    if not jersey_obs:
                        # Fallback: use player_colors observation counts
                        for j in big_jerseys:
                            jersey_obs[j] = 0

                    # Sort by observations (fewest first = most likely misclassified)
                    sorted_big = sorted(big_jerseys, key=lambda j: jersey_obs.get(j, 0))

                    # Move players until ratio <= 1.5:1 or equal
                    moved = []
                    while len(sorted_big) > len(small_jerseys) and len(sorted_big) - len(small_jerseys) > 2:
                        # Don't move GKs (jersey #1 or dominant_class == 1)
                        candidate = sorted_big[0]
                        if candidate == 1 or jersey_classes.get(candidate, 2) == 1:
                            sorted_big = sorted_big[1:]
                            continue
                        # Move candidate to smaller team
                        sorted_big.pop(0)
                        small_jerseys.append(candidate)
                        self.team_map[str(candidate)] = small_color.capitalize()
                        moved.append(candidate)
                        # Check if balanced enough
                        new_ratio = max(len(sorted_big), len(small_jerseys)) / max(1, min(len(sorted_big), len(small_jerseys)))
                        if new_ratio <= 1.5:
                            break

                    if moved:
                        print(f"[Team] Rebalanced: moved jerseys {moved} from {big_color} to {small_color}")
                        print(f"[Team] New sizes: {big_color}={len(sorted_big)}, {small_color}={len(small_jerseys)}")
                        # Update the jersey lists
                        if size_a > size_b:
                            team_a_jerseys = sorted_big
                            team_b_jerseys = small_jerseys
                        else:
                            team_b_jerseys = sorted_big
                            team_a_jerseys = small_jerseys

            # Helper for the rest
            self.primary_teams = {team_a_color.capitalize(), team_b_color.capitalize()}

            # === FIX #2: Force-assign Unknown players to nearest team ===
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
            # Only include GKs that have jersey numbers (in player_colors), not raw track IDs
            gk_jerseys = [j for j, cls_id in jersey_classes.items()
                          if cls_id == 1 and j in id_manager.player_colors]
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
            gk_jerseys = [j for j, cls_id in jersey_classes.items()
                          if cls_id == 1 and j in id_manager.player_colors]
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
