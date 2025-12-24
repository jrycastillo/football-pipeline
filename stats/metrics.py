import math
import numpy as np

class StatsEngine:
    def __init__(self):
        self.stats = {}  # Key: Player ID, Value: Dict of stats

    def initialize_player(self, player_id):
        if player_id not in self.stats:
            self.stats[player_id] = {
                # --- FIELD PLAYER: OFFENSIVE (TARGET) ---
                "goals_total": 0,
                "shots_on_target_total": 0,
                "shots_wide_total": 0,
                "penalty_total": 0,

                # --- FIELD PLAYER: OFFENSIVE (DELIVERING) ---
                "crosses_total": 0,
                "crosses_accurate_total": 0,
                "dribbles_total": 0,
                "dribbles_successful_total": 0,

                # --- FIELD PLAYER: INTERACTION ---
                "passes_total": 0,
                "passes_accurate": 0,
                # Gray Cell: "accurate_passes_percent" (Calculated at end)

                # --- FIELD PLAYER: DEFENSIVE ---
                "challenges_total": 0,
                "challenges_won_total": 0,
                "tackles_total": 0,
                "tackles_successful_total": 0,
                "ball_interceptions_total": 0,
                "fouls_total": 0,

                # --- FIELD PLAYER: xG STATS (4-WAY SPLIT) ---
                "xg_foot_no_opponent": 0.0,
                "xg_header_no_opponent": 0.0,
                "xg_foot_opponent_present": 0.0,
                "xg_header_opponent_present": 0.0,

                # --- GOALKEEPER: SHOTS SAVED ---
                "shots_saved_total": 0,
                "close_range_saved": 0,  # < 5m
                "mid_range_saved": 0,    # 5-16m
                "long_range_saved": 0,   # > 17m

                # --- GOALKEEPER: SAVE TYPES ---
                "jumping_saves_total": 0,
                "saves_without_jumping_total": 0,
                "penalties_saved": 0,
                "freekick_saved": 0,
                "corners_saved": 0,

                # --- GOALKEEPER: PASSING ---
                # (Uses standard passing fields above)
                "accurate_long_passes_total": 0, # > 30m accurate
                
                # --- GENERAL ---
                "total_distance": 0.0, # Added for compatibility with current pipeline
            }

    def update(self, player_id, event_type, context):
        """
        player_id: ID of the player performing the action.
        event_type: 'Shot', 'Pass', 'Dribble', 'Tackle', 'Save', 'Foul', 'Duel', 'Interception'
        context: Dict containing details like {'result': 'Goal', 'distance': 12, 'body_part': 'head', 'pressure': True}
        """
        self.initialize_player(player_id)
        s = self.stats[player_id]
        
        # --- 1. SHOOTING & xG ---
        if event_type == 'Shot':
            # Basic Stats
            if context.get('result') == 'Goal':
                s['goals_total'] += 1
                s['shots_on_target_total'] += 1
            elif context.get('result') == 'Saved':
                s['shots_on_target_total'] += 1
            elif context.get('result') == 'Miss':
                s['shots_wide_total'] += 1
            
            if context.get('phase') == 'Penalty':
                s['penalty_total'] += 1

            # 4-Way xG Logic
            xg_val = context.get('xg_value', 0.0)
            is_header = context.get('body_part') == 'head'
            has_pressure = context.get('pressure', False)

            if not has_pressure and not is_header:
                s['xg_foot_no_opponent'] += xg_val
            elif not has_pressure and is_header:
                s['xg_header_no_opponent'] += xg_val
            elif has_pressure and not is_header:
                s['xg_foot_opponent_present'] += xg_val
            elif has_pressure and is_header:
                s['xg_header_opponent_present'] += xg_val

        # --- 2. PASSING & CROSSING ---
        elif event_type == 'Pass':
            s['passes_total'] += 1
            is_accurate = context.get('result') == 'Complete'
            
            if is_accurate:
                s['passes_accurate'] += 1
                if context.get('distance', 0) > 30: # GK Long Pass Logic
                    s['accurate_long_passes_total'] += 1

            if context.get('subtype') == 'Cross':
                s['crosses_total'] += 1
                if is_accurate:
                    s['crosses_accurate_total'] += 1

        # --- 3. DRIBBLING ---
        elif event_type == 'Dribble':
            s['dribbles_total'] += 1
            if context.get('result') == 'Complete':
                s['dribbles_successful_total'] += 1

        # --- 4. DEFENSIVE ---
        elif event_type == 'Tackle':
            s['tackles_total'] += 1
            if context.get('result') == 'Won':
                s['tackles_successful_total'] += 1
        elif event_type == 'Duel':
            s['challenges_total'] += 1
            if context.get('result') == 'Won':
                s['challenges_won_total'] += 1
        elif event_type == 'Interception':
            s['ball_interceptions_total'] += 1
        elif event_type == 'Foul':
            s['fouls_total'] += 1

        # --- 5. GOALKEEPER SAVES ---
        elif event_type == 'Save':
            s['shots_saved_total'] += 1
            dist = context.get('shot_distance', 0)
            
            # Distance Logic
            if dist < 5:
                s['close_range_saved'] += 1
            elif 5 <= dist < 17:
                s['mid_range_saved'] += 1
            else:
                s['long_range_saved'] += 1

            # Type Logic
            if context.get('is_jumping'):
                s['jumping_saves_total'] += 1
            else:
                s['saves_without_jumping_total'] += 1

            # Phase Logic
            phase = context.get('phase')
            if phase == 'Penalty': s['penalties_saved'] += 1
            elif phase == 'Freekick': s['freekick_saved'] += 1
            elif phase == 'Corner': s['corners_saved'] += 1

    def finalize_stats(self):
        """Calculates Gray Cells (Percentages) at the end of the match."""
        for pid, s in self.stats.items():
            # Accurate Passes %
            if s['passes_total'] > 0:
                s['accurate_passes_percent'] = round((s['passes_accurate'] / s['passes_total']) * 100, 1)
            else:
                s['accurate_passes_percent'] = 0.0

            # (Add other derived stats here as needed, e.g., Tackle Success %)
        return self.stats
