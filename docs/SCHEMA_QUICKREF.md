# Schema — Quick Reference

Simple field list. Full version with notes: `AI_OUTPUT_SCHEMA.md`.

## Files
| File | Shape |
|---|---|
| `player_stats.json` | `{ "<jersey>": PlayerEntry }` |
| `raw_tracks.json` | `[ Event, … ]` |
| `clips_manifest.json` | `[ Clip, … ]` |
| `track_jersey.json` | `{ "<track_id>": {jersey, team} }` |

---

## PlayerEntry
```jsonc
{
  "player_name":         string,
  "jersey_number":       int,
  "team":                string,   // colour
  "position":            string,
  "role":                string,   // Player | Goalkeeper | Referee
  "observations":        int,
  "confidence_score":    float,    // 0–1
  "verification_status": string,   // unverified | verified | corrected
  "soft_registered":     bool,
  "stats":               { … }     // below
}
```

### stats
Counts = int · `_pct`/`percent` = float 0–100 · `xg_*` = float · distance = float(m) · `_s` = float(sec)

```
goals_total  assists_total  expected_assists  goals_standard_situation
free_kick_scored  goals_conceded  goals_standard_situation_conceded  clean_sheets

shots_on_target_total  shots_wide_total  blocked_shots_by_opponent  shots_on_post_bar
penalty_total  close_range_shots_total  mid_range_shots_total  long_range_shots_total
chances_total  chances_conversion_pct  shots_on_target_pct  penalty_conversion_pct

xg_total  xg_per_90  xg_per_goal  xg_per_shot_saved
xg_foot_no_opponent  xg_header_no_opponent  xg_foot_opponent_present  xg_header_opponent_present

passes_total  passes_accurate  accurate_passes_percent  passes_per_90
foot_passes_open_play_total  accurate_foot_passes_open_play_total  foot_passes_open_play_accurate_pct
hand_passes_total  hand_passes_accurate_total  hand_passes_accurate_pct
short_passes_total  short_passes_accurate_total  short_passes_accurate_pct
medium_passes_total  medium_passes_accurate_total  medium_passes_accurate_pct
long_passes_total  accurate_long_passes_total  long_passes_accurate_pct
crosses_total  crosses_accurate_total  crosses_accurate_pct

challenges_total  challenges_won_total  challenges_won_pct
tackles_total  tackles_successful_total  tackles_won_pct
ball_interceptions_total  ball_recoveries_total  ball_recoveries_opp_half
dribbles_total  dribbles_successful_total  dribbles_success_pct

fouls_total  fouls_suffered  offsides_total  played_offside

saves_total  shots_saved_total  shots_saved_pct
close_range_saved_total  mid_range_saved_total  long_range_saved_total
jumping_saves_total  saves_without_jumping_total
penalties_saved_total  freekick_saved_total  corners_saved_total

total_distance  minutes_played  time_on_ball_s  touch_frames  ball_touches_total  packing_total
```

---

## Event
Common: `type` · `frame` int · `time_s` float · `confidence` float · `identity_confidence` float · `status` string
Player fields = jersey number (int) or null.

| type | extra fields |
|---|---|
| `pass` | `from` `to` `complete` `identity_confidence_receiver` |
| `shot` | `player` `direction` `speed` `xg` `xg_inputs` `geometry_reliable` |
| `dribble` | `player` `against` `successful` |
| `tackle` | `by` `on` |
| `interception` | `by` |
| `foul` | `by` `on` |
| `touch` | `player` `end_frame` |
| `cross` | `from` `to` `complete` |
| `goal` * | `player` `assist` |
| `assist` * | `player` `to` `goal_frame` |
| `goal_restart` * | `frame` `time_s` |

\* coming soon (additive).

---

## Clip
```jsonc
{
  "clip": string, "type": string, "player": int,
  "confidence": float, "identity_confidence": float, "identity_confidence_receiver": float|null,
  "codec": string, "size_bytes": int, "status": string,
  "event_frame": int, "source_frame": int,
  "time_s": float, "clip_start_s": float, "clip_end_s": float,
  "player_boxed": bool, "ball_frames": int
}
```

---

## DB tables
| Table | Key fields |
|---|---|
| `matches` | `analysis_id` `matches_video_id` `user_id` `source_url` `status` `roster_json` `player_stats_json` `created_at` `updated_at` |
| `ai_player_stats` | `player_key` `jersey_number` `team_name` `player_name` `verification_status` `stats_json` `player_json` |
| `events` | `event_index` `event_type` `frame` `time_s` `primary_player` `secondary_player` `confidence` `identity_confidence` `identity_confidence_receiver` `status` |
| `clips` | `clip_index` `clip_path` `event_type` `player` `player_boxed` `upload_status` `time_s` `event_frame` `size_bytes` |
| `roster_players` | `team_name` `team_color` `jersey_number` |
| `roster_known_stats` | `metric` `team_name` `stat_value_json` |

Child tables → `matches.analysis_id`. `team_name` in `ai_player_stats` = colour; join to `roster_players.team_color` for club.
