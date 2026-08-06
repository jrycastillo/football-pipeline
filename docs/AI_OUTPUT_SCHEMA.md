# AI Output & Database Schema — for Jhan (front-end / back-end integration)

**Single source of truth** for everything the pipeline produces per match: the
JSON the AI writes, how it's stored in the database, and how both map to the admin
dashboard. Grounded in the actual current run output (not a mock-up).

**Data flow:**

```
AI pipeline ──▶ JSON files ──▶ Database ──▶ Front-end / admin dashboard
                (§2–§6)        (§7 blob or normalized tables)     (§10 KPI row)
```

**Status legend:** `CURRENT` = produced today · `COMING` = additive, landing this
week (nothing removed or renamed — no breaking changes).

---

## 0. Source model (read first)

The dashboard is a **derived view**. Every KPI comes from one of four sources —
**not all of them are the AI**:

| Source | Meaning | Who produces it |
|---|---|---|
| **AI-tagged** | An event the AI detects & tags (shot, pass, goal…); the admin **reviews & corrects** it. | **Our AI** + admin verification |
| **admin-tagged** | A one-click event the AI can't reliably detect on single camera (corner, throw-in, own goal). | **Admin only** |
| **feed** | External feed — cards (`database`), and per the dashboard spec crosses/fouls (`afrisaut`). | **Not our AI** |
| **calculated** | Derived (conversion %, xG totals, chances). | Either side, one formula |

The AI's job = the **AI-tagged events** + the **~90 per-player metrics** (§3). The
admin corrects them; the dashboard aggregates them into the home-vs-away KPI row.

---

## 1. The ONE decision we need from you

The pipeline writes results in **two shapes** — tell us which the front-end reads:

- **(A) Legacy blob** — table `MatchesVideoAnalysis_test`, column `analysis`
  (one JSON dump of the whole result). Written by `upsert_status_row()`.
- **(B) Normalized tables** — the 6 queryable tables in §7. Written by
  `persist_run_to_db()` (`--write_db`). **Recommended** — query per player / per
  event without parsing a blob.

Both are written and don't conflict; we just need to know which you build against
so we validate the same thing.

---

## 2. Output artifacts (JSON the AI writes)

| File | Shape | Purpose |
|---|---|---|
| `player_stats.json` | dict keyed by jersey number | Per-player metrics (§3) |
| `raw_tracks.json` | list of event objects | The match event stream (§4) |
| `clips_manifest.json` | list of clip objects | One entry per generated clip (§5) |
| `track_jersey.json` | dict keyed by track id | Track-id → jersey/team map for overlays (§6) |
| `match_kits.json` | dict | Discovered team colours |

These JSON files are the **inputs** to the DB layer (§7): `player_stats.json` →
`ai_player_stats` + `matches.player_stats_json`; `raw_tracks.json` → `events`;
`clips_manifest.json` → `clips`.

---

## 3. `player_stats.json`

Top level: `{ "<jersey_number>": <PlayerEntry> }` — key is the jersey number as a
string. This object is stored verbatim in `matches.player_stats_json`; each entry
becomes a row in `ai_player_stats` (its `stats` → `stats_json`).

### 3.1 PlayerEntry (meta fields)
```jsonc
{
  "player_name":         "Player 8",      // string
  "jersey_number":       8,               // int
  "team":                "White",         // string — team COLOUR, not club name (see §8)
  "position":            "Player",        // string
  "role":                "Player",        // "Player" | "Goalkeeper" | "Referee"
  "observations":        1423,            // int — frames the player was tracked
  "confidence_score":    0.87,            // float 0–1 — identity confidence
  "verification_status": "unverified",    // "unverified" | "verified" | "corrected"
  "soft_registered":     false,           // bool — jersey bound but not hard-locked
  "stats":               { ...§3.2 }
}
```

### 3.2 `stats` — all metrics (CURRENT unless marked)

Types: `_total`/counts = int; `_pct`/`percent` = float 0–100; `xg_*` = float;
distances = float (metres); `_s` = float (seconds).

**General / summary**
```
total_distance, minutes_played, time_on_ball_s, touch_frames,
ball_touches_total, packing_total
```
**Goals & assists**
```
goals_total, assists_total, expected_assists, goals_standard_situation,
free_kick_scored, goals_conceded, goals_standard_situation_conceded, clean_sheets
```
**Shooting**
```
shots_on_target_total, shots_wide_total, blocked_shots_by_opponent,
shots_on_post_bar, penalty_total,
close_range_shots_total, mid_range_shots_total, long_range_shots_total,
chances_total, chances_conversion_pct, shots_on_target_pct, penalty_conversion_pct
```
**Expected goals (xG)**
```
xg_total, xg_per_90, xg_per_goal,
xg_foot_no_opponent, xg_header_no_opponent,
xg_foot_opponent_present, xg_header_opponent_present,
xg_per_shot_saved
```
**Passing**
```
passes_total, passes_accurate, accurate_passes_percent, passes_per_90,
foot_passes_open_play_total, accurate_foot_passes_open_play_total,
foot_passes_open_play_accurate_pct,
hand_passes_total, hand_passes_accurate_total, hand_passes_accurate_pct,
short_passes_total, short_passes_accurate_total, short_passes_accurate_pct,
medium_passes_total, medium_passes_accurate_total, medium_passes_accurate_pct,
long_passes_total, accurate_long_passes_total, long_passes_accurate_pct,
crosses_total, crosses_accurate_total, crosses_accurate_pct
```
**Duels / defending**
```
challenges_total, challenges_won_total, challenges_won_pct,
tackles_total, tackles_successful_total, tackles_won_pct,
ball_interceptions_total, ball_recoveries_total, ball_recoveries_opp_half,
dribbles_total, dribbles_successful_total, dribbles_success_pct
```
**Discipline**
```
fouls_total, fouls_suffered, offsides_total, played_offside
```
**Goalkeeping**
```
saves_total, shots_saved_total, shots_saved_pct,
close_range_saved_total, mid_range_saved_total, long_range_saved_total,
jumping_saves_total, saves_without_jumping_total,
penalties_saved_total, freekick_saved_total, corners_saved_total
```

---

## 4. `raw_tracks.json` — event stream

A flat JSON list; each event → one `events` row. **Common fields:**

```jsonc
{
  "type":                 "shot",         // event type (table below)
  "frame":                655,            // int — PROCESSED frame index
  "time_s":               78.72,          // float — seconds (COMING on raw events; already in clips + DB events.time_s)*
  "confidence":           0.43,           // float 0–1 — event confidence
  "identity_confidence":  0.90,           // float 0–1 — confidence in the player's jersey id
  "status":               "unverified"    // "unverified" | "verified" | "corrected" | "rejected"
}
```
\* `time_s = source_frame / fps`, `source_frame = (frame + 1) * vid_stride`
(stride 3, 25 fps in these runs).

Per-type extra fields (player fields are **jersey numbers**, int, or `null`). In
the DB `events` table these collapse to `primary_player` (the actor: `player` /
`from` / `by`) and `secondary_player` (`to` / `on` / `against`):

| type | extra fields | notes |
|---|---|---|
| `pass` | `from`, `to`, `complete` (bool), `identity_confidence_receiver` | `to`/`complete` null/false if intercepted |
| `shot` | `player`, `direction`, `speed` (m/s float), `xg` (float), `xg_inputs` (obj), `geometry_reliable` (bool) | `geometry_reliable=false` ⇒ speed/xg nulled |
| `dribble` | `player`, `against`, `successful` (bool) | |
| `tackle` | `by`, `on` | |
| `interception` | `by` | |
| `foul` | `by`, `on` | |
| `touch` | `player`, `end_frame` (int) | possession spell |
| `cross` | `from`, `to`, `complete` (bool) | CURRENT (appears when detected) |
| `goal` | `player`, `assist` (jersey or null) | **COMING** — emitted when a shot is scored |
| `assist` | `player`, `to`, `goal_frame` (int) | **COMING** — last pass to the scorer, ≤15s |
| `goal_restart` / `kickoff` | `frame`, `time_s` | **COMING** — Babak's center-kickoff goal-confirmation cue |

---

## 5. `clips_manifest.json` — one entry per clip  (CURRENT) → `clips` table

```jsonc
{
  "clip":                          "clips/000_shot_p42_f1968.mp4", // string, relative path → clip_path
  "type":                          "shot",       // event type
  "player":                        42,            // int jersey (primary actor)
  "confidence":                    0.43,          // float
  "identity_confidence":           0.90,          // float
  "identity_confidence_receiver":  null,          // float | null (passes)
  "codec":                         "h264",        // "h264" | "mp4v"
  "size_bytes":                    3058847,       // int
  "status":                        "unverified",  // review state → upload_status/status
  "event_frame":                   655,           // int — processed frame
  "source_frame":                  1968,          // int — original-video frame
  "time_s":                        78.72,         // float — event timestamp (seconds)
  "clip_start_s":                  74.24,         // float — clip window start
  "clip_end_s":                    83.20,         // float — clip window end
  "player_boxed":                  true,          // bool — actor was highlighted
  "ball_frames":                   215            // int — frames the ball was drawn
}
```

---

## 6. `track_jersey.json` — overlay map  (CURRENT)

`{ "<track_id>": { "jersey": <int|null>, "team": <colour|null> } }`

Maps raw tracker IDs → jersey/team; powers the "box every player with T{id}
#{jersey}" overlay. `jersey` is `null` for tracks that never locked a number.

---

## 7. Database — normalized tables (shape B)

| Table | One row per | Key fields the front-end will use |
|---|---|---|
| `matches` | video | `analysis_id`, `matches_video_id`, `user_id`, `source_url`, `status`, `roster_json`, `player_stats_json`, `created_at`, `updated_at` |
| `ai_player_stats` | player | `player_key`, `jersey_number`, `team_name`* , `player_name`, `verification_status`, `stats_json` (the §3.2 object, ~90 metrics), `player_json` |
| `events` | event | `event_index`, `event_type`, `frame`, `time_s`, `primary_player`, `secondary_player`, `confidence`, `identity_confidence`, `identity_confidence_receiver`, `status` |
| `clips` | clip | `clip_index`, `clip_path`, `event_type`, `player`, `player_boxed`, `upload_status`, `time_s`, `event_frame`, `size_bytes` |
| `roster_players` | roster entry | `team_name`, `team_color`, `jersey_number` |
| `roster_known_stats` | metric/team | `metric`, `team_name`, `stat_value_json` |

All child tables key back to `matches.analysis_id` (foreign key, cascade delete).
\* `team_name` here is a **colour** — see §8.

---

## 8. Important join note (`team_name` is a colour, not the club name)

`ai_player_stats.team_name` holds the player's **team colour** (`"White"`,
`"Black"`) — not the club name. `roster_players` holds both `team_name`
(`"Hamburger SV"`) **and** `team_color` (`"white"`).

**To map a player to their club, join on colour:**
`ai_player_stats.team_name` (colour) → `roster_players.team_color`.

Roster-unique jersey numbers get the **correct** colour (a fix shipped for ~5
previously-mislabelled players). Numbers on *both* rosters (e.g. #1, #9, #17) are
colour-derived and can't be club-resolved by number alone — treat as "colour
known, club ambiguous" if surfaced.

---

## 9. Same-video handling (idempotent — no duplicates)

Re-processing a video is safe:
- `matches` upserts on `analysis_id` (`ON DUPLICATE KEY UPDATE`).
- Child rows are deleted-and-reinserted by `analysis_id`.
- `is_video_processed()` skips finished videos and reclaims crashed runs.

You never get duplicate rows for the same video.

---

## 10. Dashboard KPI → schema (the 5 blocks in the admin-dashboard spec)

`AI` = we produce it · `admin` = admin one-click tags it · `feed` = external ·
`calc` = derived.

| Block | KPI | Source | Schema field |
|---|---|---|---|
| Result | Goals | AI | `goals_total` / `goal` event |
| | Goals after standard situation | AI | `goals_standard_situation` |
| | Own goals | **admin** | *(not AI-detected)* |
| Shooting | Shots on target / wide / blocked / post-bar | AI | `shots_on_target_total`, `shots_wide_total`, `blocked_shots_by_opponent`, `shots_on_post_bar` |
| | Shots total | calc | Σ of the four above |
| | xG total | AI | `xg_total` |
| | xG per shot | calc | `xg_total / chances_total` |
| | Chances / Conversion % | AI | `chances_total`, `chances_conversion_pct` |
| | Shot distance split | AI | `close/mid/long_range_shots_total` |
| Set pieces | Corners | **admin** | *(not AI-detected)* |
| | Free kicks (shooting) | AI/admin | `free_kick_scored` (scored only) |
| | Throw-ins final third | **admin** | *(not AI-detected)* |
| | Penalties (awarded/scored/saved) | AI | `penalty_total`, `penalties_saved_total`, `penalty_conversion_pct` |
| | Crosses / accurate | AI **or feed** | `crosses_total`, `crosses_accurate_total` — resolve source |
| Discipline | Yellow / Red cards | **feed** | *(database feed, not AI)* |
| | Fouls | AI **or feed** | `fouls_total` — resolve source |
| | Offsides | AI | `offsides_total`, `played_offside` |
| Goalkeeping | Saves total / by distance | AI | `saves_total`, `close/mid/long_range_saved_total` |
| | Jumping / without | AI | `jumping_saves_total`, `saves_without_jumping_total` |
| | Pen / FK / corners saved | AI | `penalties_saved_total`, `freekick_saved_total`, `corners_saved_total` |
| | xG conceded / Defensive xG | calc | mirror of opponent shots (`goals_conceded`, `xg_per_shot_saved`) |

---

## 11. Gaps, open decisions & go-live checklist

**Gaps / decisions (for the team):**
1. **AI can't detect** own goals, corners, throw-ins-in-final-third on single
   camera → **admin one-click tags** them (matches the dashboard's "tagged" model).
2. **Cards** = `database` feed, not AI. **Crosses/Fouls** listed as `afrisaut`
   feed but the AI *also* computes them → pick one source of truth. **What is
   `afrisaut`?** — confirm whether it's a live feed.
3. **"tagged" = AI-pre-tagged + admin-verified**, not manual-from-scratch. Confirm
   this shared model — it's the whole architecture.
4. **Team vs player:** `player_stats.json` / `ai_player_stats` are **per-player**;
   the dashboard row is **team totals** = aggregation of per-player fields.
5. **COMING additive fields** (§4): `goal`/`assist`/`goal_restart` events + `time_s`
   on raw events. Safe to build against now.

**To go live we need:** (a) your answer to §1 (blob vs normalized), (b) the rotated
DB password (old one was committed, purged, must be rotated), (c) the live table
name (config points at `MatchesVideoAnalysis_test`). Then one real end-to-end write
and you can point the front-end at it.
