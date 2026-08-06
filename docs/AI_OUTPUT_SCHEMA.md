# AI Output Schema — for Jhan (front-end / back-end integration)

**What this is:** the complete schema our AI pipeline produces per match. This is
the source of truth for the admin dashboard. Grounded in the actual current run
output (not a mock-up). Companion to `DB_FIELDS_FOR_JHAN.md` (which covers the DB
tables); this doc covers the **JSON the AI writes**.

**Status legend used below:** `CURRENT` = produced today · `COMING` = additive,
landing this week (nothing is removed or renamed — no breaking changes).

---

## 0. Source model (read first)

The dashboard is a **derived view**. Every KPI comes from one of four sources —
**not all of them are the AI**:

| Source | Meaning | Who produces it |
|---|---|---|
| **AI-tagged** | An event object the AI detects and tags (shot, pass, goal…). The admin **reviews & corrects** it. | **Our AI** + admin verification |
| **admin-tagged** | A one-click event the AI can't reliably detect on single-camera (corner, throw-in, own goal). | **Admin only** |
| **feed** | External feed — cards (`database`), and per the dashboard doc crosses/fouls (`afrisaut`). | **Not our AI** |
| **calculated** | Derived from the above (conversion %, xG totals, chances). | Either side, one formula |

So the AI's job is the **AI-tagged events** + the **~90 per-player metrics** below.
The admin corrects them; the dashboard aggregates them into the home-vs-away KPI row.

---

## 1. Output artifacts

| File | Shape | Purpose |
|---|---|---|
| `player_stats.json` | dict keyed by jersey number | Per-player metrics (§2) |
| `raw_tracks.json` | list of event objects | The match event stream (§3) |
| `clips_manifest.json` | list of clip objects | One entry per generated clip (§4) |
| `track_jersey.json` | dict keyed by track id | Track-id → jersey/team map for overlays (§5) |
| `match_kits.json` | dict | Discovered team colours |
| DB tables | 6 normalized tables | See `DB_FIELDS_FOR_JHAN.md` (§6) |

---

## 2. `player_stats.json`

Top level: `{ "<jersey_number>": <PlayerEntry> }` — key is the jersey number as a
string.

### 2.1 PlayerEntry (meta fields)
```jsonc
{
  "player_name":         "Player 8",      // string
  "jersey_number":       8,               // int
  "team":                "White",         // string — team COLOUR, not club name*
  "position":            "Player",        // string
  "role":                "Player",        // "Player" | "Goalkeeper" | "Referee"
  "observations":        1423,            // int — frames the player was tracked
  "confidence_score":    0.87,            // float 0–1 — identity confidence
  "verification_status": "unverified",    // "unverified" | "verified" | "corrected"
  "soft_registered":     false,           // bool — jersey bound but not hard-locked
  "stats":               { ...§2.2 }
}
```
\* `team` is the colour (`"White"`, `"Black"`). Map to club via `roster_players`
(join on colour) — see `DB_FIELDS_FOR_JHAN.md` §3.

### 2.2 `stats` — all metrics (CURRENT unless marked)

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

## 3. `raw_tracks.json` — event stream

A flat JSON list. Every event has these **common fields**:

```jsonc
{
  "type":                 "shot",         // event type (table below)
  "frame":                655,            // int — PROCESSED frame index
  "time_s":               78.72,          // float — seconds (COMING on raw events; already in clips + DB)**
  "confidence":           0.43,           // float 0–1 — event confidence
  "identity_confidence":  0.90,           // float 0–1 — confidence in the player's jersey id
  "status":               "unverified"    // "unverified" | "verified" | "corrected" | "rejected"
}
```
\*\* `time_s = source_frame / fps`, where `source_frame = (frame + 1) * vid_stride`
(stride 3, 25 fps in these runs). Present today in `clips_manifest.json` and the DB
`events.time_s`; being added onto the raw event objects too.

Per-type extra fields (player fields are **jersey numbers**, int, or `null`):

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

## 4. `clips_manifest.json` — one entry per clip  (CURRENT)

```jsonc
{
  "clip":                          "clips/000_shot_p42_f1968.mp4", // string, relative path
  "type":                          "shot",       // event type
  "player":                        42,            // int jersey (primary actor)
  "confidence":                    0.43,          // float
  "identity_confidence":           0.90,          // float
  "identity_confidence_receiver":  null,          // float | null (passes)
  "codec":                         "h264",        // "h264" | "mp4v"
  "size_bytes":                    3058847,       // int
  "status":                        "unverified",  // review state
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

## 5. `track_jersey.json` — overlay map  (CURRENT)

`{ "<track_id>": { "jersey": <int|null>, "team": <colour|null> } }`

Maps raw tracker IDs → jersey/team. Powers the "box every player with T{id}
#{jersey}" overlay. `jersey` is `null` for tracks that never locked a number.

---

## 6. DB tables

Six normalized tables (`matches`, `ai_player_stats`, `events`, `clips`,
`roster_players`, `roster_known_stats`) — full field lists and the **one open
decision** (JSON blob vs normalized tables) are in **`DB_FIELDS_FOR_JHAN.md`**.
Note `events.time_s` already exists there.

---

## 7. Dashboard KPI → schema (the 5 blocks in Jhan's doc)

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

## 8. Gaps & open decisions (for the team)

1. **AI can't detect** own goals, corners, throw-ins-in-final-third on single
   camera → **admin one-click tags** them (matches the dashboard's "tagged" model).
2. **Cards** = `database` feed, not AI. **Crosses/Fouls** listed as `afrisaut`
   feed but the AI *also* computes them → pick one source of truth. **What is
   `afrisaut`?** — confirm whether it's a live feed.
3. **"tagged" = AI-pre-tagged + admin-verified**, not manual-from-scratch. Confirm
   this is the shared model (it is the whole architecture).
4. Team-vs-player: `player_stats.json` is **per-player**; the dashboard row is
   **team totals** = aggregation of per-player fields.
5. **COMING additive fields** (§3): `goal`/`assist`/`goal_restart` events,
   `time_s` on raw events. Safe to build against now.
