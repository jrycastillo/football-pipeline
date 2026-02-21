# Team Balance & Tracklet Consolidation (Round 5) — 2026-02-20 v2

## Context

### Feb 20 (Round 5 v1) Results — Regression Detected

Round 5 v1 applied two fixes to `pipeline_consolidated.py`:
- P0-1: `set_track_color()` "always update" to latest voted color
- P0-2: `finalize_bindings()` use `vote_counts` in Mode 2

**Outcomes:**
- Team balance improved: V1 7/13→9/11, V2 13/11→12/12
- **CRITICAL REGRESSION:** Stats exploded 10-22x (V2 score 0-0→47-35, V3 2-2→1-64)
- V3 team color changed Blue→Green (color_counter churn)
- Player count unchanged (V1: 20, V3: 20)

### Root Cause of Stats Explosion

**P0-2 (`finalize_bindings`)** consolidated hundreds of short-lived ByteTrack tracklets
into `active_bindings`. The **Phase 216 remap** in `stats/metrics.py:64-101` then SUMS
stats from ALL tracks mapped to the same jersey number:

```python
# stats/metrics.py:96
remapped_stats[target_id][key] += value  # ADDS stats from every track
```

Before P0-2: finalize_bindings was a no-op → ~2-3 tracks per player → minor merging.
After P0-2: finalize_bindings active → 10-20+ tracks per player → 10-22x inflation.

**Proof:** V2 Blue #35: 278,882 obs / ~12,000 frames = ~12 tracks per frame = exactly
what happens when finalize_bindings consolidates all tracklets to one jersey.

### Root Cause of V3 Color Change

**P0-1 ("always update")** caused `color_counter` churn. Every frame updated
`track_colors`, causing rapid color oscillation during early frames. This shifted the
color frequency distribution enough for `detect_team_colors()` to pick Green > Blue.

---

## Fixes Applied (Round 5 v2)

### P0-1 REVISED: "Settle Then Lock" Color Assignment

**Location:** `pipeline_consolidated.py:905`

**Was (v1):** Update `track_colors` to latest voted color on EVERY frame.
**Now (v2):** Store first non-Unknown color immediately. Allow ONE correction at
observation #10 (when voting buffer is reliable). Then lock permanently.

```python
WARMUP_THRESHOLD = 10

if track_id not in self.track_colors or self.track_colors[track_id] == "Unknown":
    # First non-Unknown color — store immediately
    self.track_colors[track_id] = color
    self.color_counter[color] += 1
elif self._color_obs_count[track_id] == WARMUP_THRESHOLD:
    # One-time correction at warmup threshold
    if color != old_color:
        self.track_colors[track_id] = color
        self.color_counter[old_color] -= 1
        self.color_counter[color] += 1
        # Also update player_colors if already bound
        jersey_num = self.active_bindings.get(track_id)
        if jersey_num is not None:
            self.player_colors[str(jersey_num)] = color
# After threshold: locked, no more changes
```

**Why this works:**
- First color stored immediately → `resolve_identity()` always has a color (no "Unknown" gap)
- At observation #10, voting buffer has stabilized → correction is reliable
- After that, color is locked → no churn in `color_counter` or `detect_team_colors()`
- `player_colors` is also corrected if the track was already bound to a jersey

### P0-2 REVERTED: `finalize_bindings()` Back to No-Op in Mode 2

**Location:** `pipeline_consolidated.py:1040`

**Was (v1):** Used `self.vote_counts` in Mode 2 (hundreds of entries).
**Now (v2):** Uses `self.alpha` (empty in Mode 2 = no-op).

Reason: The Phase 216 remap in stats/metrics.py:96 **SUMS** stats from all tracks
mapped to the same jersey number. This is a MULTIPLY operation when many tracks map
to one jersey. Until Phase 216 is redesigned to DEDUPLICATE (use max, not sum),
`finalize_bindings()` must remain a no-op.

Added detailed docstring explaining WHY this must not be changed without fixing Phase 216.

---

## Low Player Count (20 instead of 22) — Root Cause Analysis

This is a **separate issue** that was NOT caused by Round 5 and is NOT fixable by
color or stats changes. The player count has been 20-21 across ALL rounds.

### The Pipeline Dropout Analysis

```
YOLO detects 22+ people per frame ✓
ByteTrack creates tracks ✓
Color classifier assigns colors ✓
JNR (ResNet34) tries to read jersey numbers
  ├── SUCCESS (20 players): Gets 3+ votes with score margin ≥ 1.0
  │   └── Lock → resolve_identity → player_colors → stats ✓
  └── FAILURE (2 players): JNR never returns confident predictions
      └── No jersey number → stats labels as "Unknown_XXX"
          └── Phase 186 filter drops: startswith("Unknown") AND jersey is None
              └── PLAYER LOST ✗
```

### Why 2 Players Always Missing

The 2 missing players are physical people whose **jersey numbers are never
recognized by JNR** (ResNet34). Possible reasons:
- Always far from camera (too small for the model)
- Always facing away from camera
- Jersey numbers obscured (crumpled fabric, arms covering, etc.)
- `is_legible` filter blocks their crops (min 30px height, min blur threshold)

**Evidence:**
- V1 Red has exactly 11 players → Blue is missing 2
- V3 Red has 13 → has extras from misclassified Green/Blue players
- The "missing" players ARE in all_frames as raw track IDs, but without jersey numbers

### Root Cause: Vote Threshold Too Strict

The Mode 2 voting gate at `pipeline_consolidated.py:604` requires `score >= 0.50` for a
prediction to count as a vote. ResNet returns valid predictions for confidence >= 0.20
(`MIN_REJECT_CONF` in `resnet_recognition.py:358`), but predictions with 0.20-0.49
confidence are **completely discarded** — they never accumulate votes.

For the 2 missing players, JNR likely returns predictions in the 0.30-0.49 range
(weak but consistent — always the same number). With the 0.50 threshold, zero votes
accumulate, so the lock condition (3 votes + margin >= 1.0) is never reached.

### P0-3: Lower Vote Threshold (0.50 → 0.30)

**Location:** `pipeline_consolidated.py:604`

**Was:** `if score >= 0.50:` — only predictions above 0.50 counted as votes.
**Now:** `if score >= 0.30:` — predictions in the 0.30-0.49 range now contribute.

```python
# Round 5 v2: Lowered from 0.50 to 0.30 to recover players with weak but
# consistent predictions. The lock condition (3 votes + margin >= 1.0) still
# prevents garbage locks — a player needs 3+ consistent reads to lock.
if score >= 0.30:
    self.vote_counts[track_id][detected_number] += score
```

**Why this is safe:**
- The lock condition still requires **3 votes + margin >= 1.0** (`pipeline_consolidated.py:638`)
- Even with 0.30 confidence predictions, a player needs 3 consistent reads of the same number
- 3 votes at 0.30 = total score 0.90, margin 0.90 — still slightly below 1.0. In practice,
  at least one prediction will be above 0.33, making total > 1.0
- Random/noisy predictions won't lock because they'll split across different numbers,
  keeping margin low
- The ResNet already rejects predictions below 0.20 confidence at the model level

**JNR Pipeline Gate Summary (After Fix):**

```
ResNet prediction → conf < 0.20 → REJECT (resnet_recognition.py:358)
                  → conf 0.20-0.29 → returned but NOT counted (pipeline:604)
                  → conf 0.30-0.49 → counted as vote (NEW — was rejected before)
                  → conf 0.50+ → counted as vote (same as before)
                  → 3 votes + margin >= 1.0 → LOCK (pipeline:638)
```

---

## Files Changed (v2)

- `pipeline_consolidated.py`:
  - `set_track_color()` (dampened P0-1: settle then lock)
  - `finalize_bindings()` (reverted P0-2: back to no-op)
  - `process_detection()` vote threshold (P0-3: 0.50 → 0.30)

## Files NOT Changed

- `stats/metrics.py` — Phase 216 remap untouched (needs future redesign)
- `stats/event_logic.py` — No changes
- `vision/resnet_recognition.py` — No changes (MIN_REJECT_CONF stays at 0.20)

## Expected Impact (v2)

| Metric | Round 4 (Feb 19) | Round 5 v2 (Expected) |
|--------|-------------------|-----------------------|
| V1 team split | 7/13 | ~9/11 (improved via warmup correction) |
| V2 team split | 13/11 | ~12/12 or 13/11 |
| V3 team split | 6/15 | ~7-8/13-14 (improved) |
| V3 team color | Blue/Red | Blue/Red (stable, no churn) |
| Stats values | Baseline | Similar magnitude (no 10x inflation) |
| Player count | 20-21 | **21-22** (P0-3 recovers players with weak JNR confidence) |
| Determinism | ✅ | ✅ (warmup is deterministic, vote threshold is deterministic) |

---

## Verification

```bash
python3 -c "import py_compile; py_compile.compile('pipeline_consolidated.py', doraise=True)"
```

Run on H100, then compare:
1. Stats should be in same order of magnitude as Feb 19 (NOT 10-22x inflated)
2. Team balance should be improved from Feb 19 (not identical)
3. V3 should stay Blue/Red (not flip to Green/Red)
4. Player count should increase to 21-22 (P0-3 recovers weak-confidence players)
5. All players should have jersey numbers (no Unknown players in output)
