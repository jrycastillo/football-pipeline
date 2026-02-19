# Team Balance Fix (Round 5) — 2026-02-20 v1

## Context

Across all H100 runs (Feb 12–19), player detection totals are close to 22 (11v11),
but the **team split is severely imbalanced**:

| Video | Team A | Team B | Total | Expected |
|-------|--------|--------|-------|----------|
| V1 | Blue: **7** | Red: 13 | 20 | 11v11 |
| V2 | Blue: 13 | Green: 11 | 24 | 11v11 |
| V3 | Blue: **6** | Red: 15 | 21 | 11v11 |

V3 has 6 Blue vs 15 Red — meaning ~5 actual Blue players are classified as Red.
V2 (Blue vs Green) is balanced because Blue and Green are far apart in HSV space.

**Root cause:** The color assignment pipeline stores the FIRST frame's color prediction
permanently, ignoring subsequent majority-voted corrections. Combined with a dead-code
path in `finalize_bindings()`, some players are also lost entirely.

**Files to change:**
- `pipeline_consolidated.py` — Color assignment, finalize_bindings

**Stats code NOT changed:** `stats/event_logic.py` and `stats/metrics.py` are untouched.

---

## Root Cause Analysis

### Bug 1: First-Color-Wins in `set_track_color()` (PRIMARY — causes imbalance)

**Location:** `pipeline_consolidated.py:905-926`

**Problem:** `set_track_color()` stores the FIRST non-Unknown color detection for a track
and never updates it, even as the voting buffer stabilizes to a different color:

```python
def set_track_color(self, track_id, color, cls_id=None):
    if track_id not in self.track_colors or self.track_colors[track_id] == "Unknown":
        self.track_colors[track_id] = color  # First non-Unknown wins forever
        if color != "Unknown":
            self.color_counter[color] += 1   # Team detection also uses first color
```

**The flow:**
1. Frame 1: `predict_with_voting()` buffer = [Red] → returns "Red" (1 sample, noisy)
2. `set_track_color()` stores "Red" permanently
3. Frame 2-30: Buffer becomes [Red, Blue, Blue, Blue, ...] → returns "Blue"
4. `set_track_color()` **ignores** all calls because `track_colors[tid]` is already "Red"
5. At lock time: `player_colors[jersey] = track_colors[tid]` = "Red" (wrong!)
6. `_cluster_teams()` groups this player into Team Red

**Impact:** Blue players whose first crop is misclassified as Red are permanently assigned
to the Red team. This is especially likely for Blue vs Red matches because Red wraps
around the HSV hue wheel (H: 0-10 AND 170-180), and a purplish-blue jersey near H: 168
can initially classify as Red.

**Why V2 (Blue vs Green) is fine:** Blue (H: 100-140) and Green (H: 55-85) are far apart
in HSV space. First-frame misclassification is extremely unlikely.

### Bug 2: `finalize_bindings()` is Dead Code in Mode 2 (causes lost players)

**Location:** `pipeline_consolidated.py:1028-1063`

**Problem:** The pipeline uses Mode 2 (`self.locking_mode = 2`, line 502). Mode 2
accumulates votes in `self.vote_counts` (line 605). But `finalize_bindings()` iterates
`self.alpha.keys()` (line 1039) — which is the Mode 3 (Bayesian Dirichlet) accumulator.

In Mode 2, `self.alpha` is never populated, so `finalize_bindings()` consolidates
**zero tracks**. It's effectively dead code.

```python
def finalize_bindings(self):
    for tid in sorted(self.alpha.keys(), ...):  # alpha is EMPTY in Mode 2!
        if tid in self.active_bindings: continue
        track_alphas = self.alpha[tid]
        # ... never reached ...
```

**Impact:** Tracks that don't reach Mode 2's lock threshold (3 votes + score margin ≥ 1.0)
are permanently unbound. These players never appear in `player_colors` and get excluded
from the final player list. This explains why total count is sometimes < 22.

### Bug 3: `color_counter` reflects first-frame colors (amplifies imbalance)

**Location:** `pipeline_consolidated.py:924-926`

**Problem:** `color_counter` (used by `detect_team_colors()` to identify the two teams)
is only incremented on the FIRST color detection per track. Since the first detection is
the noisiest, the team color frequencies are based on unreliable data.

This doesn't change the TOP-2 team color detection (both Blue and Red still appear as top
colors), but it amplifies the per-player misassignment by not correcting early mistakes.

---

## Fixes Applied

### P0-1: Use Voting Buffer Color in `set_track_color()` (pipeline_consolidated.py:905)

**Fix:** Instead of permanently storing the first color, update `track_colors` every time
the voting buffer produces a different (and presumably better) result. Also update
`color_counter` to reflect the corrected color.

```python
def set_track_color(self, track_id, color, cls_id=None):
    if cls_id == 3:
        return

    old_color = self.track_colors.get(track_id)

    # Always update to latest voted color (not just first detection)
    if color != "Unknown":
        self.track_colors[track_id] = color

        # Goalkeeper: Store separately, don't count for team colors
        if cls_id == 1:
            self.goalkeeper_colors[track_id] = color
            return

        # Player: Update color counter (correct previous count if changed)
        if old_color and old_color != "Unknown" and old_color != color:
            # Decrement old color count
            if old_color in self.color_counter and self.color_counter[old_color] > 0:
                self.color_counter[old_color] -= 1
        if old_color != color:
            # Increment new color count (only if this is a change or first time)
            self.color_counter[color] = self.color_counter.get(color, 0) + 1
    elif old_color is None:
        self.track_colors[track_id] = "Unknown"
```

**Impact:** As the voting buffer stabilizes (typically after 5-10 frames), `track_colors`
converges to the correct color. A Blue player initially misclassified as Red will be
corrected once the majority vote flips. The `color_counter` also self-corrects.

### P0-2: Fix `finalize_bindings()` for Mode 2 (pipeline_consolidated.py:1028)

**Fix:** Use `self.vote_counts` instead of `self.alpha` in Mode 2, with a relaxed
threshold (total score ≥ 0.5, jersey already registered, OR total score ≥ 1.0).

```python
def finalize_bindings(self):
    log("[IdentityManager] Starting Tracklet Consolidation...")
    consolidated_count = 0

    # Use the correct vote store based on locking mode
    if self.locking_mode == 2:
        vote_store = self.vote_counts
    else:
        vote_store = self.alpha

    for tid in sorted(vote_store.keys(), key=lambda x: str(x)):
        if tid in self.active_bindings:
            continue

        track_votes = vote_store[tid]
        if not track_votes:
            continue

        best_number = max(track_votes, key=track_votes.get)
        evidence = track_votes[best_number]

        if (evidence > 0.5 and best_number in self.jersey_registry) or (evidence > 1.0):
            log(f"[Finalize] Consolidating Track {tid} -> Jersey #{best_number} (Evidence: {evidence:.1f})")
            self.active_bindings[tid] = best_number
            self.track_map[tid] = best_number
            consolidated_count += 1

    log(f"[Finalize] Consolidated {consolidated_count} fragmented tracklets.")
```

**Impact:** Tracks that accumulated some JNR votes but didn't reach the full lock threshold
(3 votes + margin ≥ 1.0) can now be consolidated. This should recover 1-2 "lost" players
per video.

### P1-1: Update `player_colors` at Lock Time from Voting Buffer (pipeline_consolidated.py:2065)

**Fix:** When a jersey is locked via `resolve_identity()`, use the latest `track_colors`
value (which is now kept up-to-date by P0-1) instead of potentially stale data.

This is automatically fixed by P0-1 — since `track_colors` now reflects the stable voted
color, the existing code at line 735 (`self.player_colors[player_key] = team_color`) will
get the correct color.

**No additional code change needed** — P0-1 fixes the upstream data.

---

## Known Bug: Double `all_frames.append` (NOT FIXED)

**Location:** `pipeline_consolidated.py` lines 2090 and 2114

Still present. Doubles all observation counts. Intentionally deferred for stats consistency.

---

## Expected Impact

| Issue | Before Fix | After Fix |
|-------|-----------|-----------|
| V1 team split | 7 Blue / 13 Red | ~10-11 Blue / 10-11 Red |
| V3 team split | 6 Blue / 15 Red | ~10-11 Blue / 10-11 Red |
| V2 team split | 13 Blue / 11 Green | ~11-12 Blue / 11-12 Green (minimal change) |
| Lost players (< 22 total) | 1-2 per video | ~0 (finalize recovers them) |
| Stats values | Will shift | Expected: more balanced team distributions |

**Note:** Since team assignment changes, per-player stats will differ from Feb 19 results.
Pass accuracy, interceptions, and tackles are all team-relative, so correcting the team
split should also improve their accuracy.

---

## Verification

```bash
python3 -c "import py_compile; py_compile.compile('pipeline_consolidated.py', doraise=True)"
```

### Comparison Test
Run the same videos on H100 and compare:
1. Team split should be closer to 11v11
2. Total player count should be ≥ 22
3. Results should still be deterministic (run twice → identical output)

---

## Files Changed

- `pipeline_consolidated.py` — P0-1 (`set_track_color` update logic), P0-2 (`finalize_bindings` Mode 2 fix)

