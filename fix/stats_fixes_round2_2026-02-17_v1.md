# Stats Fixes Round 2 — 2026-02-17 v1

## Context

Based on the H100 comparison (Feb 12 vs Feb 14), the formula fixes from Round 1
worked as intended (passes +153%, xG -23%, tackles -35%, dribbles -41%), but
revealed four new issues that need fixing.

**Files changed:**
- `stats/event_logic.py` — GK gating, shot interpolation filter, interception debounce, section 4 xG
- `stats/metrics.py` — Pass raw ball frame indices to analyzer
- `vision/ball_tracking.py` — Reduce max_gap

---

## Fixes

### P0-1: Shot Inflation from Ball Interpolation (event_logic.py + metrics.py)

**Problem:** Shots increased +60% (25→40) despite raising the speed threshold from
8→12 m/s. Root cause: ball interpolation max_gap was expanded from 25→50 frames,
creating ~2x more continuous ball track data. Linear interpolation across long gaps
can create artificial velocity spikes at gap boundaries (the ball "teleports" between
two distant detections).

**Fix:** Pass the set of raw (actually-detected) ball frame indices from BallTracker
through to the shot detection loop. Only compute ball velocity when BOTH frames (i and
i-2) are raw detections, not interpolated positions.

**Changes:**
1. `metrics.py`: Extract `raw_ball_frames = set(ball_tracker.tracks.keys())` and pass
   to `self.detector.analyze()`
2. `event_logic.py`: Update `analyze()` signature to accept `raw_ball_frames` parameter.
   In section 5 (shot detection) and section 6 (GK saves), skip velocity calculations
   when either frame is interpolated.

### P0-2: GK Stats Gating (event_logic.py)

**Problem:** Goalkeepers accumulate shots, xG, and goals from goal kicks, punts, and
clearances. In Feb 14 results:
- Red #1 (GK): 3 shots, 1.18 xG
- Green #18 (GK): 3 shots, 0.28 xG

Section 5 already has a partial GK filter (line 476) that nullifies the shooter if
dominant_class==1, but Section 4 (touch-in-box xG, line 666) has NO GK gate at all —
any GK touching the ball in their own penalty box accumulates xG.

**Fix:** Add GK gate in section 4. Skip xG accumulation for players with
dominant_class==1. The existing section 5 filter is already correct.

### P1-1: Reduce Ball Interpolation max_gap (ball_tracking.py)

**Problem:** max_gap=50 frames. At VID_STRIDE=5, that's 50*5/25 = 10 seconds of
linear interpolation. A ball can travel the entire pitch in 10 seconds — interpolating
across this gap produces meaningless positions.

**Fix:** Reduce max_gap from 50→30. At VID_STRIDE=5: 30*5/25 = 6 seconds max gap.
At VID_STRIDE=3: 30*3/25 = 3.6 seconds. Still generous but more physically plausible.

### P1-2: Interception Debounce (event_logic.py)

**Problem:** Players accumulate 15-24 interceptions per match. Real-world top
interceptors get 1-3 per match. Every incomplete pass received by an opponent counts
as an interception with no deduplication — rapid back-and-forth possession changes
in a small area generate many interceptions.

**Fix:** Add per-player debounce: max 1 interception per 3-second window per player.
Mirrors the existing shot debounce pattern.

### P1-3: Section 4 xG Uses Wrong Formula (event_logic.py)

**Problem:** Section 4 (touch-in-box xG, line 679) still uses the old non-logistic
formula: `xg = 0.1 * (10/dist) * (angle/45)`. This was overlooked in Round 1 which
only fixed the `calculate_xg()` helper used by section 5.

**Fix:** Replace section 4's inline xG formula with a call to the already-fixed
`calculate_xg()` helper function.

---

## Expected Impact

| Metric       | Feb 14 | Expected After |
|-------------|--------|----------------|
| Shots        | 40     | 15-25          |
| GK xG        | ~1.46  | 0.00           |
| Interceptions| ~130   | 40-60          |
| Section 4 xG | inflated | calibrated   |

---

## Verification

```bash
python3 -c "import py_compile; py_compile.compile('stats/event_logic.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('stats/metrics.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('vision/ball_tracking.py', doraise=True)"
```
