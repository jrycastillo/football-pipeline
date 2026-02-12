# Stats Fix Report - 2026-02-12

**Branch:** production-v1.0
**Scope:** GK clustering fix, VID_STRIDE timing correction, goal detection fix, diagnostics

---

## Fix 1: GK Clustering Completely Broken (CRITICAL)

**Files:** `stats/metrics.py`
**Severity:** CRITICAL - GK clustering was dead code

### Problem

`_cluster_teams()` relied on `id_manager.track_classes` to identify goalkeepers:

```python
# OLD - BROKEN: track_classes does NOT exist on id_manager
if hasattr(id_manager, 'track_classes') and hasattr(id_manager, 'active_bindings'):
    for track_id, jersey_num in id_manager.active_bindings.items():
        cls_id = id_manager.track_classes.get(track_id)
```

Since `track_classes` doesn't exist, `hasattr()` returned `False`, so `jersey_classes` was **always empty `{}`**. This caused:

1. GKs were **never excluded** from team clustering - GK colors polluted team color detection
2. GK team assignment code **never ran** (`gk_jerseys` was always empty)
3. Only 1 GK (or none) appeared correctly in results
4. GK wearing a unique color (e.g., orange) could create a 3rd phantom "team"

### Fix

Build the class mapping from actual `all_frames` box data (which has YOLO `cls` on every detection) instead of the non-existent `track_classes`:

```python
# NEW - WORKING: Build from frame data directly
class_counts_per_id = defaultdict(lambda: defaultdict(int))
for f in all_frames:
    for b in f["boxes"]:
        pid = b.get("id")
        cls = b.get("cls", 2)
        if pid is not None:
            class_counts_per_id[pid][cls] += 1

player_dominant_classes = {}
for pid, cls_counts in class_counts_per_id.items():
    dominant_cls = max(cls_counts.items(), key=lambda x: x[1])[0]
    player_dominant_classes[pid] = dominant_cls

self._cluster_teams(id_manager, player_dominant_classes)
```

### Impact

| Behavior | Before | After |
|----------|--------|-------|
| GK identification in clustering | Never worked | Uses YOLO class 1 from all frames |
| GK excluded from team colors | No (polluted teams) | Yes |
| GK assigned to correct team | Never ran | By jersey number proximity |
| Number of GKs in results | 0-1 (random) | All detected GKs (typically 2) |

---

## Fix 2: VID_STRIDE Not Accounted For in All Calculations (CRITICAL)

**Files:** `stats/event_logic.py`, `stats/metrics.py`
**Severity:** CRITICAL - All velocities were 3x inflated, all time windows were 3x too long

### Problem

With `VID_STRIDE=3`, only every 3rd frame is processed. Each entry in `ball_track` / `ownership` / `all_frames` is `VID_STRIDE/FPS = 3/25 = 0.12s` apart, not `1/25 = 0.04s`.

But all calculations used raw `FPS = 25`:

```python
# OLD - WRONG: Assumes each frame = 1/25 second
speed_mps = dist_m / (2.0 / FPS)  # = dist_m / 0.08 (actual gap is 0.24s)
```

This made speeds **3x too high** and all time windows **3x too long**.

### Fix

Added `EFF_FPS = FPS / VID_STRIDE` and replaced all timing references:

```python
VID_STRIDE = HEURISTICS.get("VID_STRIDE", 1)
EFF_FPS = FPS / VID_STRIDE  # 25/3 = 8.33 for VID_STRIDE=3

speed_mps = dist_m / (2.0 / EFF_FPS)  # Correct: dist_m / 0.24
```

### All Affected Calculations

| Calculation | Before (VID_STRIDE=3) | After (VID_STRIDE=3) | Location |
|-------------|----------------------|---------------------|----------|
| Shot speed threshold (8 m/s) | Effectively 2.67 m/s | Actually 8 m/s | event_logic.py:378 |
| GK save speed | 3x inflated | Correct | event_logic.py:494 |
| GK save next-frame speed | 3x inflated | Correct | event_logic.py:530 |
| Dribble debounce (1s) | 3 seconds | 1 second | event_logic.py:198 |
| Pass max gap (3s) | 9 seconds | 3 seconds | event_logic.py:244 |
| Possession retain check (1.5s) | 4.5 seconds | 1.5 seconds | event_logic.py:747 |
| Ownership smoothing gap | 15 frames = 1.8s | ~5 frames = 0.6s | event_logic.py:73 |
| Shot debounce | 10 frames = 1.2s | ~8 frames = 1s | event_logic.py:412 |
| Save debounce | 20 frames = 2.4s | ~8 frames = 1s | event_logic.py:535 |
| Goal lookahead | 10 frames = 1.2s | ~17 frames = 2s | event_logic.py:461 |
| Time on ball (seconds) | 3x too low | Correct | metrics.py:213 |
| Minutes/seconds played | 3x too low | Correct | metrics.py:137-138 |

### Impact on Stats

| Stat | Before (broken) | After (fixed) |
|------|-----------------|---------------|
| **Shots on target** | Massively over-counted (2.67 m/s threshold) | Correct (8 m/s threshold) |
| **Goals** | False positives from inflated speed + short lookahead | Fewer false positives |
| **xG** | Inflated (too many "shots") | Correct |
| **Dribbles** | Under-counted (3s debounce too long) | More accurate (1s debounce) |
| **Passes** | Over-counted (9s gap allowed) | Correct (3s max gap) |
| **Time on ball** | 3x too low | Correct |
| **Minutes played** | 3x too low | Correct |
| **Player filtering** | Players with <9s played deleted (threshold appeared as 3s) | Correct 3s threshold |

---

## Fix 3: Goal Detection Lookahead Too Short

**File:** `stats/event_logic.py`
**Severity:** HIGH

### Problem

Goal confirmation checked only 10 frames ahead. With VID_STRIDE=3, that's only `10 * 3/25 = 1.2 seconds`. A shot from 16m at 8 m/s takes 2 seconds to reach the goal. Many valid goals were missed because the lookahead expired before the ball crossed the line.

### Fix

Increased to `max(10, int(2.0 * EFF_FPS))` = ~17 processed frames = 2 seconds of lookahead.

---

## Enhancement: Ball Detection & Ownership Diagnostics

**File:** `stats/metrics.py`

### Added Logging

```
[StatsEngine] Ball track: 28500/45000 frames (63.3%)
[StatsEngine] Ownership: 22100/45000 frames (49.1%) assigned to players
[Team] Found 2 goalkeeper(s) from YOLO class detection: [1, 22]
[Team] Skipping GK #1 (color: Yellow) from team clustering
[Team] Skipping GK #22 (color: Orange) from team clustering
```

- Ball detection rate with warning if < 30%
- Ownership coverage percentage
- GK detection count and jersey numbers

This helps diagnose pass/shot undercount issues - if ball detection is low, passes will be undercounted regardless of other fixes.

---

## Remaining Architectural Limitations (Not Bugs)

| Limitation | Impact | Notes |
|------------|--------|-------|
| **Camera uses default 0.1 m/px scaling** | Goal line detection (x > 105m) depends on projection accuracy | Need per-video homography calibration for precise results |
| **Ball detection near goal** | Ball often undetected in the net | If ball not detected during lookahead, goals missed |
| **No scoreboard cross-reference** | Goals purely from trajectory analysis | Could add OCR-based validation |
| **Linear ball interpolation** | Ball arcs/deflections not captured | Max 25-frame gap interpolation |

---

## Summary of All Changes (2026-02-12)

### Files Modified

| File | Lines Changed | Changes |
|------|--------------|---------|
| `stats/event_logic.py` | +23 / -13 | VID_STRIDE/EFF_FPS for all speed + timing calculations, goal lookahead |
| `stats/metrics.py` | +47 / -25 | GK clustering from frame data, EFF_FPS for time calcs, diagnostics logging |

### Deployment

```bash
# On H100 server:
git pull origin production-v1.0
# No config changes needed - VID_STRIDE already set to 3 in both configs
python orchestrator.py --poll --parallel 10 --tracking_mode bytetrack
```

### Verification

After deployment, check the new diagnostic logs to confirm:
1. `[Team] Found 2 goalkeeper(s)` - confirms GK detection working
2. `[Team] Skipping GK #X from team clustering` - confirms GK exclusion working
3. `[StatsEngine] Ball track: X% ` - shows ball detection quality
4. `[StatsEngine] Ownership: X%` - shows possession mapping quality
5. Shot counts should decrease significantly (no more 2.67 m/s false triggers)
6. Time-on-ball values should increase ~3x (now correctly calculated)
