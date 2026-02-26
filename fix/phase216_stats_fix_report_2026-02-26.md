# Phase 216 Stats Inflation Fix — 2026-02-26

**Commit:** Pending on `production-v1.0`
**File Changed:** `stats/metrics.py`

---

## Problem

Player stats (passes, distance, goals, shots) were massively inflated on full-length matches. The H100 Round 8 results showed:

| Video | Player | Obs | Passes | Goals | Shots | Issue |
|-------|--------|-----|--------|-------|-------|-------|
| V3 | Red #18 | 83,900 | 230 | 3 | 7 | Obs exceeds total frames (~60k) |
| V3 | White #35 | 55,386 | 270 | 9 | 24 | 9 goals from 1 player |
| V3 | White #4 | 54,656 | 381 | 1 | 14 | 520 seconds on ball |
| V2 | Green #35 | 27,708 | 74 | 8 | 11 | 8 goals from 1 player |

Match scores were V3: 10-16 (26 goals) and V2: 11-6 (17 goals) — completely unrealistic.

---

## Root Cause

**Phase 216 remap** in `stats/metrics.py` (lines 64-101) consolidates stats from ByteTrack track IDs to jersey numbers. When multiple tracks map to the same jersey, it **summed** all stats with `+=`:

```python
# THE BUG (old code):
for key, value in stats_dict.items():
    if isinstance(value, (int, float)):
        remapped_stats[target_id][key] += value  # <-- SUMS duplicates
```

### Why This Causes Inflation

ByteTrack creates 50+ track IDs per physical player across a full match (occlusion, camera cuts, reacquisition). When these tracks get mapped to the same jersey number:

```
Jersey #18 ← Track 101 (passes=5, distance=100m)
Jersey #18 ← Track 204 (passes=5, distance=100m)  ← SAME passes, re-counted
Jersey #18 ← Track 389 (passes=5, distance=100m)  ← SAME passes, re-counted again
Result: passes=15, distance=300m  ← 3x inflation
```

The tracks often overlap temporally or cover the same events, so summing them counts the same events multiple times.

---

## Fix: Pick Primary Track, Don't Sum

Changed Phase 216 from summing to **picking the primary track** (most data) per jersey:

```python
# NEW: Group tracks by jersey, pick the one with most data
jersey_candidates = defaultdict(list)

for track_id, stats_dict in raw_stats.items():
    jersey_num = track_to_jersey.get(track_id)
    if jersey_num is not None:
        weight = distance_m + touch_frames  # proxy for "most data"
        jersey_candidates[jersey_num].append((track_id, stats_dict, weight))

# For each jersey, pick the track with highest weight
for jersey_num, candidates in jersey_candidates.items():
    candidates.sort(key=lambda x: x[2], reverse=True)
    primary_stats = candidates[0][1]  # Best track only
    remapped_stats[jersey_num] = primary_stats  # No += summing
```

Also fixed the **frame count remap** which had the same summing bug — changed to `max()` instead of `+=`.

### What Changes

| Aspect | Before (sum) | After (pick primary) |
|--------|-------------|---------------------|
| Stats source | All tracks summed | Best track only |
| Frame counts | Summed across tracks | Max across tracks |
| Duplicate events | Counted N times | Counted once |
| Stats magnitude | Inflated 2-22x | Realistic |

---

## Local Verification

Tested on `121364_0.mp4` (30 seconds, 750 frames, Green vs Red):

```
[Phase 216] Jersey #36: picked track 8 (weight=255), dropped 1 duplicate track(s)
[Phase 216] Jersey #23: picked track 9 (weight=233), dropped 1 duplicate track(s)
[Phase 216] Jersey #18: picked track 14 (weight=310), dropped 1 duplicate track(s)
```

### Stats Comparison (Baseline vs Fixed)

| Player | Team | Obs Before | Obs After | Change | Passes B→A | Dist B→A |
|--------|------|-----------|-----------|--------|-----------|---------|
| #18 | Green | 4,392 | 1,476 | **-66%** | 20 → 13 | 405.9 → 218.5 |
| #17 | Green | 2,892 | 1,456 | -50% | 16 → 11 | 334.9 → 140.4 |
| #36 | Red | 2,970 | 1,490 | -50% | 14 → 8 | 367.5 → 191.5 |
| #23 | Green | 1,504 | 1,428 | -5% | 7 → 6 | 218.3 → 208.8 |
| #9 | Red | 1,428 | 1,428 | **0%** | 0 → 0 | 0 → 0 |
| #24 | Red | 1,480 | 1,480 | **0%** | 0 → 0 | 153.3 → 153.3 |

- Players with duplicate tracks (#18, #17, #36): stats dropped to realistic levels
- Players without duplicates (#9, #24, #25): **completely unchanged**
- Total passes: 104 → 85
- Goals: 1 → 1 (unchanged)
- Shots: 1 → 1 (unchanged)

---

## Expected Impact on H100 Full Matches

The inflation scales with match length (more ByteTrack fragments = more summing). For full-length matches:

| Metric | Before (inflated) | Expected After |
|--------|-------------------|----------------|
| V3 #18 observations | 83,900 | ~20,000-30,000 |
| V3 #35 goals | 9 | 1-3 |
| V3 match score | 10-16 (26 goals) | 2-4 total |
| V2 #35 goals | 8 | 1-2 |
| V2 match score | 11-6 (17 goals) | 1-3 total |
| Max passes per player | 381 | 50-80 |

---

## What This Does NOT Fix

1. **Player count** — Still depends on JNR lock + Phase 186 filter. The `finalize_bindings` consolidation can now safely be explored since stats won't inflate.
2. **Pass accuracy** — Depends on ball detection quality and ownership mapping, not Phase 216.
3. **Team balance** — Depends on color classifier and `_cluster_teams`, already fixed separately.
4. **Event detection logic** — Shots, tackles, dribbles are computed in `stats/event_logic.py`. If those algorithms over-count, that's a separate issue.
