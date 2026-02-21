# Round 6 Pipeline Fixes — Player Detection & Team Balance — 2026-02-21

## Summary

Round 6 addresses two critical issues:
1. **Low player detection** — only 21-24 players detected from 22+ on the pitch
2. **V3 team imbalance** — 5 Blue / 16 Red (should be ~11/11)

Applied in two sub-rounds:
- **Round 6.0**: Kit-guided team selection + balance-aware orphan merge
- **Round 6.1**: Relaxed JNR lock thresholds + kit-aware color correction

---

## Root Cause Analysis

### Player Loss Chain

The pipeline creates thousands of ByteTrack track IDs (V3: 6,151) but Phase 186 drops all without jersey numbers, leaving only ~21. The bottleneck is the JNR lock condition:

```
Lock requires: 3 votes AND margin ≥ 1.0
Soft-registration requires: score ≥ 1.5
```

JNR detects ~30 unique jerseys but many fail to lock (e.g., #7, #11, #15, #25, #27, #33 all predicted but never reach 3 votes with 1.0 margin).

### V3 Team Imbalance

`_cluster_teams` in `stats/metrics.py` picks top-2 colors by count: Red(21) + Blue(6), orphaning Green(6). FIX#4 merges Green→Red (equal hue distance 60 to both, jersey-number tie-break picks Red). Result: 5 Blue / 16+6=22 Red.

---

## Changes Made

### Round 6.0 — Kit-Guided Team Selection

#### [MODIFY] `stats/metrics.py` — `_cluster_teams()`

**Change 1: Kit-guided team selection**

Uses `KitCoordinator.get_discovery_result()` (match_kits) to pick teams instead of naive top-2 by count. If KitCoordinator discovered 2 player kit colors, use those as team A/B.

```python
# Before: always top-2 by count
top_2 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]

# After: prefer kit-discovered colors
if match_kits and len(match_kits.get("players", [])) == 2:
    kit_a_norm = color_merge_map.get(kit_a_raw.lower(), ...).capitalize()
    kit_b_norm = color_merge_map.get(kit_b_raw.lower(), ...).capitalize()
    if kit_a_norm in counts and kit_b_norm in counts:
        team_a_color, team_b_color = kit_a_norm, kit_b_norm  # Kit-guided
```

**Change 2: Balance-aware FIX#4 tie-break**

When hue distances are equal, assign orphan players to the **smaller team** instead of jersey-number proximity:

```python
# Before: jersey-number proximity (arbitrary)
orphan_avg = sum(int(j) for j in orphan_jerseys) / len(orphan_jerseys)
nearest = team_a_color if abs(orphan_avg - avg_a) < abs(orphan_avg - avg_b) else team_b_color

# After: balance-aware
size_a = len(team_a_jerseys)
size_b = len(team_b_jerseys)
nearest = team_b_color if size_a > size_b else team_a_color
```

**Change 3: Thread `match_kits` parameter**

Added `match_kits` parameter to `process_events()` → `_cluster_teams()`.

#### [MODIFY] `pipeline_consolidated.py`

Pass `kit_coordinator.get_discovery_result()` through `StatsAdapter.process_events()`:

```python
kits = kit_coordinator.get_discovery_result()
raw_tracks, player_stats = stats_adapter.process_events(all_frames, id_manager, match_kits=kits)
```

---

### Round 6.1 — Relaxed Lock Thresholds + Kit-Aware Color

#### [MODIFY] `pipeline_consolidated.py` — `IdentityManager.process_detection()`

**Change 1: Lower lock thresholds**

```diff
-# Lock Rule: 3 Votes required, margin >= 1.0
-if best_tally >= 3 and (best_score - second_score) >= 1.0:
+# Lock Rule (Round 6.1): 2 Votes required, margin >= 0.5
+if best_tally >= 2 and (best_score - second_score) >= 0.5:
```

Rationale: A player with 2 consistent reads and clear winner is reliable. `try_lock()` still enforces global uniqueness.

**Change 2: Lower soft-registration threshold**

```diff
-if best_score >= 1.5 and best_num not in self.jersey_registry:
+if best_score >= 0.6 and best_num not in self.jersey_registry:
```

Rationale: 0.6 = two reads at 0.30 confidence each. Gets jerseys into registry faster, rescuing them from Phase 186 Unknown filter.

**Change 3: Periodic kit propagation**

After 500 processed frames, propagate discovered kit colors to `TeamColorClassifier`:

```python
if color_classifier.known_kit_colors is None and n >= 500:
    _kits = kit_coordinator.get_discovery_result()
    if len(_kits.get("players", [])) == 2:
        color_classifier.known_kit_colors = _kits["players"]
```

#### [MODIFY] `vision/color_classifier.py` — `TeamColorClassifier.predict()`

**Change 4: Kit-aware color correction**

When `known_kit_colors` is set and predicted color is NOT a kit color, check if hue is within 30° of a kit color and correct:

```python
if self.known_kit_colors and color_name not in self.known_kit_colors:
    # Find closest kit color by hue distance
    # Correct if distance <= 30° (adjacent color)
    if best_kit and best_dist <= 30:
        color_name = best_kit
```

This fixes V3 where Green-jersey players get classified as Red due to HSV boundary proximity.

---

## Results

### Player Count (Primary Goal)

| Video | Round 5v2 | Round 6.0 | **Round 6.1** | Total Gain |
|-------|-----------|-----------|---------------|------------|
| V1 | 21 | 21 | **25** | **+4** |
| V2 | 24 | 24 | **28** | **+4** |
| V3 | 21 | 21 | **29** | **+8** |

### New Players Recovered

- **V1**: `#7, #11, #23, #25` (21→25)
- **V2**: `#2, #15, #16, #33, #50` (24→28)
- **V3**: `#15, #17, #25, #27, #28, #33, #44, #55` (21→29)

### Team Balance

| Video | Round 5v2 | Round 6.0 | Round 6.1 |
|-------|-----------|-----------|-----------|
| V1 | 10/11 ✅ | 11/10 ✅ | 21/10 ⚠️ |
| V2 | 13/11 ✅ | 13/11 ✅ | 11/15 ✅ |
| V3 | 5/16 ❌ | 6/15 ⚠️ | 8/20 ⚠️ |

### Log Evidence

```
V1: Kit-guided selection: Red (21), Blue (10) (from match_kits: Red/Blue)
V1: Filtered Unknown players: 485 -> 25
V1: Identified Players: ['1','3','4','5','6','7','8','9','10','11','13','14','17','18',
    '23','24','25','28','29','30','31','35','36','40','62']

V2: Kit-guided selection: Green (11), Blue (15) (from match_kits: Green/Blue)
V2: Identified Players: 28 total

V3: Kit-guided selection: Yellow (8), Red (20) (from match_kits: Yellow/Red)
V3: Identified Players: 29 total
```

---

## Known Issues

1. **V1/V3 team balance** — new players added but some go disproportionately to one team. The color classifier still misclassifies borderline hues.
2. **V3 kit color shift** — Kit discovery at frame 500 detected "Yellow" instead of "Green" (early frame distribution differs). Later frames would detect Green but the one-shot propagation locks to the early result.

## Files Modified

| File | Changes |
|------|---------|
| `stats/metrics.py` | Kit-guided team selection, balance-aware FIX#4 tie-break, `match_kits` parameter |
| `pipeline_consolidated.py` | Lock thresholds (2/0.5), soft-reg (0.6), kit propagation, match_kits plumbing |
| `vision/color_classifier.py` | `known_kit_colors` field, kit-aware hue correction (±30°) |
