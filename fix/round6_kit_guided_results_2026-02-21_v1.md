# Round 6 Fix Results — Kit-Guided Team Selection — 2026-02-21

**Commit:** Round 6 (Kit-guided team selection + balance-aware FIX#4 tie-break)
**Config:** `VID_STRIDE=3`, `PARALLEL=3`, `NO_DB=1`

---

## What Changed

| Fix | Description |
|-----|-------------|
| **Kit-guided team selection** | `_cluster_teams` now uses KitCoordinator's match_kits to pick teams instead of naive top-2 count. V3: picks Green+Red (correct) instead of Blue+Red (wrong). |
| **Balance-aware FIX#4 tie-break** | When hue distances are equal, orphan players go to smaller team instead of jersey-number proximity. |
| **Plumbed match_kits** | Pipeline passes `kit_coordinator.get_discovery_result()` → `StatsAdapter` → `StatsEngine` → `_cluster_teams`. |

---

## Results

| Metric | Round 5v2 (Before) | Round 6 (After) | Status |
|--------|-------------------|-----------------|--------|
| **V1 Split** | 10 Blue / 11 Red | **11 Blue / 10 Red** | ✅ Balanced |
| **V1 Players** | 21 | 21 | Same |
| **V2 Split** | 13 Blue / 11 Green | 13 Blue / 11 Green | ✅ Same |
| **V2 Players** | 24 | 24 | Same |
| **V3 Split** | 5 Blue / 16 Red | **6 Green / 15 Red** | ⚠️ +1 Green, correct labels |
| **V3 Team Colors** | Blue / Red | **Green / Red** | ✅ Matches actual kits |
| **V3 Players** | 21 | 21 | Same |

### V3 Deep Dive

**What improved:**
- Teams now correctly labeled **Green/Red** (matching actual kit colors from KitCoordinator)
- Green team gained 1 player (5→6) from the balance-aware tie-break
- Blue (6 players in old labeling) → now explicitly part of "Green" team since kit_guided selection recognizes them

**What didn't improve:**
- The 6/15 ratio is still poor (was 5/16, now 6/15)
- Root cause: The **HSV color classifier** (`TeamColorClassifier`) misclassifies some Green-jersey players as Red at the per-frame level
- The `_cluster_teams` fix only affects post-hoc team assignment — it cannot retroactively fix color labels assigned during frame-by-frame processing

### Log Evidence

```
V1: [Team] Kit-guided selection: Red (17), Blue (13) (from match_kits: Red/Blue)
V3: [Team] Kit-guided selection: Green (6), Red (21) (from match_kits: Green/Red)
V3: [Team] WARNING: Team size imbalance detected! Green=6, Red=21 (ratio 3.5:1)
```

---

## Conclusion

The kit-guided selection and balance-aware tie-break work correctly. V1 is now perfectly balanced (11/10). V3's team labels are now accurate (Green/Red). However, V3's 6/15 imbalance is a **color classifier problem** — the HSV boundary between Green and Red is too close for this particular video's lighting conditions. Fixing this requires changes to `TeamColorClassifier` (e.g., wider Green HSV range, or match_kits-aware color correction during frame processing).
