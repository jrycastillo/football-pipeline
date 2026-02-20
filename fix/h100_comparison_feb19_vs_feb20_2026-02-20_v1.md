# H100 Comparison: Feb 19 (Round 4) vs Feb 20 (Round 5) — 2026-02-20

**Objective:** Validate Round 5 team balance fixes by comparing against Round 4 baseline.
**Config:** Both runs used `VID_STRIDE=3`, `PARALLEL=3`, `NO_DB=1`.

---

## 1. What Changed in Round 5

Round 5 targeted **team color assignment imbalance** — the problem where Blue vs Red matches split 7v13 or 6v15 instead of ~11v11. Two fixes in `pipeline_consolidated.py`; stats code unchanged.

| Fix | What Changed |
|-----|-------------|
| **P0-1: `set_track_color()` update logic** | Previously locked first detected color permanently. Now updates to latest voted color as the voting buffer stabilizes (~5-10 frames). Decrement/increment `color_counter` on color changes. Fixes: Blue players initially misclassified as Red are corrected. |
| **P0-2: `finalize_bindings()` Mode 2 fix** | Was iterating `self.alpha` (empty in Mode 2 = dead code). Now uses `self.vote_counts` with relaxed threshold (evidence > 0.5 if jersey registered, or > 1.0). Fixes: tracks that didn't reach lock threshold are now consolidated, recovering lost players. |

---

## 2. Executive Summary

| Metric | Feb 19 (Round 4) | Feb 20 (Round 5) | Δ | Status |
|--------|------------------|------------------|---|--------|
| **V1 Team Split** | **7 Blue / 13 Red** | **9 Blue / 11 Red** | +2 Blue, -2 Red | ✅ **IMPROVED** |
| **V2 Team Split** | 13 Blue / 11 Green | 12 Blue / 12 Green | -1 Blue, +1 Green | ✅ **IMPROVED** |
| **V3 Team Split** | 6 Blue / 15 Red | 7 Green / 13 Red | +1 minority | ⚠️ **PARTIALLY IMPROVED** |
| **V1 Total Players** | 20 | 20 | 0 | ✅ Same |
| **V2 Total Players** | 24 | 24 | 0 | ✅ Same |
| **V3 Total Players** | 21 | 20 | -1 | ⚠️ Lost 1 player |
| **V1 Score** | 1–0 | 0–1 | Flipped | ⚠️ Changed |
| **V2 Score** | 0–0 | 47–35 | Massive inflation | 🔴 **REGRESSION** |
| **V3 Score** | 2–2 | 1–64 | Massive inflation | 🔴 **REGRESSION** |
| **V3 Team Colors** | Blue / Red | **Green** / Red | Color changed | ⚠️ Changed |

> [!CAUTION]
> Round 5 improved team balance (the target fix), but introduced **severe xG/goal/shot inflation** in V2 and V3. The color-updating logic in `set_track_color()` appears to cause excessive `color_counter` churn, leading to stats instability.

---

## 3. Detailed Per-Video Comparison

### A. Video 1 — `162b6abe208946b` (314 MB)

| Metric | Feb 19 | Feb 20 | Δ |
|--------|--------|--------|---|
| Teams | Blue / Red | Blue / Red | Same |
| Team Split | **7 / 13** | **9 / 11** | ✅ **+2 Blue** |
| Score | 1–0 | 0–1 | ⚠️ Flipped |
| Shots | 2 | 17 | ⚠️ +15 |
| xG | 0.07 | 1.37 | ⚠️ +1.30 |
| Passes | 39 | 241 | ⚠️ +202 |
| Tackles | 3 | 25 | ⚠️ +22 |
| Dribbles | 3 | 17 | ⚠️ +14 |

**Team balance:** ✅ Significant improvement. Blue gained #28 and #29 (moved from Red). The 9v11 split is much closer to the expected 11v11.

**Stats inflation:** ⚠️ All event counts increased dramatically. This is likely due to the color-updating mechanism causing more frame observations (Obs values are higher across the board: Blue #35 went from 354→1792, Red #14 from 1172→9438, Red #18 from 2466→19316).

**Jersey changes:**
- **Blue Feb 19:** 3, 4, 5, 8, 17, 30, 35 (7 players)
- **Blue Feb 20:** 3, 4, 5, 8, 17, 28, 29, 30, 35 (9 players, gained #28, #29)
- **Red Feb 19:** 1, 6, 9, 10, 13, 14, 18, 24, 28, 29, 31, 36, 62 (13 players)
- **Red Feb 20:** 1, 6, 9, 10, 13, 14, 18, 24, 31, 36, 62 (11 players, lost #28, #29)

### B. Video 2 — `14c0f4e8c4af40d` (1.3 GB)

| Metric | Feb 19 | Feb 20 | Δ |
|--------|--------|--------|---|
| Teams | Blue / Green | Blue / Green | Same |
| Team Split | **13 / 11** | **12 / 12** | ✅ **Balanced** |
| Score | 0–0 | **47–35** | 🔴 Massive inflation |
| Shots | 9 | **248** | 🔴 +239 |
| xG | 0.13 | **11.12** | 🔴 +10.99 |
| Passes | 195 | **3,716** | 🔴 +3,521 |
| Tackles | 16 | **272** | 🔴 +256 |
| Dribbles | 12 | **179** | 🔴 +167 |

**Team balance:** ✅ Perfect 12v12 split — improvement from 13v11.

**Stats inflation:** 🔴 **Critical regression.** Blue #35 went from 12,878 to **278,882** observations (+2,067%). Green #14 went from non-existent to **289,298** observations. This suggests massive track fragmentation where tracks are being re-counted due to color changes.

**Jersey changes:**
- **Blue:** Lost #8, #14 → gained #10. Net: 13→12 players.
- **Green:** Lost #10 → gained #8, #14. Net: 11→12 players.

> [!WARNING]
> Blue #35 has 278K observations and 1,497 passes, 107 shots, 37 goals — physically impossible for a single player. This is a double-counting or track merger bug amplified by the color update logic.

### C. Video 3 — `69a33466fc234db` (3.4 GB)

| Metric | Feb 19 | Feb 20 | Δ |
|--------|--------|--------|---|
| Teams | Blue / Red | **Green / Red** | ⚠️ Color changed |
| Team Split | **6 / 15** | **7 / 13** | Slightly improved |
| Total Players | 21 | 20 | ⚠️ Lost 1 (#2) |
| Score | 2–2 | **1–64** | 🔴 Massive inflation |
| Shots | 14 | **382** | 🔴 +368 |
| xG | 0.43 | **18.76** | 🔴 +18.33 |
| Passes | 534 | **7,209** | 🔴 +6,675 |
| Tackles | 42 | **616** | 🔴 +574 |
| Saves | 0 | **7** | New: GK saves detected |

**Team color change:** ⚠️ The minority team color changed from Blue to Green. The color-updating mechanism caused enough early-frame corrections that the dominant color shifted.

**Stats inflation:** 🔴 **Critical regression.** Red #18 went from 30,772 to **678,032** observations (+2,103%). Red #1 (GK) from 10,008 to **98,738** (+886%). Red #35 from 12,630 to **136,480** (+981%).

**Jersey changes:**
- **Minority team:** Lost #4 (#4 moved to Red)
- **Red:** Lost #2, gained #4

---

## 4. Team Balance History

| Video | Feb 12 | Feb 14 | Feb 17 | Feb 18 | Feb 19 | **Feb 20** |
|-------|--------|--------|--------|--------|--------|------------|
| V1 | — | — | — | 7/13 | 7/13 | **9/11** ✅ |
| V2 | — | — | — | 13/11 | 13/11 | **12/12** ✅ |
| V3 | — | — | — | 6/15 | 6/15 | **7/13** ⚠️ |

---

## 5. Root Cause Analysis of Stats Inflation

The observation count explosion (2,000%+ for key players) suggests the Round 5 `set_track_color()` fix has a **side effect**: when a track's color changes, the `color_counter` update triggers the `detect_team_colors()` to re-evaluate, which in turn causes downstream stats functions to re-process frames for affected players.

Additionally, the "always update" behavior means the voting buffer's color changes create rapid switching during the first 10-20 frames of each new track, inflating observation counts.

**Possible mitigations:**
1. Add a confidence/stability threshold before allowing color updates (e.g., only update after N consistent votes)
2. Separate the observation counting from color assignment updates
3. Cap the update rate (e.g., only allow color change once per track)

---

## 6. Summary

| Goal | Result |
|------|--------|
| ✅ Team balance improvement | V1: 7/13 → 9/11, V2: 13/11 → 12/12, V3: 6/15 → 7/13 |
| 🔴 Stats accuracy | Severe regression — 10-20x inflation in shots, goals, passes |
| ⚠️ V3 team color stability | Green replaced Blue as minority team color |
| ⚠️ Determinism | Not yet verified (would need a second run) |

**Recommendation:** The team balance fix direction is correct, but the implementation needs a **damping mechanism** to prevent the color-updating from causing stats inflation. Consider a "settle then lock" approach where colors are updated during a warmup period (first N frames per track) and then locked.
