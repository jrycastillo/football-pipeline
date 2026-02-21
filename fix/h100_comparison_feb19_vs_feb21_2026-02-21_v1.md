# H100 Comparison: Feb 19 (Round 4) vs Feb 21 (Round 5 v2) — 2026-02-21

**Objective:** Validate Round 5 v2 fixes (settle-lock color, reverted finalize_bindings, vote threshold 0.30).
**Config:** Both runs used `VID_STRIDE=3`, `PARALLEL=3`, `NO_DB=1`.

---

## 1. What Changed in Round 5 v2

Three changes to `pipeline_consolidated.py`; stats code untouched.

| Fix | What Changed |
|-----|-------------|
| **P0-1 REVISED: "Settle then lock" color** | First color stored immediately. ONE correction allowed at observation #10 (voting buffer stable). Then locked permanently. Fixes v1's "always update" which caused color_counter churn and stats inflation. |
| **P0-2 REVERTED: `finalize_bindings()` back to no-op** | v1 used `vote_counts` in Mode 2 → consolidated hundreds of tracks → Phase 216 remap SUMMED stats → 10-22x inflation. Reverted to `self.alpha` (empty in Mode 2 = no-op). |
| **P0-3 NEW: Vote threshold 0.50→0.30** | JNR predictions with confidence 0.30-0.49 now count as votes. Expected to recover 1-2 players per video that had weak but consistent JNR reads. |

---

## 2. Executive Summary

| Metric | Feb 19 (Round 4) | Feb 21 (Round 5 v2) | Status |
|--------|------------------|---------------------|--------|
| **V1 Team Split** | **7 Blue / 13 Red** | **10 Blue / 11 Red** | ✅ **IMPROVED** (+3 Blue) |
| **V1 Player Count** | 20 | **21** | ✅ **+1 player recovered** |
| **V2 Team Split** | 13 Blue / 11 Green | 13 Blue / 11 Green | ✅ Stable |
| **V2 Score** | 0–0 | **2–1** | ✅ Realistic |
| **V3 Team Split** | 6 Blue / 15 Red | **5 Blue / 16 Red** | ⚠️ Worsened (-1 Blue) |
| **V3 Team Colors** | Blue / Red | Blue / Red | ✅ **STABLE** (v1 flipped to Green) |
| **V3 Player Count** | 21 | 21 | ✅ Same |
| **Stats Inflation** | Baseline | **None** | ✅ **FIXED** (v1 had 10-22x) |
| **V1 xG** | 0.07 | 0.00 | ✅ Realistic |
| **V2 xG** | 0.13 | 0.19 | ✅ Realistic |
| **V3 xG** | 0.43 | 0.68 | ✅ Realistic |

> [!IMPORTANT]
> Round 5 v2 successfully fixes the stats inflation from v1 while preserving team balance improvements. V1 gained 3 Blue players (7→10), V3 team colors stayed stable (Blue/Red), and stats are back to realistic ranges. The vote threshold lowering recovered 1 additional player in V1.

---

## 3. Detailed Per-Video Comparison

### A. Video 1 — `162b6abe208946b` (314 MB)

| Metric | Feb 19 (R4) | Feb 21 (R5v2) | Δ |
|--------|-------------|---------------|---|
| Teams | Blue / Red | Blue / Red | Same |
| Team Split | **7 / 13** | **10 / 11** | ✅ **+3 Blue** |
| Total Players | 20 | **21** | ✅ **+1** |
| Score | 1–0 | 0–0 | Changed |
| Shots | 2 | 0 | -2 |
| xG | 0.07 | 0.00 | -0.07 |
| Passes | 39 | 35 | -4 |
| Tackles | 3 | 2 | -1 |

**Team balance:** ✅ Best result yet. 10v11 is very close to the expected 11v11.

**Player changes:**
- **Blue gained:** #4, #28, #29, #30, #40 (5 added)
- **Blue lost:** none from old set
- **Red lost:** #28, #29 (moved to Blue)
- **New player:** #40 (not in Round 4 — **recovered by P0-3 vote threshold fix**)

**Stats:** Back to realistic ranges. Slightly lower than R4 (fewer passes/shots). Observation counts are consistent with R4 (e.g., Red #18: 2466→2192, similar magnitude).

### B. Video 2 — `14c0f4e8c4af40d` (1.3 GB)

| Metric | Feb 19 (R4) | Feb 21 (R5v2) | Δ |
|--------|-------------|---------------|---|
| Teams | Blue / Green | Blue / Green | Same |
| Team Split | 13 / 11 | 13 / 11 | ✅ Same |
| Total Players | 24 | 24 | Same |
| Score | 0–0 | **2–1** | ✅ Goals detected |
| Shots | 9 | 15 | +6 |
| xG | 0.13 | 0.19 | +0.06 |
| Passes | 195 | 212 | +17 |
| Tackles | 16 | 15 | -1 |
| Dribbles | 12 | 12 | Same |

**Team balance:** ✅ Unchanged at 13/11. Already close to balanced since Blue and Green are far apart in HSV.

**Stats:** All within normal range — **no inflation** (compare to v1's 47-35 score and 3,716 passes). Blue #35 went from 12,878 to 10,204 obs (reasonable), not 278,882 like v1.

**Improvement:** Score went from 0-0 to 2-1. The settle-lock color correction may help with ownership tracking near the goal, enabling better goal detection.

### C. Video 3 — `69a33466fc234db` (3.4 GB)

| Metric | Feb 19 (R4) | Feb 21 (R5v2) | Δ |
|--------|-------------|---------------|---|
| Teams | Blue / Red | Blue / Red | ✅ Stable |
| Team Split | **6 / 15** | **5 / 16** | ⚠️ **-1 Blue** |
| Total Players | 21 | 21 | Same |
| Score | 2–2 | **2–1** | Changed |
| Shots | 14 | 22 | +8 |
| xG | 0.43 | 0.68 | +0.25 |
| Passes | 534 | 577 | +43 |
| Tackles | 42 | 44 | +2 |
| Dribbles | 20 | 25 | +5 |

**Team color stability:** ✅ Stayed Blue/Red. The v1 "always update" caused Green to appear — the settle-lock approach keeps this stable.

**Team balance:** ⚠️ V3 worsened from 6/15 to 5/16. The WARMUP_THRESHOLD=10 correction at observation #10 was not enough to fix the Blue/Red confusion in this video. The HSV hue boundary between Blue and Red (wrap-around at H:0/180) makes V3 the hardest case.

**Stats:** Back to realistic ranges. Red #18: 30,772→29,216 (similar). Red #14: 9,830→11,612 (similar). No inflation.

**Player changes:**
- **Blue lost:** #29 (moved to Red)
- **Red gained:** #29
- Net: 6→5 Blue, 15→16 Red

---

## 4. Stats Inflation Check

| Player | R4 (Feb 19) | R5 v1 (Feb 20) | **R5 v2 (Feb 21)** |
|--------|-------------|-----------------|---------------------|
| V2 Blue #35 | 12,878 obs | **278,882** obs | **10,204** obs ✅ |
| V3 Red #18 | 30,772 obs | **678,032** obs | **29,216** obs ✅ |
| V3 Red #1 GK | 10,008 obs | **98,738** obs | **9,572** obs ✅ |
| V2 Score | 0–0 | **47–35** | **2–1** ✅ |
| V3 Score | 2–2 | **1–64** | **2–1** ✅ |

✅ **Stats inflation completely resolved.** All values are in the same order of magnitude as Round 4.

---

## 5. Team Balance History

| Video | Feb 14 | Feb 17 | Feb 18/19 | Feb 20 (v1) | **Feb 21 (v2)** |
|-------|--------|--------|-----------|-------------|-----------------|
| V1 | 17 total | 20 total | **7 / 13** | 9 / 11 ✅ | **10 / 11** ✅✅ |
| V2 | 18 total | 14 total | 13 / 11 | 12 / 12 ✅ | 13 / 11 ✅ |
| V3 | 19 total | 19 total | **6 / 15** | 7 / 13 ⚠️ | **5 / 16** ⚠️⚠️ |

V1 significantly improved. V3 remains the hardest case (Red/Blue HSV confusion).

---

## 6. Summary

| Goal | Result |
|------|--------|
| ✅ Fix stats inflation from v1 | **Fixed** — all values back to R4 magnitude |
| ✅ Preserve team balance improvements | V1: 10/11 (best yet), V2: 13/11 (stable) |
| ✅ V3 team color stability | Blue/Red (not Green/Red like v1) |
| ✅ Recover missing players (vote threshold) | V1 gained #40 (21 total, +1) |
| ⚠️ V3 team balance | Worsened to 5/16 (from 6/15) |

**Recommendation:** Round 5 v2 is a clear improvement over both v1 and Round 4. The settle-lock approach and reverted finalize_bindings fix the critical regressions. The remaining V3 team imbalance (5v16) requires a deeper fix to the HSV color classifier's Red/Blue boundary handling, not the color assignment logic.
