# Video 1 — Stats Report (R11 / R12 / R13)

**Video:** `1e942fd8a6344bd` (1.3 GB, ~105 min, 30fps)
**File:** f4dbcb0a_2026-03-09-22-46-45337194.webm

---

## Round 13 Results (Local MPS, stride=3)

**Pipeline:** Round 13 | Branch `production-v1.0`
**Processed on:** Apple M4 Pro (MPS, stride=3)
**Date:** 2026-03-19

**Changes from R12:**
- Phase 216 top-N merge (`_MERGE_TOP_N=5`): always sum events from top 5 fragments by weight instead of pick-primary
- Restored R8 cooldowns: Tackle 5s, Shot debounce 1s, Dribble 1s

### Match Summary

| Stat | Value |
|------|-------|
| Score | Green 3 — White 0 |
| Players Detected | 19 (Green 13 / White 6) |
| Total Passes | 168 |
| Total Tackles | 6 |
| Total Shots | 15 |
| Total Dribbles | 10 |
| Total xG | 0.72 |
| Goals | 3 |
| Saves | 0 |

### Team Green (13 players)

| # | Pos | Obs | Dist | Touch | ToB | G | S | xG | Drb | Pass | P% | Tkl | Chal | Int | Rec |
|---|-----|-----|------|-------|-----|---|---|----|-----|------|----|-----|------|-----|-----|
| 1 | GK | 266 | 49.8 | 31 | 3.72 | 0 | 2 | 0.02 | 0 | 12 | 33.3 | 0 | 0 | 6 | 1 |
| 2 | Player | 280 | 53.5 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 1 | 100.0 | 0 | 0 | 0 | 0 |
| 3 | Player | 684 | 201.2 | 55 | 6.6 | 0 | 2 | 0.02 | 0 | 6 | 50.0 | 0 | 0 | 3 | 3 |
| 4 | Player | 540 | 93.5 | 8 | 0.96 | 0 | 0 | 0.00 | 1 | 9 | 11.1 | 0 | 0 | 4 | 2 |
| 5 | Player | 734 | 112.8 | 137 | 16.44 | 2 | 4 | 0.31 | 1 | 23 | 34.8 | 1 | 0 | 10 | 8 |
| 8 | Player | 708 | 158.3 | 55 | 6.6 | 0 | 0 | 0.00 | 0 | 6 | 16.7 | 1 | 0 | 6 | 2 |
| 9 | Player | 948 | 229.8 | 42 | 5.04 | 1 | 1 | 0.01 | 1 | 22 | 31.8 | 0 | 0 | 8 | 4 |
| 13 | Player | 1166 | 331.9 | 9 | 1.08 | 0 | 1 | 0.01 | 0 | 11 | 18.2 | 0 | 0 | 10 | 8 |
| 15 | Player | 566 | 167.7 | 24 | 2.88 | 0 | 0 | 0.00 | 0 | 4 | 0.0 | 0 | 0 | 3 | 2 |
| 18 | Player | 842 | 235.3 | 15 | 1.8 | 0 | 2 | 0.07 | 2 | 18 | 16.7 | 0 | 1 | 12 | 12 |
| 27 | Player | 618 | 172.1 | 80 | 9.6 | 0 | 2 | 0.27 | 0 | 14 | 7.1 | 0 | 0 | 7 | 7 |
| 30 | Player | 290 | 49.2 | 33 | 3.96 | 0 | 0 | 0.00 | 0 | 1 | 0.0 | 0 | 1 | 1 | 1 |
| 44 | Player | 662 | 165.1 | 18 | 2.16 | 0 | 0 | 0.00 | 1 | 2 | 50.0 | 0 | 0 | 1 | 1 |

### Team White (6 players)

| # | Pos | Obs | Dist | Touch | ToB | G | S | xG | Drb | Pass | P% | Tkl | Chal | Int | Rec |
|---|-----|-----|------|-------|-----|---|---|----|-----|------|----|-----|------|-----|-----|
| 10 | Player | 480 | 142.2 | 32 | 3.84 | 0 | 0 | 0.00 | 0 | 3 | 100.0 | 0 | 0 | 0 | 0 |
| 14 | Player | 1322 | 373.8 | 59 | 7.08 | 0 | 1 | 0.01 | 3 | 29 | 79.3 | 4 | 2 | 6 | 6 |
| 25 | Player | 52 | 10.8 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 29 | Player | 584 | 160.9 | 0 | 0.0 | 0 | 0 | 0.00 | 1 | 1 | 0.0 | 0 | 0 | 1 | 1 |
| 35 | Player | 1138 | 0.0 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 36 | Player | 286 | 53.1 | 90 | 10.8 | 0 | 0 | 0.00 | 0 | 6 | 100.0 | 0 | 0 | 1 | 1 |

---

## Round 12 Results (Local MPS, stride=3)

**Pipeline:** Round 12 | Commit `c000bd91`
**Processed on:** Apple M4 Pro (MPS, stride=3)
**Date:** 2026-03-18

**Changes from R11:** Phase 216 smart merge threshold (<=5 fragments = sum, >5 = pick-primary), moderate cooldowns, orchestrator stride fix

### Match Summary

| Stat | Value |
|------|-------|
| Score | Green 2 — White 0 |
| Players Detected | 19 (Green 13 / White 6) |
| Total Passes | 55 |
| Total Tackles | 3 |
| Total Shots | 5 |
| Total Dribbles | 3 |
| Total xG | 0.57 |
| Saves | 0 |

### Team Green (13 players)

| # | Pos | Obs | Dist | Touch | ToB | G | S | xG | Drb | Pass | P% | Tkl | Chal | Int | Rec |
|---|-----|-----|------|-------|-----|---|---|----|-----|------|----|-----|------|-----|-----|
| 1 | GK | 266 | 49.8 | 31 | 3.72 | 0 | 0 | 0.00 | 0 | 1 | 0.0 | 0 | 0 | 1 | 0 |
| 2 | Player | 280 | 53.5 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 1 | 100.0 | 0 | 0 | 0 | 0 |
| 3 | Player | 684 | 201.2 | 55 | 6.6 | 0 | 1 | 0.01 | 0 | 4 | 50.0 | 0 | 0 | 2 | 2 |
| 4 | Player | 540 | 93.5 | 8 | 0.96 | 0 | 0 | 0.00 | 0 | 1 | 100.0 | 0 | 0 | 0 | 0 |
| 5 | Player | 734 | 112.8 | 137 | 16.44 | 2 | 3 | 0.30 | 0 | 10 | 30.0 | 1 | 0 | 6 | 6 |
| 8 | Player | 708 | 158.3 | 55 | 6.6 | 0 | 0 | 0.00 | 0 | 3 | 0.0 | 1 | 0 | 2 | 1 |
| 9 | Player | 948 | 229.8 | 42 | 5.04 | 0 | 0 | 0.00 | 0 | 2 | 50.0 | 0 | 0 | 2 | 2 |
| 13 | Player | 1166 | 331.9 | 9 | 1.08 | 0 | 0 | 0.00 | 0 | 3 | 0.0 | 0 | 0 | 2 | 2 |
| 15 | Player | 566 | 167.7 | 24 | 2.88 | 0 | 0 | 0.00 | 0 | 4 | 0.0 | 0 | 0 | 3 | 2 |
| 18 | Player | 842 | 235.3 | 15 | 1.8 | 0 | 0 | 0.00 | 1 | 3 | 0.0 | 0 | 1 | 2 | 2 |
| 27 | Player | 618 | 172.1 | 80 | 9.6 | 0 | 1 | 0.26 | 0 | 11 | 9.1 | 0 | 0 | 3 | 3 |
| 30 | Player | 290 | 49.2 | 33 | 3.96 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 1 | 0 | 0 |
| 44 | Player | 662 | 165.1 | 18 | 2.16 | 0 | 0 | 0.00 | 1 | 1 | 100.0 | 0 | 0 | 0 | 0 |

### Team White (6 players)

| # | Pos | Obs | Dist | Touch | ToB | G | S | xG | Drb | Pass | P% | Tkl | Chal | Int | Rec |
|---|-----|-----|------|-------|-----|---|---|----|-----|------|----|-----|------|-----|-----|
| 10 | Player | 480 | 142.2 | 32 | 3.84 | 0 | 0 | 0.00 | 0 | 2 | 100.0 | 0 | 0 | 0 | 0 |
| 14 | Player | 1322 | 373.8 | 59 | 7.08 | 0 | 0 | 0.00 | 1 | 6 | 83.3 | 1 | 1 | 2 | 2 |
| 25 | Player | 52 | 10.8 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 29 | Player | 584 | 160.9 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 35 | Player | 1138 | 0.0 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 36 | Player | 286 | 53.1 | 90 | 10.8 | 0 | 0 | 0.00 | 0 | 3 | 100.0 | 0 | 0 | 0 | 0 |

---

## Round 11 Results (H100 GPU, stride=3)

**Pipeline:** Round 11 | Commit `74d65cd`
**Processed on:** H100 GPU (stride=3)
**Date:** 2026-03-14

### Match Summary

| Stat | Value |
|------|-------|
| Score | Green 0 — White 0 |
| Players Detected | 30 (Green 19 / White 11) |
| Total Passes | 96 |
| Total Tackles | 9 |
| Total Shots | 7 |
| Total Dribbles | 2 |
| Total xG | 0.07 |
| Saves | 0 |

### Team Green (19 players)

| # | Pos | Obs | Dist | Touch | ToB | G | S | xG | Drb | Pass | P% | Tkl | Chal | Int | Rec |
|---|-----|-----|------|-------|-----|---|---|----|-----|------|----|-----|------|-----|-----|
| 1 | GK | 1598 | 222.0 | 243 | 29.16 | 0 | 7 | 0.07 | 1 | 16 | 68.8 | 0 | 0 | 6 | 2 |
| 2 | Player | 160 | 7.6 | 28 | 3.36 | 0 | 1 | 0.01 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 3 | Player | 5930 | 920.4 | 46 | 5.52 | 0 | 0 | 0.00 | 0 | 6 | 50.0 | 1 | 0 | 0 | 0 |
| 4 | Player | 1166 | 113.0 | 122 | 14.64 | 0 | 0 | 0.00 | 0 | 6 | 83.3 | 3 | 0 | 4 | 4 |
| 5 | Player | 1792 | 354.2 | 92 | 11.04 | 0 | 0 | 0.00 | 0 | 3 | 33.3 | 0 | 0 | 0 | 0 |
| 8 | Player | 2376 | 475.4 | 167 | 20.04 | 0 | 0 | 0.00 | 0 | 6 | 16.7 | 1 | 0 | 4 | 3 |
| 10 | Player | 796 | 45.9 | 58 | 6.96 | 0 | 0 | 0.00 | 0 | 2 | 0.0 | 0 | 0 | 2 | 0 |
| 11 | Player | 230 | 0 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 15 | Player | 746 | 111.1 | 40 | 4.8 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 1 | 0 |
| 17 | Player | 114 | 22.8 | 46 | 5.52 | 0 | 0 | 0.00 | 0 | 2 | 50.0 | 0 | 0 | 1 | 0 |
| 18 | Player | 2270 | 435.6 | 102 | 12.24 | 0 | 0 | 0.00 | 0 | 3 | 33.3 | 2 | 1 | 1 | 1 |
| 25 | Player | 134 | 28.1 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 27 | Player | 1006 | 0.0 | 2 | 0.24 | 0 | 0 | 0.00 | 0 | 1 | 0.0 | 0 | 0 | 1 | 1 |
| 29 | Player | 542 | 45.6 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 30 | Player | 1126 | 139.4 | 59 | 7.08 | 0 | 1 | 0.01 | 0 | 4 | 100.0 | 1 | 0 | 1 | 1 |
| 34 | Player | 246 | 0 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 35 | Player | 4238 | 674.5 | 171 | 20.52 | 0 | 0 | 0.00 | 1 | 10 | 60.0 | 0 | 0 | 2 | 2 |
| 44 | Player | 474 | 67.0 | 6 | 0.72 | 0 | 0 | 0.00 | 0 | 1 | 100.0 | 0 | 0 | 0 | 0 |
| 62 | Player | 70 | 12.4 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |

### Team White (11 players)

| # | Pos | Obs | Dist | Touch | ToB | G | S | xG | Drb | Pass | P% | Tkl | Chal | Int | Rec |
|---|-----|-----|------|-------|-----|---|---|----|-----|------|----|-----|------|-----|-----|
| 6 | Player | 136 | 15.2 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 9 | Player | 2820 | 389.3 | 320 | 38.4 | 0 | 0 | 0.00 | 0 | 14 | 14.3 | 0 | 0 | 10 | 5 |
| 13 | Player | 3530 | 570.1 | 76 | 9.12 | 0 | 0 | 0.00 | 0 | 12 | 66.7 | 0 | 0 | 4 | 3 |
| 14 | Player | 3060 | 245.6 | 582 | 69.84 | 0 | 0 | 0.00 | 0 | 5 | 40.0 | 1 | 0 | 1 | 0 |
| 22 | Player | 222 | 18.3 | 36 | 4.32 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 23 | Player | 38 | 2.6 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 24 | Player | 612 | 137.1 | 2 | 0.24 | 0 | 0 | 0.00 | 0 | 1 | 100.0 | 0 | 0 | 0 | 0 |
| 33 | Player | 244 | 0 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 36 | Player | 736 | 74.2 | 190 | 22.8 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 38 | Player | 316 | 59.5 | 32 | 3.84 | 0 | 1 | 0.01 | 0 | 4 | 100.0 | 1 | 0 | 1 | 0 |
| 40 | Player | 240 | 57.1 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 |

---

## Round Comparison (R8 baseline → R11 → R12 → R13)

| Stat | R8 Baseline (H100) | R11 (H100) | R12 (MPS) | R13 (MPS) |
|------|---------------------|------------|-----------|-----------|
| Players | 27 (13/14) | 30 (19G/11W) | 19 (13G/6W) | 19 (13G/6W) |
| Score | 2-1 (Red) | 0-0 | 2-0 (Green) | 3-0 (Green) |
| Passes | 178 | 96 | 55 | **168** |
| Tackles | 21 | 9 | 3 | **6** |
| Shots | 10 | 7 | 5 | **15** |
| Dribbles | 5 | 2 | 3 | **10** |
| xG | 0.50 | 0.07 | 0.57 | **0.72** |
| Goals | 2 | 0 | 2 | **3** |

**Note:** R8 used a different video file (`162b6abe`, 314MB) vs R11-R13 (`1e942fd8a6344bd`, 1.3GB). R8 also used the old Phase 216 (sum-all fragments) and R8 cooldowns. Direct comparison is approximate.

### R13 Assessment

R13 is a major improvement over R12:
- **Passes**: 55 → 168 (3.1x improvement, nearly matches R8's 178)
- **Shots**: 5 → 15 (3x improvement, exceeds R8's 10)
- **Dribbles**: 3 → 10 (3.3x improvement, double R8's 5)
- **Tackles**: 3 → 6 (2x improvement, but still below R8's 21)

The Phase 216 top-N merge recovered the vast majority of lost events. R8 cooldowns further boosted tackles/dribbles. On H100 with better detection, R13 is expected to match or exceed R8 across all categories.

**Remaining gap — Tackles (6 vs R8's 21):** MPS produces fewer detections and shorter proximity windows, directly limiting tackle detection. H100 should close most of this gap.

---

## Known Issues

1. **Team Imbalance**: 13 vs 6 players (expected ~11 vs 11). White team severely under-detected on MPS. Many White players likely merged into a few jersey IDs or misclassified to Green.
2. **Ghost Player #35 (White)**: 1,138 observations but 0 distance, 0 touches, 0 everything. All fragments failed to produce stats — likely a detection artifact or duplicate jersey.
3. **Low Tackle Count**: 6 total (all from 3 players: #5, #8 Green + #14 White). MPS has shorter track durations and fewer proximity detections.
4. **Pass Accuracy Low**: Many players show <30% pass accuracy. Ownership mapping in crowded areas causes false "inaccurate" passes.
5. **MPS vs CUDA Gap**: Local MPS produces ~40-50% fewer raw events than H100. R13 numbers should be multiplied by ~1.5-2x when estimating H100 output.

---

## Next Steps

See `reports/2026-03-18/next_steps.md` for detailed improvement plan.

---

## Legend
| Col | Meaning |
|-----|---------|
| # | Jersey number |
| Pos | Position |
| Obs | Frames seen |
| Dist | Distance (m) |
| Touch | Ball touches |
| ToB | Time on ball (s) |
| G | Goals |
| S | Shots |
| xG | Expected Goals |
| Drb | Dribbles |
| Pass | Passes |
| P% | Pass accuracy (%) |
| Tkl | Tackles |
| Chal | Challenges |
| Int | Interceptions |
| Rec | Ball recoveries (opp half) |
