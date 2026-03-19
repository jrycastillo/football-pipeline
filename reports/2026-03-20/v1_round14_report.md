# Video 1 — Round 14 Stats Report

**Video:** `1e942fd8a6344bd` (1.3 GB, ~105 min, 30fps)
**Pipeline:** Round 14 | Branch `production-v1.0`
**Processed on:** Apple M4 Pro (MPS, stride=3)
**Date:** 2026-03-20

---

## R14 Changes from R13

1. **Strict observation filter lowered**: 3.0s → 1.5s (recover short-track White players)
2. **Tackle cooldown reduced**: 5s → 3s per player (more tackles detected)
3. **Tackle proximity widened**: 5m → 7m (stride=3 causes 6-9m movement between frames)
4. **Minimum ownership duration for passes**: 0.3s (~2-3 frames) filters rapid-switching noise
5. **Ghost player filter**: Remove players with >200 obs but 0 distance + 0 touches + 0 passes

---

## Match Summary

| Stat | Value |
|------|-------|
| Score | Green 3 — White 0 |
| Players Detected | 19 (Green 14 / White 5) |
| Total Passes | 125 |
| Total Tackles | 17 |
| Total Shots | 15 |
| Total Dribbles | 10 |
| Total xG | 0.72 |
| Goals | 3 |
| Interceptions | 65 |

### Team Green (14 players)

| # | Pos | Obs | Dist | Touch | ToB | G | S | xG | Drb | Pass | P% | Tkl | Int | Rec |
|---|-----|-----|------|-------|-----|---|---|----|-----|------|----|-----|-----|-----|
| 1 | GK | 266 | 49.8 | 31 | 3.72 | 0 | 2 | 0.02 | 0 | 10 | 30.0 | 1 | 5 | 1 |
| 2 | Player | 280 | 53.5 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 1 | 100.0 | 0 | 0 | 0 |
| 3 | Player | 684 | 201.2 | 55 | 6.6 | 0 | 2 | 0.02 | 0 | 5 | 60.0 | 0 | 3 | 3 |
| 4 | Player | 540 | 93.5 | 8 | 0.96 | 0 | 0 | 0.00 | 1 | 5 | 20.0 | 2 | 2 | 2 |
| 5 | Player | 734 | 112.8 | 137 | 16.44 | 2 | 4 | 0.31 | 1 | 22 | 31.8 | 2 | 9 | 8 |
| 8 | Player | 708 | 158.3 | 55 | 6.6 | 0 | 0 | 0.00 | 0 | 5 | 20.0 | 1 | 6 | 2 |
| 9 | Player | 948 | 229.8 | 42 | 5.04 | 1 | 1 | 0.01 | 1 | 16 | 31.2 | 2 | 6 | 4 |
| 13 | Player | 1166 | 331.9 | 9 | 1.08 | 0 | 1 | 0.01 | 0 | 5 | 20.0 | 1 | 9 | 8 |
| 15 | Player | 566 | 167.7 | 24 | 2.88 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 2 | 2 |
| 18 | Player | 842 | 235.3 | 15 | 1.8 | 0 | 2 | 0.07 | 2 | 13 | 23.1 | 2 | 9 | 12 |
| 27 | Player | 618 | 172.1 | 80 | 9.6 | 0 | 2 | 0.27 | 0 | 11 | 9.1 | 1 | 6 | 7 |
| 30 | Player | 290 | 49.2 | 33 | 3.96 | 0 | 0 | 0.00 | 0 | 1 | 0.0 | 0 | 1 | 1 |
| 38 | Player | 22 | 8.9 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| 44 | Player | 662 | 165.1 | 18 | 2.16 | 0 | 0 | 0.00 | 1 | 2 | 50.0 | 0 | 1 | 1 |

### Team White (5 players)

| # | Pos | Obs | Dist | Touch | ToB | G | S | xG | Drb | Pass | P% | Tkl | Int | Rec |
|---|-----|-----|------|-------|-----|---|---|----|-----|------|----|-----|-----|-----|
| 10 | Player | 480 | 142.2 | 32 | 3.84 | 0 | 0 | 0.00 | 0 | 2 | 100.0 | 0 | 0 | 0 |
| 14 | Player | 1322 | 373.8 | 59 | 7.08 | 0 | 1 | 0.01 | 3 | 21 | 85.7 | 4 | 4 | 6 |
| 25 | Player | 52 | 10.8 | 0 | 0.0 | 0 | 0 | 0.00 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| 29 | Player | 584 | 160.9 | 0 | 0.0 | 0 | 0 | 0.00 | 1 | 1 | 0.0 | 0 | 1 | 1 |
| 36 | Player | 286 | 53.1 | 90 | 10.8 | 0 | 0 | 0.00 | 0 | 5 | 100.0 | 1 | 1 | 1 |

---

## Round Comparison (R8 → R13 → R14)

| Stat | R8 Baseline (H100) | R13 (MPS) | R14 (MPS) | R14 vs R13 |
|------|---------------------|-----------|-----------|------------|
| Players | 27 (13/14) | 19 (13G/6W) | 19 (14G/5W) | Ghost removed, #38 recovered |
| Score | 2-1 | 3-0 (Green) | 3-0 (Green) | Same |
| Passes | 178 | 168 | **125** | -26% (noise filtered) |
| Tackles | 21 | 6 | **17** | **+183%** |
| Shots | 10 | 15 | **15** | Same |
| Dribbles | 5 | 10 | **10** | Same |
| xG | 0.50 | 0.72 | **0.72** | Same |
| Goals | 2 | 3 | **3** | Same |
| Interceptions | — | 71 | **65** | -8% |

### R14 Assessment

**Improvements:**
- **Tackles: 6 → 17** — Major recovery (2.8x). Close to R8's 21. 3s cooldown + 7m proximity worked.
- **Ghost #35 removed** — 1,138 obs / 0 stats artifact gone from output.
- **Pass accuracy improved** — Min-ownership filter removed noise. #14: 79% → 86%, #3: 50% → 60%.
- **#38 recovered** — New player (22 obs) passed the 1.5s filter.

**Regressions:**
- **Passes: 168 → 125** — Min-ownership filter (0.3s) removed noise but also some real short passes.
- **Team balance worse: 13G/6W → 14G/5W** — Ghost #35 (White) removed, #38 (Green) added. Net: White lost a player.

**Remaining issues:**
- Team imbalance (14G vs 5W) — fundamental color classification problem on MPS
- Player count (19) — same as R13, the 1.5s filter recovered #38 but ghost filter removed #35

---

## Known Issues

1. **Team Imbalance (14G vs 5W)**: White team severely under-detected. Root cause: White jerseys at S~55-68 boundary misclassified as Green by color classifier. Not a filtering issue — needs color classifier fix.
2. **Pass count dropped**: 168 → 125 from min-ownership filter. May need to lower from 0.3s to 0.2s.
3. **#38 low quality**: Only 22 obs, 8.9m distance, 0 touches — likely a brief sideline detection, not a real player.

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
| Int | Interceptions |
| Rec | Ball recoveries (opp half) |
