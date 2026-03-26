# Round 20 Report: g5718885-fullhd.mp4

**Date:** 2026-03-26
**Video:** `test_videos/g5718885-fullhd.mp4` (Hamburger SV 2-2 Bayern Munich, Bundesliga R20, 31/01/2026)
**Match:** Hamburger SV (White) vs Bayern Munich (Black/dark kit)
**Final Score (Real):** HSV 2 - 2 Bayern
**Round:** R20 (goal rescue dedup + per-player cap)

---

## Changes in R20

### Fix: Phase 216 Goal Rescue Dedup (`stats/metrics.py`)
R19's goal rescue blindly summed goals from ALL remaining ByteTrack fragments, inflating Black team goals from 1 to 13. R20 applies two constraints:

1. **Rescue only if top-N found 0 goals** — if the top-5 fragments already captured goals, don't scan remaining fragments
2. **Rescue at most 1 goal** per player from minor fragments
3. **Hard cap of 2 goals per player** — catches over-counting in top-5 merge itself (where multiple fragments detect the same real goal)

---

## Processing Details

| Parameter | Value |
|-----------|-------|
| Video resolution | 1920x1080 (Full HD) |
| Duration | ~106 minutes (159,625 frames) |
| File size | 3.4 GB |
| FPS | 25 |
| VID_STRIDE | 3 |
| DET_IMG_SIZE | 640 |
| DET_CONF | 0.10 |
| Platform | Apple M4 Pro (MPS) |
| Processing time | ~4 hours 30 minutes |
| Frames processed | ~53,208 (stride=3) |
| Video output | Disabled (`--no_video_output`) |

---

## Pipeline Results Summary

| Metric | Baseline (R16) | R18 | R19 | **R20** | Wyscout (Real) | R20 Rate |
|--------|----------------|-----|-----|---------|----------------|----------|
| **Team colors** | White/Yellow | White/Black | White/Black | **White/Black** | White/Black | **100%** |
| **Players detected** | 35 | 26 | 26 | **26** | 28 (11+11 + 6 subs) | 93% |
| **Goals** | 4 (all Yellow) | 1 | 15 | **7** | 4 (2-2) | 175% (over) |
| **Shots on target** | 23 | 45 | 27 | **27** | 10 | 270% (over) |
| **Total passes** | 247 | 343 | 346 | **346** | 945 | 37% |
| **Pass accuracy** | 13.6% | 54.5% | 53.2% | **53.2%** | ~80% | - |
| **Tackles** | 49 | 28 | 28 | **28** | 5 sliding / 77 int | 560% / 36% |
| **Dribbles** | 14 | 36 | 35 | **36** | 53 | 68% |
| **Interceptions** | 94 | 98 | 97 | **98** | 77 | 127% |
| **xG** | 0.52 | 0.46 | 0.26 | **0.28** | 3.75 | 7% |
| **Fouls** | 0 | 0 | 0 | **0** | 28 | 0% |

### Score Detection

| | Pipeline | Real |
|---|---------|------|
| White (HSV) | 2 | 2 |
| Black (Bayern) | 5 | 2 |
| **Total** | **7** | **4** |

**Goal scorers (Pipeline):**

| Player | Team | Goals |
|--------|------|-------|
| #9 | Black | 1 |
| #14 | Black | 1 |
| #18 | Black | 1 |
| #36 | Black | 1 |
| #62 | Black | 1 |
| #29 | White | 1 |
| #30 | White | 1 |

**Real goal scorers:** #20 Vieira (HSV), #9 Kane (Bayern), #14 Diaz (Bayern), #44 Vuskovic (HSV)

**Analysis:** White goals (2) are correct. Black goals reduced from 13 (R19) to 5 (R20) thanks to per-player cap. However, 5 different Black players each detected 1 goal from their top-5 fragments. The per-player cap of 2 can't help here — the issue is that the same real goal event is detected by different nearby players' track fragments. A per-team cap or cross-player frame deduplication is needed.

Pipeline correctly identified #9 (Kane) and #14 (Diaz) as Black scorers — both are real goal scorers.

---

## Team Breakdown

### White Team (HSV) -- 15 players detected

| # | Pos | Obs | Pass | Acc | Acc% | Tck | SoT | Gls | Drb | Int | Dist | ToB(s) | xG |
|---|-----|-----|------|-----|------|-----|-----|-----|-----|-----|------|--------|-----|
| 1 | GK | 420 | 6 | 4 | 66.7 | 2 | 0 | 0 | 2 | 2 | 119.6 | 0.36 | 0.00 |
| 29 | P | 854 | 26 | 22 | 84.6 | 1 | 2 | 1 | 4 | 6 | 248.5 | 3.12 | 0.02 |
| 23 | P | 602 | 13 | 11 | 84.6 | 0 | 0 | 0 | 1 | 3 | 149.9 | 2.04 | 0.00 |
| 30 | P | 408 | 15 | 11 | 73.3 | 0 | 1 | 1 | 0 | 4 | 112.8 | 1.92 | 0.01 |
| 5 | P | 332 | 3 | 2 | 66.7 | 4 | 0 | 0 | 2 | 0 | 91.4 | 0.96 | 0.00 |
| 8 | P | 264 | 13 | 12 | 92.3 | 1 | 0 | 0 | 1 | 2 | 49.7 | 6.00 | 0.00 |
| 25 | P | 228 | 3 | 3 | 100.0 | 0 | 1 | 0 | 2 | 2 | 62.1 | 0.72 | 0.01 |
| 33 | P | 210 | 4 | 2 | 50.0 | 0 | 1 | 0 | 0 | 3 | 27.3 | 1.80 | 0.01 |
| 11 | P | 178 | 3 | 2 | 66.7 | 0 | 1 | 0 | 1 | 2 | 65.3 | 1.68 | 0.02 |
| 2 | P | 132 | 4 | 3 | 75.0 | 0 | 0 | 0 | 0 | 1 | 27.0 | 2.64 | 0.00 |
| 15 | P | 106 | 3 | 3 | 100.0 | 0 | 0 | 0 | 1 | 0 | 28.8 | 1.80 | 0.00 |
| 19 | P | 98 | 4 | 2 | 50.0 | 0 | 0 | 0 | 0 | 0 | 11.3 | 3.12 | 0.00 |
| 35 | P | 82 | 3 | 3 | 100.0 | 0 | 0 | 0 | 0 | 0 | 17.8 | 2.40 | 0.00 |
| 34 | P | 76 | 2 | 1 | 50.0 | 0 | 1 | 0 | 0 | 1 | 14.8 | 1.92 | 0.01 |
| 7 | P | 50 | 1 | 1 | 100.0 | 0 | 0 | 0 | 0 | 0 | 9.1 | 1.20 | 0.00 |
| **Total** | | **4,040** | **100** | **82** | **82.0** | **8** | **7** | **2** | **14** | **26** | **1,035.3** | **31.68** | **0.08** |

### Black Team (Bayern) -- 11 players detected

| # | Pos | Obs | Pass | Acc | Acc% | Tck | SoT | Gls | Drb | Int | Dist | ToB(s) | xG |
|---|-----|-----|------|-----|------|-----|-----|-----|-----|-----|------|--------|-----|
| 14 | P | 1122 | 55 | 37 | 67.3 | 2 | 2 | 1 | 5 | 9 | 328.8 | 24.00 | 0.02 |
| 9 | P | 1142 | 37 | 29 | 78.4 | 3 | 6 | 1 | 6 | 6 | 314.2 | 18.12 | 0.06 |
| 62 | P | 882 | 6 | 2 | 33.3 | 1 | 1 | 1 | 0 | 3 | 208.0 | 3.12 | 0.01 |
| 13 | P | 876 | 8 | 0 | 0.0 | 1 | 1 | 0 | 1 | 5 | 216.2 | 1.20 | 0.01 |
| 17 | P | 874 | 25 | 19 | 76.0 | 6 | 3 | 0 | 5 | 4 | 245.8 | 9.12 | 0.03 |
| 18 | P | 832 | 32 | 10 | 31.2 | 1 | 0 | 1 | 1 | 12 | 129.6 | 20.52 | 0.00 |
| 36 | P | 720 | 22 | 3 | 13.6 | 2 | 3 | 1 | 1 | 12 | 192.7 | 7.44 | 0.03 |
| 24 | P | 642 | 17 | 1 | 5.9 | 3 | 0 | 0 | 1 | 7 | 152.6 | 9.60 | 0.00 |
| 10 | P | 526 | 15 | 1 | 6.7 | 0 | 0 | 0 | 2 | 5 | 171.4 | 2.52 | 0.00 |
| 3 | P | 580 | 23 | 2 | 8.7 | 1 | 3 | 0 | 0 | 9 | 119.4 | 16.20 | 0.03 |
| 27 | P | 120 | 3 | 1 | 33.3 | 0 | 1 | 0 | 0 | 0 | 39.7 | 1.20 | 0.01 |
| **Total** | | **8,316** | **243** | **105** | **43.2** | **20** | **20** | **5** | **22** | **72** | **2,118.3** | **113.04** | **0.20** |

---

## Per-Team Comparison: Pipeline vs Wyscout

### HSV (White)

| Metric | Baseline | R18 | R19 | **R20** | Wyscout | R20 Rate |
|--------|----------|-----|-----|---------|---------|----------|
| Players | 19 | 15 | 15 | **15** | 16 (11+5 subs) | 94% |
| Passes | 66 | 103 | 100 | **100** | 269 | 37% |
| Pass accuracy | 13.6% | 79.6% | 82.0% | **82.0%** | 79% | **104%** |
| Shots on target | 5 | 10 | 7 | **7** | 4 | 175% |
| Goals | 0 | 0 | 2 | **2** | 2 | **100%** |
| Dribbles | 4 | 14 | 14 | **14** | 15 | **93%** |
| Interceptions | 37 | 26 | 26 | **26** | 38 | 68% |
| Tackles | 15 | 8 | 8 | **8** | 4 (sliding) | 200% |
| xG | 0.20 | 0.11 | 0.08 | **0.08** | 2.01 | 4% |

### Bayern (Black)

| Metric | Baseline | R18 | R19 | **R20** | Wyscout | R20 Rate |
|--------|----------|-----|-----|---------|---------|----------|
| Players | 16 | 11 | 11 | **11** | 14 (11+3 subs) | 79% |
| Passes | 181 | 243 | 243 | **243** | 676 | 36% |
| Pass accuracy | 63.0% | 43.2% | 43.2% | **43.2%** | 92% | - |
| Shots on target | 18 | 35 | 20 | **20** | 6 | 333% |
| Goals | 4 | 1 | 13 | **5** | 2 | 250% |
| Dribbles | 10 | 22 | 22 | **22** | 38 | 58% |
| Interceptions | 57 | 72 | 72 | **72** | 39 | 185% |
| Tackles | 34 | 20 | 20 | **20** | 1 (sliding) | 2000% |
| xG | 0.32 | 0.35 | 0.20 | **0.20** | 1.74 | 11% |

---

## Round-over-Round Goal Progress

| Round | White Goals | Black Goals | Total | Real |
|-------|------------|-------------|-------|------|
| Baseline (R16) | 0 | 4 | 4 | 4 (2-2) |
| R18 | 0 | 1 | 1 | 4 (2-2) |
| R19 | 2 | 13 | 15 | 4 (2-2) |
| **R20** | **2** | **5** | **7** | 4 (2-2) |

---

## Improvement Priorities (R21)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | **Black team goals still 5 vs 2 real** — 5 different players each detect 1 goal from same real events. Need cross-player frame dedup. | Critical | Medium |
| 2 | **Shot over-count** (27 vs 10 real) | High | Medium |
| 3 | **xG too low** (0.28 vs 3.75 real) | Medium | Medium |
| 4 | **Pass detection** (37% of real) | Medium | High |
| 5 | **Tackle over-count** (28 vs 5 sliding) | Low | Low |
