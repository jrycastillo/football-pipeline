# Round 18 Report: g5718885-fullhd.mp4

**Date:** 2026-03-24
**Video:** `test_videos/g5718885-fullhd.mp4` (Hamburger SV 2-2 Bayern Munich, Bundesliga R20, 31/01/2026)
**Match:** Hamburger SV (White) vs Bayern Munich (Black/dark kit)
**Final Score (Real):** HSV 2 - 2 Bayern
**Round:** R18 (color fix round — Bayern dark kit classified as Black instead of Green/Red)

---

## Changes in R18

### Fix 1: Green Early-Exit Guard (`vision/color_classifier.py`)
Dark jerseys with grass background triggered the green early-exit (>60% green in center crop). Added a dark pixel ratio check: if >15% of the center crop is dark (S<70, V<100), skip the green exit and continue to normal HSV classification.

### Fix 2: Black Threshold Raised (`vision/color_classifier.py`)
`_classify_hsv()` Black threshold raised from V<60 to V<80. Dark jerseys with V 60-80 were being classified as White instead of Black.

### Fix 3: Grass Mask Extended (`vision/color_classifier.py`)
Grass mask H lower bound extended from H>=35 to H>=30. Yellowish-green grass at H 30-34 was leaking through and classifying as "Yellow".

### Fix 4: V-Histogram Mode (`vision/color_classifier.py`)
Replaced `np.median()` with V-channel histogram mode for achromatic path. Dark jerseys have mixed pixels: dark fabric (V~30), skin (V~160), white trim (V~200). Median gets pulled to V~130 (White). Histogram mode (6 bins) finds the dominant V band, correctly identifying dark fabric.

### Fix 5: Partial Kit-Guided Selection (`stats/metrics.py`)
When one kit color is present in `player_colors` but the other is missing (e.g., "Black" has 0 locked players), trust `match_kits.json` discovery and create an empty entry for the missing color. Orphan players get assigned to the underrepresented team.

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
| Processing time | ~4 hours 29 minutes |
| Frames processed | ~53,208 (stride=3) |

---

## Pipeline Results Summary

| Metric | Baseline (R16) | R17 | **R18** | Wyscout (Real) | R18 Rate |
|--------|----------------|-----|---------|----------------|----------|
| **Team colors** | White/Yellow | White/Green | **White/Black** | White/Black | **100%** |
| **Players detected** | 35 | 26 | **26** | 28 (11+11 + 6 subs) | 93% |
| **Goals** | 4 (all Yellow) | 3 | **1** | 4 (2-2) | 25% |
| **Shots on target** | 23 | 42 | **45** | 10 | 450% (over) |
| **Total passes** | 247 | ~300 | **343** | 945 | 36% |
| **Pass accuracy** | 13.6% | - | **54.5%** | ~80% | - |
| **Tackles** | 49 | 28 | **28** | 5 sliding / 77 int | 560% / 36% |
| **Dribbles** | 14 | - | **36** | 53 | 68% |
| **Interceptions** | 94 | - | **98** | 77 | 127% |
| **xG** | 0.52 | 0.42 | **0.46** | 3.75 | 12% |
| **Fouls** | 0 | 0 | **0** | 28 | 0% |

### Score Detection

| | Pipeline | Real |
|---|---------|------|
| White (HSV) | 0 | 2 |
| Black (Bayern) | 1 | 2 |
| **Total** | **1** | **4** |

Goal log shows 8 goal-correction events. Many shots near the goal are flagged but fail final confirmation. Only 1 goal confirmed (Player #9, Black).

---

## Team Breakdown

### White Team (HSV) — 15 players detected

| # | Pos | Obs | Pass | Acc | Acc% | Tck | SoT | Gls | Drb | Int | Dist | ToB(s) | xG |
|---|-----|-----|------|-----|------|-----|-----|-----|-----|-----|------|--------|-----|
| 1 | GK | 420 | 6 | 4 | 66.7 | 2 | 0 | 0 | 2 | 2 | 119.6 | 0.36 | 0.00 |
| 29 | P | 854 | 26 | 22 | 84.6 | 1 | 4 | 0 | 4 | 6 | 248.5 | 3.12 | 0.04 |
| 23 | P | 602 | 13 | 11 | 84.6 | 0 | 0 | 0 | 1 | 3 | 149.9 | 2.04 | 0.00 |
| 30 | P | 408 | 15 | 11 | 73.3 | 0 | 1 | 0 | 0 | 4 | 112.8 | 1.92 | 0.01 |
| 5 | P | 332 | 3 | 2 | 66.7 | 4 | 0 | 0 | 2 | 0 | 91.4 | 0.96 | 0.00 |
| 8 | P | 264 | 13 | 12 | 92.3 | 1 | 1 | 0 | 1 | 2 | 49.7 | 6.00 | 0.01 |
| 25 | P | 228 | 3 | 3 | 100.0 | 0 | 1 | 0 | 2 | 2 | 62.1 | 0.72 | 0.01 |
| 33 | P | 210 | 4 | 2 | 50.0 | 0 | 1 | 0 | 0 | 3 | 27.3 | 1.80 | 0.01 |
| 11 | P | 178 | 3 | 2 | 66.7 | 0 | 1 | 0 | 1 | 2 | 65.3 | 1.68 | 0.02 |
| 2 | P | 132 | 4 | 3 | 75.0 | 0 | 0 | 0 | 0 | 1 | 27.0 | 2.64 | 0.00 |
| 15 | P | 106 | 3 | 3 | 100.0 | 0 | 0 | 0 | 1 | 0 | 28.8 | 1.80 | 0.00 |
| 19 | P | 98 | 4 | 2 | 50.0 | 0 | 0 | 0 | 0 | 0 | 11.3 | 3.12 | 0.00 |
| 35 | P | 82 | 3 | 3 | 100.0 | 0 | 0 | 0 | 0 | 0 | 17.8 | 2.40 | 0.00 |
| 34 | P | 76 | 2 | 1 | 50.0 | 0 | 1 | 0 | 0 | 1 | 14.8 | 1.92 | 0.01 |
| 7 | P | 50 | 1 | 1 | 100.0 | 0 | 0 | 0 | 0 | 0 | 9.1 | 1.20 | 0.00 |
| **Total** | | **4,040** | **103** | **82** | **79.6** | **8** | **10** | **0** | **14** | **26** | **1,035.3** | **31.68** | **0.11** |

### Black Team (Bayern) — 11 players detected

| # | Pos | Obs | Pass | Acc | Acc% | Tck | SoT | Gls | Drb | Int | Dist | ToB(s) | xG |
|---|-----|-----|------|-----|------|-----|-----|-----|-----|-----|------|--------|-----|
| 14 | P | 1122 | 55 | 37 | 67.3 | 2 | 3 | 0 | 5 | 9 | 328.8 | 24.00 | 0.03 |
| 9 | P | 1142 | 37 | 29 | 78.4 | 3 | 10 | 1 | 6 | 6 | 314.2 | 18.12 | 0.10 |
| 62 | P | 882 | 6 | 2 | 33.3 | 1 | 2 | 0 | 0 | 3 | 208.0 | 3.12 | 0.02 |
| 13 | P | 876 | 8 | 0 | 0.0 | 1 | 1 | 0 | 1 | 5 | 216.2 | 1.20 | 0.01 |
| 17 | P | 874 | 25 | 19 | 76.0 | 6 | 4 | 0 | 5 | 4 | 245.8 | 9.12 | 0.04 |
| 18 | P | 832 | 32 | 10 | 31.2 | 1 | 2 | 0 | 1 | 12 | 129.6 | 20.52 | 0.02 |
| 36 | P | 720 | 22 | 3 | 13.6 | 2 | 4 | 0 | 1 | 12 | 192.7 | 7.44 | 0.04 |
| 24 | P | 642 | 17 | 1 | 5.9 | 3 | 3 | 0 | 1 | 7 | 152.6 | 9.60 | 0.03 |
| 3 | P | 580 | 23 | 2 | 8.7 | 1 | 5 | 0 | 0 | 9 | 119.4 | 16.20 | 0.05 |
| 10 | P | 526 | 15 | 1 | 6.7 | 0 | 0 | 0 | 2 | 5 | 171.4 | 2.52 | 0.00 |
| 27 | P | 120 | 3 | 1 | 33.3 | 0 | 1 | 0 | 0 | 0 | 39.7 | 1.20 | 0.01 |
| **Total** | | **8,316** | **243** | **105** | **43.2** | **20** | **35** | **1** | **22** | **72** | **2,118.3** | **113.04** | **0.35** |

---

## Per-Team Comparison: Pipeline vs Wyscout

### HSV (White)

| Metric | Baseline | R17 | **R18** | Wyscout | R18 Rate |
|--------|----------|-----|---------|---------|----------|
| Players | 19 | 14 | **15** | 16 (11+5 subs) | 94% |
| Passes | 66 | - | **103** | 269 | 38% |
| Pass accuracy | 13.6% | - | **79.6%** | 79% | **100%** |
| Shots on target | 5 | - | **10** | 4 | 250% |
| Goals | 0 | - | **0** | 2 | **0%** |
| Dribbles | 4 | - | **14** | 15 | **93%** |
| Interceptions | 37 | - | **26** | 38 | 68% |
| Tackles | 15 | - | **8** | 4 (sliding) | 200% |
| xG | 0.20 | - | **0.11** | 2.01 | 5% |

### Bayern (Black)

| Metric | Baseline | R17 | **R18** | Wyscout | R18 Rate |
|--------|----------|-----|---------|---------|----------|
| Players | 16 | 12 | **11** | 14 (11+3 subs) | 79% |
| Passes | 181 | - | **243** | 676 | 36% |
| Pass accuracy | 63.0% | - | **43.2%** | 92% | - |
| Shots on target | 18 | - | **35** | 6 | 583% |
| Goals | 4 | - | **1** | 2 | 50% |
| Dribbles | 10 | - | **22** | 38 | 58% |
| Interceptions | 57 | - | **72** | 39 | 185% |
| Tackles | 34 | - | **20** | 1 (sliding) | 2000% |
| xG | 0.32 | - | **0.35** | 1.74 | 20% |

---

## Round-over-Round Comparison

| Metric | Baseline | R16 | R17 | **R18** | Wyscout | Trend |
|--------|----------|-----|-----|---------|---------|-------|
| Team colors | W/Yellow | W/Yellow | W/Green | **W/Black** | W/Black | FIXED |
| Players | 35 | 35 | 26 | **26** | 28 | stable |
| Goals | 4 (0-4) | 0 | 3 | **1** (0-1) | 4 (2-2) | regressed |
| Shots | 23 | - | 42 | **45** | 10 | over-counted |
| Tackles | 49 | - | 28 | **28** | ~25-35 | in range |
| Passes | 247 | - | ~300 | **343** | 945 | improving |
| xG | 0.52 | - | 0.42 | **0.46** | 3.75 | stable-low |
| Interceptions | 94 | - | - | **98** | 77 | close |
| Dribbles | 14 | - | - | **36** | 53 | improved |
| Pass Accuracy | 13.6% | - | - | **54.5%** | ~80% | improved |

---

## What Improved in R18

1. **Team colors FIXED** — White + Black correctly identified (was Green in R17, Yellow in Baseline). Bayern's dark kit now properly classified.
2. **Tackles in range** — 28 tackles, within the 25-35 target (unchanged from R17, confirms R17 fix held).
3. **Pass accuracy improved** — 54.5% overall (White team 79.6% matches Wyscout's 79% exactly). Up from 13.6% at Baseline.
4. **Pass count up** — 343 vs 247 Baseline (+39%). Still 36% of Wyscout but trending right.
5. **Dribbles improved** — 36 vs 14 Baseline (+157%). Now at 68% of Wyscout's 53.
6. **Interceptions solid** — 98 vs Wyscout 77 (127%). Consistent across rounds.
7. **Ghost filtering improved** — 26 players from 81 raw (was 35 at Baseline). Down from 35→26.

## What Regressed

1. **Goals dropped** — 1 goal (was 3 in R17, 4 at Baseline). Goal detection is unreliable. 8 goal-correction events in log but only 1 confirmed.
2. **Shots still 4.5x over** — 45 shots (was 42 in R17, 23 at Baseline). Shot detection sensitivity too high; many ball movements near goal counted as shots.

## Remaining Issues (Priority Order)

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| 1 | **Shots over-counted** (45 vs 10) | Critical | 4.5x over-count. Ball speed + proximity triggers too easily. Shot debounce and speed threshold need tuning. |
| 2 | **xG very low** (0.46 vs 3.75) | High | Even with 45 shots, avg xG per shot is ~0.01. xG distance normalization and coefficients undervalue shots. |
| 3 | **Goals under-detected** (1 vs 4) | High | Goal-line crossing logic is fragile. 8 candidates detected but most rejected. Need to relax goal confirmation or fix team-direction validation. |
| 4 | **Team imbalance** (15W vs 11B) | Medium | White has 4-5 ghost players (#7=50obs, #34=76obs, #35=82obs). Need stricter ghost filter. |
| 5 | **Passes still low** (343 vs 945) | Medium | 36% of Wyscout. Ownership mapping in crowded areas drops many transitions. |
| 6 | **Missing Black GK** | Low | Only White GK (#1) detected. Bayern GK not identified with GK role. |
| 7 | **Bayern pass accuracy low** (43.2% vs 92%) | Low | Many Bayern passes counted as inaccurate due to ownership mapping noise. |
| 8 | **Fouls** (0 vs 28) | Low | Not implemented. |

---

## Top Performers

### White (HSV)
| Player | Highlight |
|--------|-----------|
| #29 | Most passes (26), most shots (4), most dribbles (4), highest distance (248.5m) |
| #8 | Highest pass accuracy (92.3%, 12/13) |
| #5 | Most tackles (4) |

### Black (Bayern)
| Player | Highlight |
|--------|-----------|
| #14 | Most passes (55), most time on ball (24.0s), highest distance (328.8m) |
| #9 | Only goal scorer, most shots (10), most dribbles (6) |
| #17 | Most tackles (6), 5 dribbles |
| #18 | Most interceptions (12, tied with #36), most time on ball after #14 (20.5s) |

---

## Next Steps for R19

1. **Fix shot over-detection**: Raise `SHOT_SPEED_THRESHOLD` and increase shot debounce. Consider requiring ball to be in final third.
2. **Fix goal detection**: Review goal-line crossing logic; relax confirmation criteria or fix team-direction inference.
3. **xG calibration**: Increase xG coefficients or adjust distance normalization so close-range shots produce xG > 0.3.
4. **Ghost player filter**: Players with <100 observations and no significant stats → remove.
5. **Pass detection**: Investigate ownership handoff logic for crowded areas.
