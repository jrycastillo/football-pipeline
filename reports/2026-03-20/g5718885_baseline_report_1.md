# Baseline Report: g5718885-fullhd.mp4

**Date:** 2026-03-20
**Video:** `test_videos/g5718885-fullhd.mp4` (Hamburger SV 2-2 Bayern Munich, Bundesliga R20, 31/01/2026)
**Match:** Hamburger SV (White) vs Bayern Munich (Red kit, detected as Yellow)
**Final Score (Real):** HSV 2 - 2 Bayern

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
| Processing time | ~7 hours 28 minutes |
| Frames processed | ~53,208 (stride=3) |

---

## Pipeline Results Summary

| Metric | Pipeline | Wyscout (Real) | Detection Rate |
|--------|----------|----------------|----------------|
| **Players detected** | 35 | 28 (11+11 + 6 subs) | 125% (over-count) |
| **Goals** | 4 | 4 | **100%** |
| **Shots on target** | 23 | 10 | 230% (over-count) |
| **Total passes** | 247 | 945 | 26% |
| **Tackles** | 49 | 5 (sliding only) | 980% (over-count) |
| **Dribbles** | 14 | 53 | 26% |
| **Interceptions** | 94 | 77 | 122% |
| **xG** | 0.52 | 3.75 | 14% |
| **Fouls** | 0 | 28 | 0% |
| **Crosses** | 20 | N/A | - |

### Score Detection

| | Pipeline | Real |
|---|---------|------|
| White (HSV) | 0 | 2 |
| Yellow (Bayern) | 4 | 2 |
| **Total** | **4** | **4** |

Pipeline detected the correct total goals (4) but attributed all goals to Yellow (Bayern). In reality HSV scored 2 and Bayern scored 2.

---

## Team Breakdown

### White Team (HSV) — 19 players detected

| # | Pos | Obs | Pass | Acc | Acc% | Tck | SoT | Gls | Drb | Int | Dist | ToB(s) | xG |
|---|-----|-----|------|-----|------|-----|-----|-----|-----|-----|------|--------|-----|
| 1 | GK | 420 | 6 | 1 | 16.7 | 3 | 0 | 0 | 0 | 5 | 91.2 | 1.68 | 0.00 |
| 3 | P | 580 | 10 | 3 | 30.0 | 1 | 0 | 0 | 1 | 4 | 147.2 | 3.24 | 0.00 |
| 5 | P | 290 | 8 | 2 | 25.0 | 2 | 0 | 0 | 2 | 4 | 67.1 | 0.00 | 0.00 |
| 8 | P | 264 | 7 | 0 | 0.0 | 2 | 0 | 0 | 0 | 5 | 50.9 | 6.00 | 0.00 |
| 11 | P | 178 | 6 | 1 | 16.7 | 0 | 1 | 0 | 0 | 3 | 47.3 | 2.88 | 0.01 |
| 23 | P | 602 | 5 | 0 | 0.0 | 2 | 1 | 0 | 0 | 2 | 134.6 | 3.12 | 0.01 |
| 30 | P | 408 | 7 | 1 | 14.3 | 2 | 3 | 0 | 0 | 2 | 52.7 | 4.68 | 0.18 |
| 25 | P | 228 | 5 | 0 | 0.0 | 0 | 0 | 0 | 1 | 5 | 37.2 | 2.40 | 0.00 |
| 2 | P | 132 | 3 | 0 | 0.0 | 0 | 0 | 0 | 0 | 1 | 33.2 | 2.52 | 0.00 |
| 22 | P | 114 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 25.7 | 0.00 | 0.00 |
| 15 | P | 106 | 2 | 0 | 0.0 | 1 | 0 | 0 | 0 | 2 | 20.0 | 1.68 | 0.00 |
| 19 | P | 98 | 1 | 1 | 100.0 | 0 | 0 | 0 | 0 | 1 | 8.6 | 1.68 | 0.00 |
| 35 | P | 82 | 5 | 0 | 0.0 | 1 | 0 | 0 | 0 | 1 | 22.7 | 0.96 | 0.00 |
| 34 | P | 76 | 1 | 0 | 0.0 | 1 | 0 | 0 | 0 | 2 | 14.7 | 1.68 | 0.00 |
| 7 | P | 50 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 15.7 | 0.24 | 0.00 |
| 28 | P | 34 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 4.1 | 0.00 | 0.00 |
| 21 | P | 30 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 4.5 | 0.00 | 0.00 |
| 40 | P | 28 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 2.3 | 0.00 | 0.00 |
| 6 | P | 22 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0.2 | 0.00 | 0.00 |
| **Total** | | **3,742** | **66** | **9** | **13.6** | **15** | **5** | **0** | **4** | **37** | **779.9** | **32.76** | **0.20** |

### Yellow Team (Bayern) — 16 players detected

| # | Pos | Obs | Pass | Acc | Acc% | Tck | SoT | Gls | Drb | Int | Dist | ToB(s) | xG |
|---|-----|-----|------|-----|------|-----|-----|-----|-----|-----|------|--------|-----|
| 14 | P | 1122 | 42 | 36 | 85.7 | 7 | 3 | 2 | 2 | 5 | 297.9 | 15.12 | 0.06 |
| 9 | P | 1142 | 29 | 22 | 75.9 | 7 | 5 | 1 | 0 | 4 | 194.4 | 18.36 | 0.14 |
| 18 | P | 832 | 26 | 23 | 88.5 | 3 | 3 | 0 | 0 | 5 | 197.6 | 15.12 | 0.04 |
| 17 | P | 874 | 18 | 9 | 50.0 | 7 | 0 | 0 | 4 | 17 | 185.3 | 7.92 | 0.00 |
| 29 | P | 854 | 17 | 3 | 17.6 | 2 | 2 | 0 | 1 | 12 | 158.6 | 6.72 | 0.02 |
| 36 | P | 720 | 15 | 11 | 73.3 | 3 | 2 | 1 | 2 | 3 | 203.7 | 3.36 | 0.03 |
| 24 | P | 642 | 15 | 3 | 20.0 | 3 | 2 | 0 | 1 | 3 | 167.6 | 5.40 | 0.02 |
| 13 | P | 876 | 7 | 0 | 0.0 | 1 | 0 | 0 | 0 | 3 | 152.4 | 2.64 | 0.00 |
| 10 | P | 188 | 5 | 4 | 80.0 | 1 | 0 | 0 | 0 | 1 | 39.0 | 1.44 | 0.00 |
| 33 | P | 210 | 4 | 2 | 50.0 | 0 | 0 | 0 | 0 | 2 | 38.9 | 3.00 | 0.00 |
| 62 | P | 246 | 2 | 1 | 50.0 | 0 | 0 | 0 | 0 | 1 | 46.5 | 0.96 | 0.00 |
| 27 | P | 90 | 1 | 0 | 0.0 | 0 | 1 | 0 | 0 | 1 | 3.2 | 1.68 | 0.01 |
| 4 | P | 86 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 16.8 | 0.00 | 0.00 |
| 38 | P | 48 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 6.9 | 0.00 | 0.00 |
| 31 | P | 24 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.00 | 0.00 |
| 16 | P | 20 | 0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 5.1 | 0.00 | 0.00 |
| **Total** | | **7,974** | **181** | **114** | **63.0** | **34** | **18** | **4** | **10** | **57** | **1,713.9** | **81.72** | **0.32** |

---

## Per-Team Comparison: Pipeline vs Wyscout

### HSV (White)

| Metric | Pipeline | Wyscout | Rate |
|--------|----------|---------|------|
| Players | 19 | 16 (11+5 subs) | 119% |
| Passes | 66 | 269 | 25% |
| Pass accuracy | 13.6% | 79% | - |
| Shots on target | 5 | 4 | 125% |
| Goals | 0 | 2 | **0%** |
| Dribbles | 4 | 15 | 27% |
| Interceptions | 37 | 38 | **97%** |
| Tackles | 15 | 4 (sliding) | 375% |
| Fouls | 0 | 12 | 0% |
| xG | 0.20 | 2.01 | 10% |

### Bayern (Yellow)

| Metric | Pipeline | Wyscout | Rate |
|--------|----------|---------|------|
| Players | 16 | 14 (11+3 subs) | 114% |
| Passes | 181 | 676 | 27% |
| Pass accuracy | 63.0% | 92% | - |
| Shots on target | 18 | 6 | 300% |
| Goals | 4 | 2 | 200% |
| Dribbles | 10 | 38 | 26% |
| Interceptions | 57 | 39 | 146% |
| Tackles | 34 | 1 (sliding) | 3400% |
| Fouls | 0 | 16 | 0% |
| xG | 0.32 | 1.74 | 18% |

---

## Improvement Priorities (New Video)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | **Pass detection** (26% of real) | Critical — core stat | High |
| 2 | **Tackle definition** (proximity vs sliding) | High — 10x over-count | Medium |
| 3 | **Goal attribution** (all to one team) | High — wrong score | Medium |
| 4 | **xG model** (14% of real) | Medium — derivative stat | Medium |
| 5 | **Player ghost filtering** (35 vs ~28) | Medium — noisy output | Low |
| 6 | **Red→Yellow color misclass** | Low — cosmetic | Low |
| 7 | **Foul detection** (0%) | Low — not implemented | High |
| 8 | **GK detection for both teams** | Low — missing 1 GK | Low |
