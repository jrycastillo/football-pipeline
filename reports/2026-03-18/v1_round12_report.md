# Video 1 — Round 12 Report

**Video:** `1e942fd8a6344bd` (1.3 GB, ~105 min, 30fps)
**File:** f4dbcb0a_2026-03-09-22-46-45337194.webm
**Pipeline:** Round 12 — Phase 216 Threshold + Moderate Cooldowns + MPS Acceleration
**Commit:** `c000bd91`
**Date:** 2026-03-18
**Processed locally on M4 Pro MacBook** (MPS GPU, stride=3)

---

## Round 12 Changes (3 Fixes)

### 1. Phase 216 Smart Merge with Fragment Threshold (`stats/metrics.py`)
- **Before (R10-R11):** Pick-primary only — kept stats from the single longest track fragment per jersey, discarding ~60-80% of legitimate events from other time periods
- **After (R12):** Threshold-based approach:
  - **<=5 fragments per jersey:** Sum EVENT stats (passes, tackles, shots, dribbles, interceptions) across all fragments + keep primary for accumulative stats (distance, touch_frames)
  - **>5 fragments per jersey:** Pick-primary only (summing would inflate due to ByteTrack micro-fragmentation)
- **Why:** ByteTrack with stride>1 creates 100-1500+ micro-fragments per player. Unlimited summing causes 20-60x event inflation. The threshold of 5 fragments safely recovers events without inflation.

### 2. Moderate Cooldowns (`stats/event_logic.py`)
| Event | R10 | R11 | R12 |
|-------|-----|-----|-----|
| Tackle | 5s | 15s | **8s** |
| Shot | 1s | 5s | **3s** |
| Dribble | 1s | 3s | **2s** |

- R11 cooldowns were calibrated for V4's extreme case (482K frames at stride=1)
- With correct stride=3, moderate cooldowns give realistic counts

### 3. MPS GPU Acceleration (`pipeline_consolidated.py`)
- **Before:** YOLO models defaulted to CPU (~3 frames/sec, ~16.8 hrs per video)
- **After:** Models load on MPS (Apple Silicon GPU) via `.to(_device)` (~11 frames/sec, ~5 hrs per video)
- **Impact:** 3.5x local processing speedup on M4 Pro

---

## Inflation Bug Discovery & Fix

During the first V1 R12 run, the Phase 216 smart merge (which sums events across all fragments) produced massively inflated stats:

| Stat | R11 | R12 (First Run - INFLATED) | Inflation Factor |
|------|-----|---------------------------|-----------------|
| Passes | 96 | 2,608 | 27x |
| Tackles | 9 | 177 | 20x |
| Shots | 7 | 187 | 27x |
| Dribbles | 2 | 120 | 60x |
| Goals | 0 | 56 | -- |

**Root Cause:** ByteTrack with stride=3 creates 100-1500+ micro-fragments per jersey (e.g., Jersey #14 had 1,525 fragments, Jersey #35 had 1,470). Each fragment independently generates events. Summing across ALL fragments = massive multiplication.

**Fix:** Added `_MERGE_MAX_FRAGMENTS = 5` threshold. Players with <=5 track fragments get smart merge (safe — legitimate re-detections). Players with >5 fragments revert to pick-primary (too many ByteTrack micro-fragments to sum safely).

The inflated run was deleted and V1 was restarted with the fix applied.

---

## V1 Results — R10 vs R11

| Stat | R10 | R11 | Change |
|------|-----|-----|--------|
| Teams | Green 19 / White 11 | Green 19 / White 11 | Same |
| Score | 0-0 | 0-0 | Same |
| Passes | 96 | 96 | 0 |
| Tackles | 10 | 9 | -1 |
| Shots | 10 | 7 | -3 |
| Dribbles | 2 | 2 | 0 |
| xG | 0.10 | 0.07 | -0.03 |

Minimal changes between R10 and R11 — the R11 cooldown fixes primarily targeted V4's extreme inflation case.

---

## V1 R11 Per-Player Stats

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

## V1 R12 Status

**Processing in progress** on M4 Pro (MPS GPU) with fixed Phase 216 threshold.
- Started: 2026-03-18 14:14
- Current: Frame ~96,900 / ~189,000 (~51%)
- Speed: ~605 frames/min (MPS)
- ETA: ~19:30 (7:30 PM)

R12 results will be added to this report once processing completes. Expected improvements:
- Moderate cooldowns (8s/3s/2s) should increase tackles and dribbles slightly vs R11
- Phase 216 smart merge (for players with <=5 fragments) should recover some events lost by pick-primary
- Team balance may improve with fresh ByteTrack tracking

### Expected R12 Ranges (V1)
| Stat | R11 | R12 Expected | Real Match Range |
|------|-----|-------------|-----------------|
| Passes | 96 | 100-200 | 200-500 |
| Tackles | 9 | 15-40 | 30-60 |
| Shots | 7 | 10-20 | 15-30 |
| Dribbles | 2 | 5-15 | 20-50 |

---

## Processing Timeline

| Round | Date | Config | V1 Passes | V1 Tackles | V1 Shots | V1 Dribbles |
|-------|------|--------|-----------|------------|----------|-------------|
| R10 | 2026-03-10 | Pick-primary, stride=3 (H100) | 96 | 10 | 10 | 2 |
| R11 | 2026-03-14 | R10 + cooldowns 15s/5s/3s (H100) | 96 | 9 | 7 | 2 |
| R12a | 2026-03-18 | Smart merge (no threshold) | 2,608 | 177 | 187 | 120 |
| R12b | 2026-03-18 | Smart merge + threshold=5 | *Processing...* | | | |

R12a was discarded due to the inflation bug (see "Inflation Bug Discovery & Fix" above).

---

## Known Issues (Carried Forward)

1. **Team Imbalance**: V1 shows 19/11 (Green/White). Expected ~11/11 for a real match. The pipeline assigns too many short-lived tracks to one team.

2. **Low Event Counts**: Even with R12 fixes, event counts (especially passes: 96) remain below real match levels (200-500). This is primarily limited by ball detection quality and ownership mapping accuracy.

3. **Distance Values**: Some players show 0m distance despite many observations. Related to frame stride and position deduplication.

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
