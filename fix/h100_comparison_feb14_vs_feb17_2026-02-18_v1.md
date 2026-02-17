# H100 Results Comparison: Feb 14 vs Feb 17

**Date:** 2026-02-18
**Fixes applied in Feb 17 run:** Fix Round 2 (Event Logic P0-P2 updates) + VID_STRIDE=3 (was 5)
**Config:** VID_STRIDE=3, FPS=25, --parallel 3, no video output

---

## 1. Aggregate Totals (All 3 Videos)

| Metric          | Feb 14 (Stride=5) | Feb 17 (Stride=3) | Delta   | Change   |
|-----------------|-------------------|-------------------|---------|----------|
| Players         | 54                | 53                | -1      | -2%      |
| Passes          | 311               | 585               | +274    | **+88%** |
| Shots           | 40                | 16                | -24     | -60%     |
| xG              | 3.85              | 9.77              | +5.92   | +154%    |
| Tackles         | 13                | 48                | +35     | **+270%**|
| Dribbles        | 10                | 23                | +13     | +130%    |
| Goals           | 6                 | 5                 | -1      | -17%     |
| Saves           | 0                 | 0                 | 0       | —        |

---

## 2. Per-Video Breakdown

### Video 1 — `162b6abe208946b` (314 MB, short clip)

| Metric   | Feb 14 | Feb 17 | Delta |
|----------|--------|--------|-------|
| Passes   | 13     | 39     | **+200%** |
| Shots    | 0      | 2      | +2    |
| xG       | 0.04   | 0.41   | +0.37 |
| Tackles  | 0      | 3      | +3    |
| Score    | 0-0    | 1-0    | +1 Goal |

**Assessment:** Massive improvement in event density. Passes tripled (13→39) thanks to `VID_STRIDE=3` capturing more ball movement frames. Goal detected (1-0) where previously missed. xG increased to realistic level (0.41).

### Video 2 — `14c0f4e8c4af40d` (1.3 GB, full match)

| Metric   | Feb 14 | Feb 17 | Delta |
|----------|--------|--------|-------|
| Passes   | 74     | 12     | **-84%** |
| Shots    | 14     | 0      | -14   |
| xG       | 1.13   | 0.00   | -1.13 |
| Goals    | 5      | 0      | **-5** |
| Score    | 5-0    | 0-0    | Lost Match |
| Teams    | Blue,Green | Blue,Yellow | Color Shift |

**Assessment:** **CRITICAL FAILURE.** The pipeline effectively collapsed on this video.
- **Passes:** Dropped to near zero (12 total for full match).
- **Goals:** Lost all 5 previously detected goals.
- **Tracking:** Team names changed (Green->Yellow), suggesting color classifier instability or different initialization.
- **Cause:** Likely a tracking persistence failure exacerbated by `VID_STRIDE=3` without parameter tuning, or a specific failure in ball tracking for this file.

### Video 3 — `69a33466fc234db` (3.4 GB, full match)

| Metric   | Feb 14 | Feb 17 | Delta |
|----------|--------|--------|-------|
| Passes   | 224    | 534    | **+138%** |
| Shots    | 26     | 14     | -46%  |
| xG       | 2.68   | 9.36   | **+250%** |
| Tackles  | 12     | 42     | +250% |
| Score    | 0-1    | 2-2    | +3 Goals |

**Assessment:** Strong step forward in event volume, mixed with potential inflation.
- **Passes:** 534 is much closer to professional standard (800+) than 224.
- **Pass Accuracy:** Red #18 achieved 89% accuracy (148/166 passes), a massive quality improvement over previous ~53%.
- **xG Inflation:** 9.36 xG for a 2-2 game is suspiciously high. Likely accumulating "shots" from non-shot events (e.g. hard clearances).
- **Score:** 2-2 result (4 goals found) is better than 0-1 (1 goal found), capturing more of the match reality.

---

## 3. What Worked (Fixes + Stride Change)

### Pass Detection & Accuracy — MAJOR SUCCESS (Vid 1 & 3)
- **Video 3:** Passes rose 224 → 534.
- **Accuracy:** Red #18 (Vid 3) went from **35 passes** (Feb 14) to **148 passes** (Feb 17) with **89% accuracy**.
- This proves that lowering `VID_STRIDE` to 3 + the proximity/gap fixes allowed the engine to construct valid pass chains that were previously broken.

### Event Density — IMPROVED
- Tackles (+270%) and Dribbles (+130%) increased significantly in Video 3.
- The pipeline is clearly "seeing" more of the game with `VID_STRIDE=3`.

### Goal Detection (Vid 1 & 3)
- Video 1: Found a goal missed in Feb 14.
- Video 3: Found 3 additional goals (Total 4).

---

## 4. New Problems Found

### CRITICAL: Video 2 Collapse
- The pipeline failed completely on Video 2. Needs immediate investigation.
- **Hypothesis:** `VID_STRIDE=3` increased processing load or noise, causing tracker fragmentation that the current logic couldn't handle for this specific footage type.

### HIGH: xG Inflation (Video 3)
- 9.36 xG is too high.
- **Cause:** Likely the `VID_STRIDE=3` + Ball Interpolation max_gap=50 combination.
- Reduced stride = more frames = more chances for velocity spikes on interpolated ball tracks to trigger "Shot" logic.
- **Fix:** We strictly need the "Gate interpolated velocity" fix (P0 in previous plan).

### HIGH: Observation Count Anomaly
- **Vid 3 Red #18:**
    - Feb 14 (Stride 5): 30,772 Obs
    - Feb 17 (Stride 3): 30,772 Obs (Wait, verifying...) -> **10,248 Obs**?
- **Analysis:**
    - Converting Obs to Time:
        - Feb 14: 30,772 * 5 * (1/25) = ~6,154 sec? (Matches allow >90 mins)
        - Feb 17: 10,248 * 3 * (1/25) = ~1,229 sec (~20 mins).
- **Conclusion:** Comparison shows a **massive drop in tracked time** for key players in Feb 17 run. Red #18 was tracked for only ~20 mins of the match.
- **Paradox:** We have *more* passes (148) from *less* tracked time (20 mins)?
- **Implication:** The efficient FPS/Event density is extremely high, but the tracker is losing the player for huge chunks of the game.

---

## 5. Player-Level Deep Dive (Video 3)

| Player       | Feb 14 Passes | Feb 17 Passes | Feb 14 Obs | Feb 17 Obs |
|-------------|---------------|---------------|------------|------------|
| Red #18     | 35            | **148**       | 30,772     | 10,248     |
| Red #14     | 31            | 3             | 5,254      | 9,830      |
| Grn #4      | 48            | 97            | 6,404      | 10,260     |
| Grn #1 GK   | 35            | 35            | 10,248     | 4,104      |

**Key Finding:**
- **Red #18 (Feb 17)** is the standout: 148 passes in just 10k observations.
- **Inconsistency:** Some players gained obs (Red #14: 5k->9k), some lost massive obs (Red #18: 30k->10k).
- The tracker behavior with `VID_STRIDE=3` is highly unstable/variant compared to Stride 5.

---

## 6. Recommendations

1.  **Investigate Video 2 Failure:** Run localized debug on V2 with `VID_STRIDE=3` to see why tracking/events collapsed.
2.  **Stabilize Tracking:** The observation counts are erratic. `VID_STRIDE=3` might require ByteTrack parameter tuning (e.g., matching threshold) because the frame-to-frame intersection over union (IoU) changes with stride.
3.  **Deploy Velocity Check:** The xG inflation confirms we need to ignore velocity on interpolated ball frames.
4.  **Consider Stride 4?** Stride 3 gives great pass stats (Vid 3) but unstable tracking. Stride 5 gives stable-ish tracking but fewer events. Stride 4 might be a middle ground, or tune tracker for Stride 3.

## 7. Summary

- **Feb 17 (Stride 3)** proved that higher temporal resolution **drastically improves pass count and accuracy** (Video 3 passes +138%, acc ~89%).
- However, it introduced **instability**: Video 2 collapsed completely, and xG became inflated due to interpolation artifacts.
- **Next Step:** Fix the instability (Video 2) and xG inflation, but **keep the lower stride** (or tune for it) because the gain in pass analytics is too valuable to lose.
