# H100 Results Comparison: Feb 12 vs Feb 14

**Date:** 2026-02-17
**Fixes applied in Feb 14 run:** Stats formula fixes (P0-P2) + ball interpolation (max_gap 25→50) + SAM2 removal
**Config:** VID_STRIDE=5, FPS=25, no video output

---

## 1. Aggregate Totals (All 3 Videos)

| Metric          | Feb 12 | Feb 14 | Delta   | Change   |
|-----------------|--------|--------|---------|----------|
| Players         | 71     | 54     | -17     | -24%     |
| Passes          | 123    | 311    | +188    | **+153%**|
| Shots           | 25     | 40     | +15     | +60%     |
| xG              | 5.02   | 3.85   | -1.17   | -23%     |
| Tackles         | 20     | 13     | -7      | -35%     |
| Dribbles        | 17     | 10     | -7      | -41%     |
| Goals           | 6      | 6      | 0       | 0%       |
| Saves           | 0      | 0      | 0       | —        |
| Total Obs       | 670,756| 68,602 | -602,154| **-90%** |

---

## 2. Per-Video Breakdown

### Video 1 — `162b6abe208946b` (314 MB, short clip)

| Metric   | Feb 12 | Feb 14 | Delta |
|----------|--------|--------|-------|
| Players  | 23     | 17     | -6    |
| Passes   | 8      | 13     | +5    |
| Shots    | 3      | 0      | -3    |
| xG       | 0.17   | 0.04   | -0.13 |
| Tackles  | 2      | 0      | -2    |
| Dribbles | 0      | 1      | +1    |
| Score    | 0-0    | 0-0    | same  |

**Assessment:** Low-data video. Passes improved slightly (+62%). Shots dropped to 0 — likely the higher
threshold (12 m/s) correctly filtered false positives from this short clip. xG dropped 76% from
the logistic model fix. Observation counts collapsed (avg 1,834→186 per player).

### Video 2 — `14c0f4e8c4af40d` (1.3 GB, full match)

| Metric   | Feb 12 | Feb 14 | Delta |
|----------|--------|--------|-------|
| Players  | 25     | 18     | -7    |
| Passes   | 55     | 74     | +19   |
| Shots    | 8      | 14     | +6    |
| xG       | 1.30   | 1.13   | -0.17 |
| Tackles  | 9      | 1      | **-8**|
| Dribbles | 9      | 2      | -7    |
| Goals    | 3      | 5      | +2    |
| Score    | 3-0    | 5-0    | +2 goals |

**Assessment:** Passes +35% (pass gap-fill fix working). Tackles dropped 89% (dedup fix aggressive
but directionally correct). Dribbles -78% (movement check filtering standing-still events). xG only
-13% — the logistic model effect was mild here since shots were already at reasonable distances.
Goals increased from 3→5 (better ball tracking may be finding more goal-line crossings). Blue #9
(28k obs, 1 goal, 5 passes in Feb 12) completely disappeared from Feb 14.

### Video 3 — `69a33466fc234db` (3.4 GB, full match)

| Metric   | Feb 12 | Feb 14 | Delta |
|----------|--------|--------|-------|
| Players  | 23     | 19     | -4    |
| Passes   | 60     | 224    | **+164** |
| Shots    | 14     | 26     | +12   |
| xG       | 3.55   | 2.68   | -0.87 |
| Tackles  | 9      | 12     | +3    |
| Dribbles | 8      | 7      | -1    |
| Goals    | 3      | 1      | -2    |
| Score    | 1-2    | 0-1    | -2 goals |

**Assessment:** Most dramatic changes. Passes exploded +273% (the none-gap fix + ball interpolation
gave much more continuous ownership data). But 224 passes for 19 players over a match is still
below real-world (~800-1000). Shots nearly doubled despite higher threshold — the expanded ball
interpolation (max_gap 50) creates more continuous tracking, enabling more velocity calculations.
xG dropped 25% from logistic model. Goals dropped from 3→1, losing 2 detected goals — possibly
from reduced observation quality.

---

## 3. What Worked (Fixes Confirmed)

### Pass Detection — IMPROVED
- Total passes: 123 → 311 (+153%)
- Video 3 showed the biggest gain: 60 → 224
- The none-gap fix and ball interpolation improvements are clearly working
- Pass-per-minute rates are now more realistic, though still below professional match averages

### xG Model — IMPROVED
- Total xG: 5.02 → 3.85 (-23%)
- The biggest single correction: Green #35 in Video 3 dropped from 2.36 → 0.13 xG
- The logistic model eliminates the 0.99 xG cap issue for close-range shots
- Per-shot xG is now in realistic ranges (0.01-0.56 vs 0.01-2.36 before)

### Tackle Deduplication — IMPROVED
- Total tackles: 20 → 13 (-35%)
- Video 2 shows the clearest effect: 9 → 1 (-89%)
- No more section 1 + section 3 double-counting

### Dribble Movement Check — IMPROVED
- Total dribbles: 17 → 10 (-41%)
- Standing-still-with-opponent events no longer counted as dribbles

---

## 4. New Problems Found

### CRITICAL: Observation Count Collapse (-90%)

| Video | Feb 12 Obs  | Feb 14 Obs | Drop  |
|-------|-------------|------------|-------|
| V1    | 42,178      | 3,156      | -93%  |
| V2    | 261,996     | 23,972     | -91%  |
| V3    | 366,582     | 41,474     | -89%  |

Average observations per player dropped from ~9,400 to ~1,270. This is NOT caused by the
stats formula fixes — it's a detection/tracking regression, likely from:
- **SAM2 removal** — SAM2 was providing segmentation masks that improved tracking persistence.
  Without it, ByteTrack loses players more frequently, resulting in shorter tracks.
- This cascades into every stat: fewer frames observed → less data for ownership, distance,
  events, etc.

**Impact:** All per-player stats are computed from fewer data points. Distance measurements
are lower, touch counts are lower, and event detection has less data to work with. This is
the single biggest issue to investigate.

### HIGH: Shot Count Increased (+60%) Despite Higher Threshold

Expected: Fewer shots (threshold 8→12 m/s should filter fast passes).
Actual: 25 → 40 shots.

**Root cause:** Ball interpolation (max_gap 25→50) creates ~2x more continuous ball track
segments. More continuous data → more frame pairs where velocity can be calculated → more
shots detected. The expanded interpolation generates enough additional data to more than
offset the higher speed threshold.

**Concern:** Linear interpolation across 50-frame gaps (~6 seconds at VID_STRIDE=5) may
create artificial velocity spikes at gap boundaries. When the ball "jumps" from one detected
position to another across a long gap, the interpolated positions at the edges can show
high apparent velocity that doesn't represent a real shot.

**Recommendation:** Either reduce max_gap back to 25-30, or add a check to skip velocity
calculations on interpolated (non-detected) ball positions.

### HIGH: GK Shots/xG Not Gated

| Player | Video | Shots | xG   |
|--------|-------|-------|------|
| Red #1 (GK) | V3 | 3 | 1.18 |
| Grn #18 (GK) | V3 | 3 | 0.28 |
| Grn #1 (GK) | V3-Feb14 | 3 | 1.18 |

Goalkeepers are accumulating shot and xG stats. Goal kicks, punts, and clearances at high
velocity toward the opposite goal are being detected as shots. The GK filter in the shot
detection section (line 444) only blocks the shooter assignment, but Section 4 (touch-in-box
xG, line 629) has no GK gate at all.

**Status:** This was identified as P1 in the original audit but was NOT included in the
Feb 14 formula fixes. Needs separate fix.

### HIGH: Interception Inflation

| Player         | Feb 12 Int | Feb 14 Int |
|----------------|------------|------------|
| Grn #35 (V3)   | 4          | 20         |
| Red #18 (V3)   | —          | 24         |
| Red #1 GK (V3) | 3          | 18         |
| Blue #14 (V2)  | 1          | 17         |

Real-world: Top interceptors get 1-3 per match. Current logic counts every incomplete
pass received by an opponent as an interception. With pass accuracy at ~53%, roughly half
of all passes generate an interception. Interceptions = incomplete passes received, which
is correct by definition, but the raw numbers are inflated because pass accuracy is too low
(too many false "incomplete" passes from team clustering errors).

**Root cause chain:** Bad team clustering → same-team passes marked as cross-team →
incomplete pass → inflated interception count.

### MEDIUM: Pass Accuracy Still Low (~53%)

Average pass accuracy across players with >2 passes is ~53%. Real football is 75-85%.
Possible causes:
1. **Team clustering errors** — If two same-team players are assigned different teams,
   every pass between them is "incomplete"
2. **Tackle-as-pass still leaking** — The 2.0m proximity filter may not catch all
   physical dispossessions
3. **Observation collapse** — Fewer tracking frames means fewer ownership assignments,
   creating more ownership gaps that break pass chains

### MEDIUM: Goal Detection Inconsistency

- Video 2: 3 goals (Feb 12) → 5 goals (Feb 14) — gained 2
- Video 3: 3 goals (Feb 12) → 1 goal (Feb 14) — lost 2

Total goals stayed at 6, but the distribution shifted. Goal detection depends on ball
tracking quality at the goal line, which is sensitive to ball detection + interpolation.

---

## 5. Player-Level Deep Dive (Video 3 — Most Data)

### Feb 12 → Feb 14 Comparison (matched jerseys)

| Player       | Obs 12→14    | Pass 12→14 | xG 12→14    | Shots 12→14 | Int 12→14 |
|-------------|-------------|------------|-------------|-------------|-----------|
| Grn #4      | 45,010→6,404 | 6→48      | 0.08→0.15   | 0→9         | 2→16      |
| Grn #35     | 41,958→5,532 | 14→41     | 2.36→0.13   | 2→4         | 4→20      |
| Grn #18 GK  | 81,210→10,248| 15→35     | 0.04→0.28   | 1→3         | 3→24      |
| Red #1 GK   | 43,544→4,104 | 8→35      | 0.54→1.18   | 3→3         | 3→18      |
| Red #14     | 39,956→5,254 | 6→31      | 0.33→0.49   | 3→5         | 3→10      |
| Red #30     | 24,488→1,746 | 5→4       | 0.24→0.00   | 2→0         | 3→1       |

**Key observations:**
- Despite 87% fewer observations, passes increased 3-8x per player — the pass detection
  fix is very effective
- xG correction worked perfectly on Grn #35 (2.36→0.13) but Red #1 GK rose (0.54→1.18) —
  the GK gate is missing
- Interceptions scaled proportionally with passes (expected since Int = incomplete passes
  received by opponent)
- Shot counts increased for most players despite the higher threshold

---

## 6. Priority Actions for Next Fix

| Priority | Issue | Expected Impact |
|----------|-------|-----------------|
| **P0** | Investigate observation collapse — is SAM2 removal the cause? Consider re-enabling or finding alternative tracking persistence | Fixes all downstream stat reliability |
| **P0** | Reduce ball interpolation max_gap (50→30) or skip velocity calc on interpolated frames | Fixes false shot inflation |
| **P1** | Gate GK stats — block shots/xG/goals for dominant_class=1 | Fixes GK anomalies |
| **P1** | Investigate team clustering accuracy for Feb 14 runs | Fixes pass accuracy + interception inflation |
| **P2** | Add interception cap or dedup (max 1 per 3-second window) | Reduces inflated interceptions |

---

## 7. Summary

The Feb 14 stats formula fixes achieved their intended goals:
- **Passes +153%** — significantly more realistic event detection
- **xG -23%** — logistic model producing calibrated values
- **Tackles -35%** — deduplication working
- **Dribbles -41%** — movement check filtering false positives

However, a separate regression in **detection/tracking** (likely SAM2 removal) caused a
**90% drop in observation counts**, which undermines the reliability of all stats. The ball
interpolation expansion (max_gap 50) also created a side effect of **+60% more shots**
from velocity calculations on interpolated positions.

**Net assessment:** Stats formulas are better, but the tracking quality regression needs to
be resolved before the numbers are production-ready.
