# Pipeline Progress Report — Feb 9 → Feb 11 → Feb 12, 2026

**3 Videos Tracked:** `162b6abe208946b` (V1, 314 MB), `14c0f4e8c4af40d` (V2, 1.3 GB), `69a33466fc234db` (V3, 3.4 GB)

---

## 1. Processing Configuration

| Setting | Feb 9 | Feb 11 | Feb 12 |
|---------|-------|--------|--------|
| Pipeline | Fix V5 | Fix V5 + H100 Speedup | Fix V5 + H100 Speedup |
| VID_STRIDE | 3 | 3 | 5 |
| JNR_STRIDE | 5 | 10 | 5 |
| DET_IMG_SIZE | 640 | 640 | 640 |
| --no_video_output | No | Yes | Yes |
| EFF_FPS scaling | ❌ Not applied | ❌ Not applied | ✅ Applied (fd35c74) |

---

## 2. Processing Time

| Video | Feb 9 | Feb 11 | Feb 12 |
|-------|------:|-------:|-------:|
| V1 (314 MB) | 38 min | 37 min | 30 min |
| V2 (1.3 GB) | 8h 8m | 7h 55m | 10h 9m |
| V3 (3.4 GB) | 8h 40m | 8h 25m | 10h 46m |

> [!WARNING]
> Feb 12 was slower for V2/V3 despite higher VID_STRIDE. Likely caused by sequential memory pressure from running all 3 back-to-back in a single orchestrator session.

---

## 3. Team Color Assignment

| Video | Feb 9 | Feb 11 | Feb 12 |
|-------|-------|--------|--------|
| V1 | **Cyan, Red** (2 teams) | **Blue, Red** (2 teams) ✅ | **Blue, Red** (2 teams) ✅ |
| V2 | **Blue, Green, Yellow, Cyan, Navy** (5!) | **Blue, Green** (2 teams) ✅ | **Blue, Green, Yellow** (3 teams) ⚠️ |
| V3 | **Green, Red, Gold, Orange, Cyan** (5!) | **Green, Red** (2 teams) ✅ | **Blue, Green, Red, Yellow** (4 teams) ⚠️ |

### Status: ⚠️ Partially Solved

- **Feb 9 → Feb 11:** Major improvement. Color merge map (`lime→green`, `navy→blue`, etc.) eliminated fragmentation from 5 colors to 2 per video.
- **Feb 11 → Feb 12:** Regression. GK kit colors (Yellow) and color misclassifications (Blue variant of Green) reappeared as separate teams.
- **Fix Applied (Feb 13):** FIX #4 in `stats/metrics.py` — orphan color groups now auto-merge into nearest primary team using HSV hue distance. **Awaiting verification on next run.**

---

## 4. Goalkeeper Issue

**Problem:** GK #1 in V3 recorded 4 goals (Feb 9) and 4 goals (Feb 11) — impossible for a goalkeeper.

| Run | V3 GK #1 Goals | V3 GK #1 Shots | V3 GK #1 xG |
|-----|:--------------:|:--------------:|:------------:|
| Feb 9 | **4** ❌ | 8 | 1.71 |
| Feb 11 | **4** ❌ | 8 | 1.71 |
| Feb 12 | **1** ⬇️ | 3 | 0.54 |

### Status: ⬇️ Improved, Not Fully Solved

- Feb 12's EFF_FPS scaling reduced the inflated stats significantly (4→1 goals, 8→3 shots, 1.71→0.54 xG).
- However, 1 goal for a GK is still suspicious — likely a shot-detection false positive where the ball exits the GK's hands and is counted as a "shot on target → goal."
- **Root cause:** Shot detection uses ball velocity + direction toward goal. GK goal kicks and long throws can trigger false positives.

---

## 5. Goals Comparison

| Video | Feb 9 | Feb 11 | Feb 12 |
|-------|:-----:|:------:|:------:|
| V1 | 0 | 0 | 0 |
| V2 | 6 | 4 | 3 |
| V3 | 7 | 7 | 3 |
| **Total** | **13** | **11** | **6** |

### Status: ⬇️ Trending Down (Likely Overcorrected)

- The EFF_FPS scaling in Feb 12 tightened debounce windows, reducing duplicate goal detections.
- However, the drop from 7→3 (V3) and 6→3 (V2) is too steep — real goals may be getting filtered out.
- **Likely issue:** Goal lookahead window (`max(10, int(2.0 * EFF_FPS))` = 10 frames at stride 5) may be too short to confirm ball entering the net for slower shots.

---

## 6. Passes Comparison

| Video | Feb 9 | Feb 11 | Feb 12 |
|-------|:-----:|:------:|:------:|
| V1 | 3 | 13 | 8 |
| V2 | 69 | 82 | 55 |
| V3 | 74 | 86 | 60 |
| **Total** | **146** | **181** | **123** |

### Status: ⚠️ Inconsistent Across Runs

- **Feb 9 → Feb 11:** Improved (+24%) — stride 3 with better pass detection logic captured more events.
- **Feb 11 → Feb 12:** Dropped (−32%) — EFF_FPS scaling tightened the pass gap threshold (`EFF_FPS * 3` = 15 frames vs old `FPS * 3` = 75 frames). This means passes with ball out of view for more than 3 seconds (real time) are rejected, but the frame count is now measured in processed frames, not real frames.
- **Root cause:** The gap check `gap_frames > EFF_FPS * 3` uses processed frame indices, but the gap between segments is already in processed-frame space. The threshold should likely remain `FPS * 3` or be adjusted to account for the stride.

---

## 7. Tackles / Dribbles / Other Events

| Stat | Feb 9 Total | Feb 11 Total | Feb 12 Total |
|------|:----------:|:-----------:|:-----------:|
| Tackles | 154 | 34 | 20 |
| Challenges | 144 | — | — |
| Dribbles | 133 | 20 | 17 |
| Interceptions | 36 | — | — |

### Status: ⬇️ Significant Drop

- Tackles dropped **154 → 34 → 20** across runs. The Feb 9 numbers were inflated (Player #14 V1 had 12 tackles in a short clip), but 20 total across 3 full matches is too low.
- Dribbles similarly dropped from 133 → 20 → 17.
- **Root cause:** EFF_FPS debounce scaling reduced duplicate event detection, but the thresholds may now be too aggressive.

---

## 8. Summary of Issues

### ✅ Solved
| Issue | Status |
|-------|--------|
| Team color fragmentation (5+ colors) | Fixed via color merge map (Feb 11) |
| Unknown player team assignment | Fixed via FIX #2 (jersey proximity) |
| Stats for nonexistent players | Fixed via Phase 186 filtering |

### ⚠️ Partially Solved
| Issue | Status | Next Step |
|-------|--------|-----------|
| Orphan team colors (GK kits) | FIX #4 applied (Feb 13) | Verify on next run |
| GK false goals | Improved (4→1) | Add GK role-based shot suppression |
| Inflated event counts | Reduced | Fine-tune EFF_FPS thresholds |

### ❌ New Issues (Feb 12)
| Issue | Cause | Proposed Fix |
|-------|-------|--------------|
| Goals undercounted (11→6) | Goal lookahead too short at stride 5 | Increase `goal_lookahead` to `max(15, int(3.0 * EFF_FPS))` |
| Passes undercounted (181→123) | Pass gap threshold scaled incorrectly | Use `FPS * 3` (real-time) instead of `EFF_FPS * 3` |
| Tackles/dribbles too low | Debounce windows too tight | Review tackle proximity thresholds |
| Processing time regression | Memory pressure from sequential runs | Run videos individually or implement proper memory cleanup between runs |

---

## 9. Recommended Next Steps

### Priority 1 — Fix EFF_FPS Overcorrection
The `EFF_FPS` scaling introduced in `fd35c74` correctly accounts for frame skipping in *time-based* calculations, but **incorrectly applies it to frame-index-based thresholds** (gap checks, lookaheads). These need to stay in real-frame space or be converted properly.

### Priority 2 — GK Shot Suppression
Add a rule: if `dominant_class == GK` and shot origin is inside the penalty area facing outward, suppress the shot event. This prevents goal kicks and throws from being counted.

### Priority 3 — Verify Orphan Color Fix
Run one video (V2 or V3) with the FIX #4 code and confirm `player_stats.json` has exactly 2 team colors.

### Priority 4 — Memory Cleanup
Add explicit `gc.collect()` and `torch.cuda.empty_cache()` between videos in the orchestrator polling loop to prevent performance degradation.
