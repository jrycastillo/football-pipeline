# Color Classifier Fix Report — 2026-02-25

**Commit:** `605b101` on `production-v1.0`
**Files Changed:** `vision/color_classifier.py`, `stats/metrics.py`

---

## Problem

White jerseys were never detected. The classifier labeled every Algeria player (white kit) as **Green** or **Blue** due to green pitch reflection on the white fabric. This caused:

- Video 3 (ALG vs MAR): 0 White, 6 Green, 5 Blue — all wrong
- Kit discovery: `["Green", "Red"]` instead of `["White", "Red"]`
- Team balance: 6/15 (should be ~11/11)

### Root Cause: Three Cascading Failures

1. **Grass mask deletes white pixels** — White jerseys reflect the green pitch, giving them S=55-68. The grass mask (H 35-90, S > 25) removed these as "grass."

2. **Green early-exit fires on background** — The bounding box includes grass behind the player. With threshold `ratio > 0.35` on the full crop, background grass triggered a "Green" classification before the main classifier even ran.

3. **Red numbers dominate survivors** — When the grass mask spared some pixels, the red jersey numbers on Algeria's white shirts took up enough area to classify as "Red."

---

## Fix: Three Changes in `color_classifier.py`

| Parameter | Before | After | Reasoning |
|-----------|--------|-------|-----------|
| Achromatic threshold | `S > 55` | `S > 70` | White jerseys with green tint have S=55-68. Now classified as achromatic (White). Genuine colors have S > 150. |
| Torso ROI | 15-65% height | 15-45% height | Targets shoulders only. Avoids red chest numbers and grass at lower body. |
| Green early-exit | `S > 80`, `ratio > 0.35`, full crop | `S > 80`, `ratio > 0.60`, center crop only | Tight center crop (20-50% height, 40-60% width) prevents background grass from inflating ratio. |

### Additional Fix in `stats/metrics.py`

**Achromatic-aware orphan assignment** in `_cluster_teams` FIX#4:

When one team is White (achromatic), `_hue_dist()` always returns 180 (max), so orphan colors were never assigned to the White team. Green orphans went to Red instead.

New rule: If orphan hue distance to the chromatic team is > 30°, assign to the achromatic team.

| Orphan Color | Before (broken) | After (fixed) |
|-------------|-----------------|---------------|
| Green | Red (dist 60 vs 180) | **White** (60 > 30° threshold) |
| Blue | Red (dist 60 vs 180) | **White** (120 > 30° threshold) |
| Orange | Red (dist 15 vs 180) | Red (15 ≤ 30° — correct) |

---

## Verification: All 4 Local Test Videos

### Video 1: `121364_0.mp4` — Green vs Red

![121364](../output/color_report/121364_green_vs_red.jpg)

| Metric | Result |
|--------|--------|
| Kit discovery | **Red vs Green** |
| Classification | Green: 9, Red: 6, Unknown: 7 |
| Status | Correct — green jerseys detected properly, no regression |

### Video 2: `69a33466fc234db.mp4` — White vs Red (ALG vs MAR, frame 40000)

![69a33_40k](../output/color_report/69a33_white_vs_red_40k.jpg)

| Metric | Result |
|--------|--------|
| Kit discovery | **Red vs White** |
| Classification | White: 10, Red: 7, Green: 2 |
| Status | Correct — Algeria white jerseys now detected (was 0 White before) |

### Video 2: `69a33466fc234db.mp4` — White vs Red (ALG vs MAR, frame 80850)

![69a33_80k](../output/color_report/69a33_white_vs_red_80k.jpg)

| Metric | Result |
|--------|--------|
| Classification | White: 8, Red: 7, Green: 3, Unknown: 1 |
| Status | Correct — white boxes align with Algeria players, red with Morocco |

### Video 3: `clipped_ikorudo_tornadoes.mp4` — White vs Red

![ikorudo](../output/color_report/ikorudo_white_vs_red.jpg)

| Metric | Result |
|--------|--------|
| Kit discovery | **White vs Red** |
| Classification | Red: 9, White: 7, Unknown: 1 |
| Status | Correct — clean Red/White separation |

### Video 4: `endpoint_14c0f4e8c4af40d.mp4` — Green vs White (TUN vs ALG)

![14c0f4](../output/color_report/14c0f4_green_vs_white.jpg)

| Metric | Result |
|--------|--------|
| Kit discovery | **Green vs White** |
| Classification | White: 10, Green: 10, Blue: 1 |
| Status | Correct — Green and White teams balanced |

---

## Kit Discovery Summary (All Videos)

| Video | Before Fix | After Fix | Status |
|-------|-----------|-----------|--------|
| `121364_0` | Red vs Green | **Red vs Green** | No change (correct) |
| `69a33466fc234db` | Green vs Red | **Red vs White** | Fixed |
| `clipped_ikorudo` | White vs Red | **White vs Red** | No change (correct) |
| `endpoint_14c0f4e8` | Green vs White | **Green vs White** | No change (correct) |

---

## Critical Thresholds (Do Not Change)

| Parameter | Value | Why |
|-----------|-------|-----|
| Achromatic threshold | `S > 70` | S=160 broke ALL color detection — everything became White |
| HSV_COLOR_RANGES S_min | 40-80 | S=160 made only ultra-vivid colors detectable |
| Green early-exit saturation | `S > 80` | S=175 made green jersey detection near-impossible |
| Green early-exit ratio | `> 0.60` on center crop | `> 0.35` on full crop triggered on background grass |
| Grass mask | `S > 70` | Matches achromatic threshold — pixels removed as grass won't be misclassified |
| Orphan hue threshold | `> 30°` from chromatic team | Below 30° = likely same team. Above 30° = goes to achromatic team |
