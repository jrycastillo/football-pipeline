# Processing Speed Analysis — Why H100 Was Slow & V4 Failed

**Date:** 2026-03-16
**Author:** AI Pipeline Analysis
**Context:** Round 11 reprocessing on H100 (2026-03-14) took 15-17 hours per video and V4 timed out at 24 hours

---

## Current V4 Local Processing Status

| Metric | Value |
|--------|-------|
| Progress | **16.2%** — Frame 78,120 / 482,880 |
| YOLO Frames Done | 11,160 / 68,982 |
| Elapsed | 3.1 hours |
| Speed | 1.0 YOLO fps (M4 Pro MPS) |
| ETA | **~2026-03-17 07:30 AM** (~16 hrs remaining) |
| Settings | `--vid_stride 7 --no_video_output --tracking_mode bytetrack` |

---

## Root Cause: `--vid_stride` Not Passed to Pipeline

**The single root cause of ALL processing slowness is a missing CLI argument.**

The orchestrator (`orchestrator.py`) launches the pipeline like this:

```python
cmd = [sys.executable, "pipeline_consolidated.py",
       "--video", local_video_path,
       "--output_dir", output_dir,
       "--locking_mode", str(locking_mode)]
```

The `--vid_stride` argument is **only passed if explicitly provided** by the operator:

```python
if vid_stride:
    cmd.extend(["--vid_stride", str(vid_stride)])
```

When not provided (which is the default for `--poll` mode), the pipeline falls back to its own default:

```python
parser.add_argument("--vid_stride", type=int, default=1, ...)
```

**`default=1` means process EVERY frame.** The `config.yaml` value `VID_STRIDE: 3` is only used for stats timing calculations (`EFF_FPS = FPS / VID_STRIDE`), NOT for actual frame skipping.

---

## Impact: 3-8x More Work Than Intended

### Video Properties (Actual)

| Video | FPS | Resolution | Total Frames | Duration |
|-------|-----|-----------|-------------|----------|
| V1 (`1e942f`) | ~30 | 1920x1080 | ~190,000 | ~105 min |
| V2 (`82d037`) | ~30 | 1920x1080 | ~190,000 | ~105 min |
| V3 (`e4d860`) | 30 | 1920x1080 | ~180,000 | ~100 min |
| V4 (`f56151`) | **60** | 1280x720 | **482,880** | **134 min** |

### Frames Actually Processed vs Intended

| Video | stride=1 (ACTUAL) | stride=3 (INTENDED) | Overwork Factor |
|-------|-------------------|---------------------|-----------------|
| V1 | ~190,000 | ~63,000 | **3.0x** |
| V2 | ~190,000 | ~63,000 | **3.0x** |
| V3 | ~180,000 | ~60,000 | **3.0x** |
| V4 | **482,880** | ~160,000 | **3.0x** (and 2.7x more total than V3) |

### Processing Time Explanation

| Video | H100 Time | Expected w/ stride=3 | Why |
|-------|-----------|----------------------|-----|
| V1 | 15.6 hrs | ~5.2 hrs | 3x overwork (stride=1) |
| V2 | 14.8 hrs | ~4.9 hrs | 3x overwork (stride=1) |
| V3 | 17.3 hrs | ~5.8 hrs | 3x overwork (stride=1) |
| V4 | **TIMEOUT (24h)** | ~8.0 hrs (stride=3) | 3x overwork + 60fps = 482K frames |

The config comments claim "~1.5-2 hours per 90-min match" with stride=3. The actual results of 15-17 hours confirm stride=1 was used — processing 3x more frames at ~5 hours each = ~15 hours.

---

## Why V4 Specifically Failed

V4 has two compounding issues:

1. **60fps instead of 30fps**: Double the frame rate means double the frames per minute of video
2. **134 minutes**: Longer than the other videos (100-105 min)

Combined: V4 has **482,880 frames** vs V3's **~180,000 frames** — that's **2.7x more frames**. At stride=1, the H100 needed:

- V3 took 17.3 hrs at ~180K frames
- V4 needed ~17.3 × 2.7 = **~46.7 hours** — well beyond the 24-hour timeout

---

## The Fix (Two Parts)

### 1. Orchestrator: Pass `--vid_stride` from config (Recommended)

The orchestrator should read `VID_STRIDE` from config and always pass it:

```python
# In orchestrator.py, around line 174:
vid_stride_cfg = CONFIG.get("heuristics", {}).get("VID_STRIDE", 3)
cmd.extend(["--vid_stride", str(vid_stride or vid_stride_cfg)])
```

This ensures the config's `VID_STRIDE=3` is always applied for frame skipping, not just stats timing.

### 2. Handle 60fps Videos: Auto-detect and adjust stride

For videos at 60fps, `VID_STRIDE=3` gives 20 effective fps — still more than needed. The pipeline should auto-detect the video FPS and scale the stride:

```python
# Target ~8-10 effective fps (matches 25fps/stride=3 behavior)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
if actual_fps > 30:
    adjusted_stride = max(vid_stride, int(round(actual_fps / 8.33)))
    # 60fps → stride=7, 50fps → stride=6, 30fps → stride=3 (unchanged)
```

---

## Expected Improvement After Fix

| Video | Current (stride=1) | Fixed (stride=3) | Fixed + 60fps Auto |
|-------|-------------------|-------------------|-------------------|
| V1 | 15.6 hrs | **~5.2 hrs** | ~5.2 hrs |
| V2 | 14.8 hrs | **~4.9 hrs** | ~4.9 hrs |
| V3 | 17.3 hrs | **~5.8 hrs** | ~5.8 hrs |
| V4 | TIMEOUT | ~15.5 hrs | **~5.5 hrs** |

With both fixes, all 4 videos should complete in **~5-6 hours each** on H100.

---

## Local V4 Processing (Current)

Processing V4 on local M4 Pro with `--vid_stride 7`:
- 68,982 YOLO frames at ~1.0 fps
- Estimated **~18 hours total** (ETA: March 17, ~7:30 AM)
- This is slower than H100 would be with proper stride, but will complete successfully without timeout

---

## Stats Quality Note

Using stride=7 for V4 (60fps) gives ~8.57 effective fps — very close to the 8.33 effective fps used for V1/V2/V3 (25-30fps / stride=3). Stats quality should be equivalent. The config's `EFF_FPS = 25/3 = 8.33` is used for cooldown timing and matches closely enough (3% difference).

---

## Summary

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| ALL videos 3x slower than expected | `--vid_stride` defaults to 1, not passed from orchestrator | Pass config's VID_STRIDE to CLI |
| V4 timeout (24h) | 60fps × 134min = 482K frames at stride=1 | Auto-detect 60fps, use stride=7 |
| V4 stats were 3-10x inflated (Round 10) | Team imbalance (30W/7R) + no event cooldowns | Round 11 cooldown fixes (tackle 15s, shot 5s, dribble 3s) |
