# H100 Reprocessing Instructions — Round 12

**Date:** 2026-03-16
**Branch:** `production-v1.0`
**Commit:** (pending push)

---

## What Changed (3 Fixes)

### 1. Phase 216 Smart Merge (metrics.py)
- **Before:** Pick-primary — only kept stats from the single longest track fragment per jersey
- **After:** Sum EVENT stats (passes, tackles, shots, dribbles, interceptions) across ALL fragments, keep primary for accumulative stats (distance, touch_frames)
- **Why:** Pick-primary discarded ~60-80% of legitimate events from non-primary time periods. A player with 5 track fragments only had 1/5 of their events counted.

### 2. Moderate Cooldowns (event_logic.py)
| Event | R10 | R11 | R12 (Now) |
|-------|-----|-----|-----------|
| Tackle | 5s | 15s | **8s** |
| Shot | 1s | 5s | **3s** |
| Dribble | 1s | 3s | **2s** |

- R11 cooldowns were calibrated for V4's extreme case (482K frames at stride=1)
- With proper stride=3, the effective frame rate is correct and moderate cooldowns give realistic counts

### 3. Orchestrator VID_STRIDE Fix (orchestrator.py)
- **Before:** `--vid_stride` only passed if explicitly provided by operator. Default: stride=1 (every frame)
- **After:** Always reads `VID_STRIDE` from config.yaml (default=3) and passes it
- **Impact:** 3x faster processing (5-6 hrs per video instead of 15-17 hrs). V4 (60fps) will now complete within timeout.

---

## Steps

```bash
# 1. Pull latest
cd ~/Babak
git pull origin production-v1.0

# 2. Verify the fixes
grep "EFF_FPS \* 8" stats/event_logic.py     # Tackle 8s
grep "EFF_FPS \* 3\b" stats/event_logic.py   # Shot 3s
grep "EFF_FPS \* 2" stats/event_logic.py     # Dribble 2s
grep "vid_stride_cfg" orchestrator.py          # Orchestrator fix
grep "_EVENT_KEYS" stats/metrics.py            # Phase 216 smart merge

# 3. Reprocess ALL 4 videos (orchestrator now passes stride=3 automatically)
python orchestrator.py --poll --parallel 4 --tracking_mode bytetrack

# OR if you want to run specific videos manually:
python pipeline_consolidated.py --video <path> --output_dir output/<dir> \
  --locking_mode 2 --no_video_output --tracking_mode bytetrack --vid_stride 3

# For V4 (60fps) specifically, use stride 7:
python pipeline_consolidated.py --video <v4_path> --output_dir output/v4 \
  --locking_mode 2 --no_video_output --tracking_mode bytetrack --vid_stride 7
```

---

## Expected Processing Times (with stride=3 fix)

| Video | R11 Time (stride=1) | R12 Expected (stride=3) |
|-------|---------------------|------------------------|
| V1 | 15.6 hrs | **~5 hrs** |
| V2 | 14.8 hrs | **~5 hrs** |
| V3 | 17.3 hrs | **~6 hrs** |
| V4 | TIMEOUT (24h) | **~6 hrs** (stride=7 for 60fps) |

---

## Expected Stats Improvement

| Metric | R11 (V1) | R11 (V2) | R11 (V3) | R12 Expected | Real Range |
|--------|----------|----------|----------|-------------|------------|
| Passes | 96 | 60 | 283 | 150-400 | 200-500 |
| Tackles | 9 | 5 | 20 | 30-60 | 30-60 |
| Shots | 7 | 2 | 10 | 15-30 | 15-30 |
| Dribbles | 2 | 2 | 3 | 15-40 | 20-50 |

The biggest improvement comes from Phase 216 smart merge (recovers events from all track fragments) and correct EFF_FPS timing (stride=3 instead of stride=1).

---

## Verification After Processing

```bash
# Check stats totals per video
python -c "
import json, sys
stats = json.load(open(sys.argv[1]))
passes = sum(p['stats']['passes_total'] for p in stats.values())
tackles = sum(p['stats']['tackles_total'] for p in stats.values())
shots = sum(p['stats']['shots_on_target_total'] for p in stats.values())
dribbles = sum(p['stats']['dribbles_total'] for p in stats.values())
teams = {}
for p in stats.values():
    t = p['team']
    teams[t] = teams.get(t, 0) + 1
print(f'Passes: {passes} | Tackles: {tackles} | Shots: {shots} | Dribbles: {dribbles}')
print(f'Teams: {teams}')
" output/<dir>/player_stats.json
```

Expected ranges per video:
- Passes: 150-500 (up from 60-283)
- Tackles: 30-60 (up from 5-20)
- Shots: 15-30 (up from 2-10)
- Dribbles: 15-40 (up from 2-3)
