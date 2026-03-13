# Round 11 — Stats Inflation Fix + Distance Bug Fix

**Date:** 2026-03-13
**Commit:** `3d7b0343` on `production-v1.0`
**Target:** Reprocess user `f4dbcb0a` (3 videos) + cached `3235879b` (1 video)

---

## What Changed

Five fixes in `stats/event_logic.py` to reduce stats inflation on long videos:

1. **Pass debounce (NEW)**: 3-second per-passer cooldown. Previously had NO debounce — every ownership transition counted as a pass.
2. **Tackle cooldown**: Increased from 5s → 15s per tackler.
3. **Shot debounce**: Increased from 1s → 5s global window.
4. **Dribble cooldown**: Increased from 1s → 3s per player.
5. **Distance dedup**: Skips duplicate player IDs within a single frame. Fixes near-zero distance (V2 bug: 20K observations → 1m distance).

---

## Steps

### 1. Pull the fix

```bash
cd /root/football
git pull origin production-v1.0
```

Verify commit:
```bash
git log --oneline -1
# Expected: 3d7b0343 fix: Round 11 — reduce stats inflation with tighter event cooldowns
```

### 2. Verify the changes

```bash
grep "EFF_FPS \* 3" stats/event_logic.py | head -3
# Should see: pass debounce (3s), dribble debounce (3s)

grep "EFF_FPS \* 15" stats/event_logic.py
# Should see: tackle debounce (15s)

grep "EFF_FPS \* 5)" stats/event_logic.py
# Should see: shot debounce (5s)

grep "seen_pids" stats/event_logic.py
# Should see: PID dedup in distance calculation
```

### 3. Reprocess all 4 videos

```bash
python orchestrator.py --poll --parallel 3 --tracking_mode bytetrack
```

Or if targeting specific videos:
```bash
# User f4dbcb0a (3 videos)
python pipeline_consolidated.py --video <video_path_1> &
python pipeline_consolidated.py --video <video_path_2> &
python pipeline_consolidated.py --video <video_path_3> &

# Cached 3235879b video (if still in /tmp/)
python pipeline_consolidated.py --video /tmp/3235879b_2026-03-10-02-10-20767443.mp4 &
```

### 4. Verification checklist

After processing, check the stats report for these improvements:

- [ ] **V2 distance**: Players with 10K+ observations should now show realistic distances (100-2000m), not 1-20m
- [ ] **V4 passes**: Should drop from 1,582 to ~400-600 range
- [ ] **V4 tackles**: Should drop from 302 to ~30-80 range
- [ ] **V4 shots**: Should drop from 85 to ~10-25 range
- [ ] **V4 dribbles**: Should drop from 153 to ~20-50 range
- [ ] **Pass debug log**: Look for `pass_debounced: N` in output (N > 0 confirms debounce is active)
- [ ] **Finalize log**: `Consolidated N fragmented tracklets` should still appear (N > 0)

### 5. Generate and push report

```bash
python generate_report.py
git add reports/
git commit -m "feat: Round 11 reprocessing with stats inflation fixes (commit 3d7b0343)"
git push origin production-v1.0
```

---

## Expected Impact

| Metric | V4 Before | V4 Expected After |
|--------|-----------|-------------------|
| Passes | 1,582 | 400-600 |
| Tackles | 302 | 30-80 |
| Shots | 85 | 10-25 |
| Dribbles | 153 | 20-50 |
| V2 Distance | 1-20m per player | 100-2000m per player |

---

## Rollback

If stats drop too aggressively, the cooldown values can be tuned in `stats/event_logic.py`:

```python
# Pass debounce: line ~347
if transition_frame - last_pf < int(EFF_FPS * 3):  # Change 3 to lower value

# Tackle debounce: line ~693
if end_frame - last_tkl < int(EFF_FPS * 15):  # Change 15 to lower value

# Shot debounce: line ~514
shot_debounce = max(10, int(EFF_FPS * 5))  # Change 5 to lower value

# Dribble debounce: line ~238
if t - last_frame > EFF_FPS * 3:  # Change 3 to lower value
```
