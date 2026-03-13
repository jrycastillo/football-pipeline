# Round 11 — Stats Inflation Fix + Distance Bug Fix (Updated)

**Date:** 2026-03-13
**Commit:** `ff24830b` on `production-v1.0`
**Target:** Reprocess user `f4dbcb0a` (3 videos) + cached `3235879b` (1 video)

---

## What Changed

Four fixes in `stats/event_logic.py` to reduce stats inflation on long videos:

1. **Tackle cooldown**: Increased from 5s → 15s per tackler.
2. **Shot debounce**: Increased from 1s → 5s global window.
3. **Dribble cooldown**: Increased from 1s → 3s per player.
4. **Distance dedup**: Skips duplicate player IDs within a single frame. Fixes near-zero distance (V2 bug: 20K observations → 1m distance).

**Note:** Pass debounce was initially included but removed after analysis showed V1/V2/V3 pass counts (96, 60, 283) are already at realistic levels. V4's pass inflation (1,582) is caused by team imbalance (30W/7R), not rapid ownership oscillations.

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
# Expected: ff24830b fix: Remove pass debounce — passes already at realistic levels in R10 V1/V2/V3
```

### 2. Verify the changes

```bash
grep "EFF_FPS \* 3" stats/event_logic.py | head -3
# Should see: dribble debounce (3s) — NO pass debounce

grep "EFF_FPS \* 15" stats/event_logic.py
# Should see: tackle debounce (15s)

grep "EFF_FPS \* 5)" stats/event_logic.py
# Should see: shot debounce (5s)

grep "seen_pids" stats/event_logic.py
# Should see: PID dedup in distance calculation

grep "_last_pass_frame" stats/event_logic.py
# Should return NOTHING (pass debounce removed)
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
- [ ] **V1/V2/V3 passes**: Should remain at similar levels to Round 10 (no pass debounce applied)
- [ ] **V4 tackles**: Should drop from 302 to ~30-80 range
- [ ] **V4 shots**: Should drop from 85 to ~10-25 range
- [ ] **V4 dribbles**: Should drop from 153 to ~20-50 range
- [ ] **Finalize log**: `Consolidated N fragmented tracklets` should still appear (N > 0)

### 5. Generate and push report

```bash
python generate_report.py
git add reports/
git commit -m "feat: Round 11 reprocessing with stats inflation fixes (commit ff24830b)"
git push origin production-v1.0
```

---

## Expected Impact

| Metric | V4 Before | V4 Expected After |
|--------|-----------|-------------------|
| Passes | 1,582 | ~1,582 (unchanged — fix requires team balance) |
| Tackles | 302 | 30-80 |
| Shots | 85 | 10-25 |
| Dribbles | 153 | 20-50 |
| V2 Distance | 1-20m per player | 100-2000m per player |

---

## Rollback

If stats drop too aggressively, the cooldown values can be tuned in `stats/event_logic.py`:

```python
# Tackle debounce: line ~688
if end_frame - last_tkl < int(EFF_FPS * 15):  # Change 15 to lower value

# Shot debounce: line ~502
shot_debounce = max(10, int(EFF_FPS * 5))  # Change 5 to lower value

# Dribble debounce: line ~235
if t - last_frame > EFF_FPS * 3:  # Change 3 to lower value
```

## Next Steps (Future Rounds)

V4's pass inflation (1,582) is caused by team imbalance (30 White vs 7 Red). Fixing team clustering to produce balanced 11v11 assignments will resolve pass inflation without needing pass-specific cooldowns.
