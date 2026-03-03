# H100 Agent Instructions — Round 9 Reprocessing

**Date:** 2026-02-26
**Fix Commit:** `b04d368` on `production-v1.0`
**Fix:** Phase 216 remap picks primary track instead of summing duplicates

---

## 1. Pull the Latest Code

```bash
cd /root/Babak
git pull origin production-v1.0
```

Verify you're on commit `b04d368`:

```bash
git log --oneline -1
# Expected: b04d368 fix: Phase 216 remap picks primary track instead of summing duplicates
```

---

## 2. What Changed

The Phase 216 remap in `stats/metrics.py` was **summing** stats when multiple ByteTrack track IDs mapped to the same jersey number. This caused 2-22x inflation on full-length matches (V3 had 26 goals, V2 had 17 goals).

The fix **picks the primary track** (most distance + touches) per jersey instead of summing. Frame count remap also changed from sum to max.

**Do NOT change any code — just pull and run.**

---

## 3. Reprocess All 3 Videos

Run each video with the same config as Round 8:

```bash
python orchestrator.py --config config_h100.yaml --video_ids 162b6abe208946b --no_video_output
python orchestrator.py --config config_h100.yaml --video_ids 14c0f4e8c4af40d --no_video_output
python orchestrator.py --config config_h100.yaml --video_ids 69a33466fc234db --no_video_output
```

---

## 4. Save Results

Copy the output stats to a dated report directory:

```bash
mkdir -p reports/2026-02-26
python generate_report.py --date 2026-02-26
```

---

## 5. Verification Checklist

After each video, check these in `player_stats.json`:

| Check | Expected | Red Flag |
|-------|----------|----------|
| Total goals per match | 0-5 | More than 6 |
| Max passes per player | 30-80 | More than 100 |
| Max observations per player | Less than total frames | Obs > total video frames |
| Any player with >3 goals | Should be rare | Multiple players with 5+ goals |
| Match score (team totals) | 0-3 vs 0-3 | Double digits |
| Phase 216 log lines | Should show "picked track X, dropped N duplicate track(s)" | No Phase 216 log lines = not applied |

Look for these log lines during processing to confirm the fix is active:

```
[Phase 216] Jersey #XX: picked track YY (weight=ZZZ), dropped N duplicate track(s)
[Phase 216] Remapped X track IDs to jersey numbers (pick-primary, no summing)
```

---

## 6. Commit Results

```bash
git add reports/2026-02-26/
git commit -m "feat: Round 9 reprocessing with Phase 216 pick-primary fix (commit b04d368)"
git push origin production-v1.0
```
