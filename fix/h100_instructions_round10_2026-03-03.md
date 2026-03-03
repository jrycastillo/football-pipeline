# H100 Agent Instructions — Round 10 Reprocessing

**Date:** 2026-03-03
**Fix Commit:** `9ac76ff` on `production-v1.0`
**Fix:** Re-enable finalize_bindings for Mode 2 (tracklet consolidation)

---

## 1. Pull the Latest Code

```bash
cd /root/Babak
git pull origin production-v1.0
```

Verify you're on commit `9ac76ff`:

```bash
git log --oneline -1
# Expected: 9ac76ff Update pipeline consolidation and add round 9 instructions
```

---

## 2. What Changed

`finalize_bindings()` in `pipeline_consolidated.py` was previously a no-op in Mode 2 because it iterated `self.alpha` (Mode 3 only, always empty). Now it uses `vote_counts` (Mode 2) to consolidate orphan tracklets that didn't reach the lock threshold.

ByteTrack creates 50+ track IDs per player across a full match. Many of these fragments accumulate JNR votes but never lock (too few consistent reads on a single track). These orphans were previously lost — now they get consolidated into their correct jersey number.

This is safe because the Phase 216 remap (commit `b04d368`) picks the primary track per jersey instead of summing.

**Do NOT change any code — just pull and run.**

---

## 3. Reprocess All 3 Videos

Run each video with the same config as previous rounds:

```bash
python orchestrator.py --config config_h100.yaml --video_ids 162b6abe208946b --no_video_output
python orchestrator.py --config config_h100.yaml --video_ids 14c0f4e8c4af40d --no_video_output
python orchestrator.py --config config_h100.yaml --video_ids 69a33466fc234db --no_video_output
```

---

## 4. Save Results

```bash
mkdir -p reports/2026-03-03
python generate_report.py --date 2026-03-03
```

---

## 5. Verification Checklist

### A. Finalize_bindings is active

Look for these log lines during processing:

```
🔍 [IdentityManager] Starting Bayesian Tracklet Consolidation...
🔄 [Finalize] Consolidating Track XXX -> Jersey #YY (Evidence: Z.Z)
✅ [Finalize] Consolidated N fragmented tracklets.
```

**N should be > 0.** If N = 0, the fix is not working. On full-length matches, expect 20-100+ consolidations.

### B. Stats are not inflated (Phase 216 still working)

| Check | Expected | Red Flag |
|-------|----------|----------|
| Total goals per match | 0-5 | More than 6 |
| Max passes per player | 30-80 | More than 100 |
| Match score (team totals) | 0-3 vs 0-3 | Double digits |
| Phase 216 log | "pick-primary, no summing" | Missing or shows summing |

### C. Compare with Round 9 results

| Metric | Round 9 (baseline) | Round 10 (expected) |
|--------|-------------------|---------------------|
| V1 players | 27 | Same or fewer (consolidation) |
| V2 players | 31 | Closer to 22-25 |
| V3 players | 32 | Closer to 22-25 |
| V2 team balance | Green 20 / White 11 | More balanced |
| V3 team balance | Red 20 / White 12 | More balanced |
| V1 score | 0-2 | Same |
| V2 score | 1-0 | Same or similar |
| V3 score | 0-1 | Same or similar |

The main improvement should be **player count closer to 22** and **better team balance** (closer to 11v11). Stats per player should be richer (more touches, passes) since orphan fragments are now attributed correctly.

---

## 6. Commit Results

```bash
git add reports/2026-03-03/
git commit -m "feat: Round 10 reprocessing with finalize_bindings enabled (commit 9ac76ff)"
git push origin production-v1.0
```
