# Next Steps — Post R13 Improvement Plan

**Date:** 2026-03-19
**Current state:** R13 on local MPS recovered passes (168), shots (15), dribbles (10) to near-R8 levels. Tackles (6) still below target (21). Team imbalance persists (13G vs 6W).

---

## Priority 1: Validate on H100

**Goal:** Confirm R13 code produces R8-level or better results on H100 with full CUDA detection.

- Push current code to H100 and reprocess V1 (`1e942fd8a6344bd`)
- Expected: passes ~250+, tackles ~20+, shots ~15+, players ~25-30
- MPS gives ~50-60% of H100 event counts, so R13's 168 passes should scale to ~280-340 on H100
- This is the single most impactful step — if H100 results match R8, remaining fixes are refinements

---

## Priority 2: Fix Team Imbalance (13G vs 6W)

**Problem:** White team consistently under-detected. Only 6 White players found vs expected 11.

**Root causes to investigate:**
1. **Color classifier bias**: White jerseys reflecting green pitch may still be misclassified as Green despite S>70 fix. Check `match_kits.json` — if White players are getting Green labels, they merge into the wrong team.
2. **JNR fragmentation**: White players may be spread across many jersey IDs that don't meet the registration threshold. Check `raw_tracks.json` for unregistered White tracks.
3. **Low observation counts**: White players may have short tracks (< minimum observation threshold) and get filtered out entirely.

**Potential fixes:**
- Lower soft-registration threshold for under-represented team
- Post-hoc team balancing: if one team has >14 players, redistribute borderline players
- Improve color classifier's white detection in varied lighting

---

## Priority 3: Improve Tackle Detection

**Problem:** 6 tackles in R13 (R8 had 21). Only 3 players registered any tackles.

**Root causes:**
1. **Proximity window too short on MPS**: stride=3 means fewer frames to detect player proximity events. ByteTrack gaps between detections cause missed tackle windows.
2. **`TACKLE_PX` threshold**: Currently 100px. May be too tight for MPS resolution. Consider adaptive threshold based on average player bounding box size.
3. **Tackle cooldown**: 5s is correct (R8 value), but the tackle detection logic requires specific frame sequences (approach → contact → ball change) that are harder to capture at stride=3.

**Potential fixes:**
- Increase `TACKLE_PX` from 100 to 120-150 for MPS processing
- Reduce minimum proximity frames required for tackle detection
- Consider stride-aware scaling: multiply proximity thresholds by `VID_STRIDE` factor

---

## Priority 4: Fix Ghost Players (Zero-Stats Players)

**Problem:** Player #35 (White) has 1,138 observations but 0 distance, 0 touches, 0 everything.

**Root causes:**
1. **Phase 216 top-N merge**: The primary fragment for #35 may have 0 stats while secondary fragments have the actual stats. If primary is selected by weight (observations) but has no ball proximity, stats are lost.
2. **Distance calculation**: 0.0m distance with 1,138 observations means all position data is identical or missing. Likely a stationary detection (camera artifact, billboard, or sideline person).

**Potential fixes:**
- Add minimum distance filter: if distance < 5m over 500+ observations, flag as artifact
- In top-N merge, also consider non-primary fragments for distance/movement stats
- Filter out tracks with 0 movement from final output

---

## Priority 5: Improve Pass Accuracy Calculation

**Problem:** Many players show unrealistically low pass accuracy (7-17%). Total pass accuracy ~37%.

**Root causes:**
1. **Ownership mapping noise**: Ball detection gaps create false ownership transitions. A→None→B gets counted as an inaccurate pass from A even if the ball stayed with A.
2. **Crowded areas**: Multiple players near the ball cause rapid ownership switching, each switch counted as a pass attempt.
3. **Long ball gaps**: When ball is undetected for >25 frames, the interpolated position may assign ownership incorrectly.

**Potential fixes:**
- Require minimum ownership duration (e.g., 0.5s) before counting a pass
- Increase `MAX_GAP` for ball interpolation from 25 to 50 frames
- Add pass validation: a "pass" must have the ball travel minimum distance (e.g., >50px)

---

## Priority 6: Multi-Video Validation

**Goal:** After H100 validation on V1, run V2 and V3 to ensure fixes are generalizable.

- V2 (`67b36815`, 870MB): Different lighting, different teams
- V3 (`7d4ee81d`, 1.1GB): Strong sunlight, challenging color classification
- Compare all three against R8 baselines
- If any video regresses, investigate per-video tuning vs universal parameters

---

## Summary Timeline

| Step | Priority | Effort | Impact |
|------|----------|--------|--------|
| H100 reprocess V1 | P1 | Low (just deploy) | High — validates all R13 fixes |
| Fix team imbalance | P2 | Medium | High — affects all downstream stats |
| Improve tackle detection | P3 | Medium | Medium — tackles are 1 stat category |
| Fix ghost players | P4 | Low | Low — cosmetic, affects 1-2 players |
| Pass accuracy | P5 | High | Medium — complex ownership logic |
| Multi-video validation | P6 | Low | High — confirms generalization |
