# Pitch Homography A/B — Hamburg–Bayern 5-min Broadcast Segment

**Date:** 2026-07-04
**Setup:** identical 5-min gameplay trim of the HSV–Bayern Full HD broadcast
(`hb_5min.mp4`, minutes 1–6), two runs on the worker differing ONLY in
`--pitch_homography`. Broadcast config (FPS 25, NEW_TRACK_THRESH 0.85, stride 3).
Feature commits: `df2005d` (estimator), `07ff57e` (integration).

## Results

| | A: flat scale | B: `--pitch_homography` |
|---|---|---|
| Runtime (tracking) | 1292.6 s | 1305.5 s (+1%) |
| Homography fit rate | — | **25/125 calibration frames (20%)**, 100 reused last-valid; 12 landmarks used on the last fit |
| Frames carrying per-frame H | — | 2441/2500 (97.6%, once first fit landed) |
| Players in output | 16 | 16 |
| Distance median / mean / max (m per 5 min) | 101.3 / 195.7 / 805.6 | 74.7 / 187.9 / 805.6 |
| Events (pass/shot/int/tackle/dribble/touch) | 73 / 9 / 8 / 10 / 14 / 134 | 59 / 9 / 5 / 12 / 19 / 123 |
| Confidence bands (hi/mid/lo) | 149 / 97 / 2 | 130 / 92 / 5 |

## What worked

- **Mechanically sound end-to-end:** model loads, fits with RANSAC, per-frame H
  flows through to the stats engine, negligible runtime cost (+13 s), no crashes,
  and the baseline run confirms the code is inert without the flag.
- When the model does fit, it uses ~12 confident landmarks — comfortably above
  the 4-point minimum, so the fits themselves are well-constrained.

## What didn't (yet)

1. **Fit rate is the blocker: 20%.** 100 of 125 calibration frames fell back to
   the last valid H. On a panning broadcast camera a stale H mis-projects
   everything until the next fit, so most frames carried *some* homography but
   often the wrong one for that camera pose.
2. **Distance realism did not improve** (median 101 m → 75 m per 5 min; real
   players cover 500–700 m). Two compounding causes, neither fixable by
   projection alone:
   - The per-frame speed gate in `stats/event_logic.py` (12 m/s ⇒ max 1.44 m per
     processed frame) discards any apparent movement above it. Under a stale H,
     a camera pan reads as world movement for every player at once and gets
     discarded; ByteTrack fragmentation additionally resets distance
     accumulation per fragment. Distance under-measurement is a stack of issues
     (gate + fragmentation + projection), not a projection-only problem.
   - Two tracks still pin at the identical 805.6 m rate cap — fragment-merge
     inflation masked by the cap, unrelated to projection.
3. **Event counts moved mildly** (passes 73→59, interceptions 8→5) — projection
   changes the 3 m pass gate and speed thresholds; without event-level ground
   truth for this segment we cannot say which side is more correct. Shots
   unchanged at 9 (still likely over-counted).

## Verdict & next steps

Keep the feature **opt-in** (not default). The architecture is right and now
proven end-to-end, but it pays off only once the fit rate is fixed:

1. **Raise fit rate** — likeliest wins: lower the keypoint confidence gate
   (0.50 → sweep 0.30–0.45; failures are frames with <4 confident points while
   successful fits have ~12), and calibrate more often than every 60 source
   frames (pans change the pose in far less than 2.4 s).
2. **Diagnose the keypoint model on this footage** — dump per-calibration
   landmark counts/confidences to see whether the model under-detects on this
   stadium's look; if so, retraining data is the fix (`train_pitch_keypoints.py`
   exists in the v3 archive).
3. **Fix the distance stack in order:** fit rate → then re-examine the 12 m/s
   speed gate under real projection → fragmentation belongs to the planned
   identity-merge rework.
4. Re-run this A/B after (1)–(2); promote to default only when distance medians
   land in a physically plausible range AND event counts hold up against an
   event-level GT list (harness in progress).

## Addendum — tuned re-run (`run_hb5_homog2`, same day)

After the measured tuning (detection conf 0.30→0.05, commit `abaf3dd`):
fit rate **20% → 31%** (31/100 calibrations), shots 9→7 (expected direction:
fewer far-field phantom fast balls), passes 60, distance median back to ~100 m
(speed gate + fragmentation still dominate distance, as analyzed above).
A cadence bug found in the process: `n % 25` only fires on frames that also
pass the stride filter, so calibration ran every lcm(25,3)=75 frames — fixed
to a stride-compatible 24 (`075ca84`); expect ~3× more fit attempts on the
next run. Detection-level sweep confirms the ~31% ceiling stands until the
keypoint model is retrained on our footage domain (prep task parked pending
the contractor's answer on retraining ownership).

The run also produced the first real fragment dataset for the identity-merge
rework (`--dump_fragments`, commit `9cbd017`): 289 fragments over 5 minutes
(~13 per real player), 94% with ReID embeddings, median fragment ~4 s.
