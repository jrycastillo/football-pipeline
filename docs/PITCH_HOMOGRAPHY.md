# Pitch Keypoint Homography — Real Pixel→Meter Projection

**Status:** implemented, opt-in via `--pitch_homography` (July 2026)
**Files:** `vision/pitch_homography.py` (estimator + landmark schema), `test_pitch_homography.py` (unit tests), integration in `pipeline_consolidated.py` + `stats/event_logic.py`
**Model:** `models/pitch_keypoints.pt` (54 MB YOLO keypoint model, 32 pitch landmarks — not in git; lives on the worker alongside the other models)

---

## 1. Why

Every spatial number the pipeline produces — player distance, pass length, shot distance, xG inputs, penalty-box membership, possession zones — goes through one function: `Camera.project_point(px) → meters`. Until now that projection was a **flat diagonal scale** (1920 px ≡ 105 m, 1080 px ≡ 68 m). A broadcast camera is perspective: a 50 px step near the far touchline covers several times more real ground than the same 50 px in the near corner, and the camera pans/zooms continuously. The flat scale therefore systematically distorts distances, speeds (shot detection gates on m/s), and every box/zone test.

This feature replaces the flat scale with a **real projective homography** fitted from the pitch markings visible in each frame.

## 2. How it works

```
frame ──► YOLO keypoint model (32 named pitch landmarks)
              │  (x,y,conf) per landmark
              ▼
    filter: conf ≥ 0.50 AND inside frame          (invisible landmarks emit (0,0))
              │  ≥ 4 landmarks required
              ▼
    cv2.findHomography(camera_px, pitch_meters, RANSAC, 8 px)
              │
              ▼
    H (3×3): camera px → pitch meters   ──►  Camera.update(H)
                                        ──►  frame_data["H"] (per-frame record)
```

- **Landmark schema:** 32 named points (corners, penalty/goal box vertices, penalty spots, halfway line, center circle) with fixed model-output index order, defined in `PITCH_VERTICES_M`. The grid is expressed in **our** 105×68 m convention (origin top-left, goals at x=0/105) so the fitted H is a drop-in for the existing `Camera`. The source schema assumed a 105×70 pitch; width-relative points were re-centered, box dimensions preserved.
- **Partial pitch is normal:** broadcast views rarely show more than a third of the pitch. Only visible landmarks are used; when fewer than 4 are confident, the **last valid H is reused** (`frames_reused` counter) so projection never dies mid-video.
- **Cadence:** the estimator runs on the existing pitch-calibration step (every 60 source frames, ~2.4 s), not per frame — measured overhead is negligible.
- **Per-frame H for post-hoc stats (the key design point):** the stats engine runs after the video pass and previously built its *own* flat `Camera` — it never saw calibration at all. Now every processed frame stores the H that was valid at capture (`frame_data["H"]`), and `AdvancedEventDetector` points its camera at the frame's matrix before each projection (ownership distances, distance accumulation, pass/shot/save/foul geometry, box checks). With a panning camera, a single global H would be wrong; per-frame is the correct unit.

## 3. Usage

```bash
python3 pipeline_consolidated.py --video match.mp4 --tracking_mode bytetrack \
    --output_dir output/run --vid_stride 3 --enable_reid true \
    --pitch_homography
```

- Model path override: `env.PITCH_KP_WEIGHTS` in `config.yaml` (default `models/pitch_keypoints.pt`).
- **Fully backward-compatible:** without the flag, or if the model file is missing, no frame carries an H and every code path behaves exactly as before (verified by the synthetic stats regression suite — byte-identical outputs).
- Run log reports `[PitchHomography] fit stats: {frames_seen, frames_fitted, frames_reused, ...}` — `frames_fitted / frames_seen` is the health signal. A low fit rate means the keypoint model doesn't see this footage's markings well (train more landmarks data, see §6).

## 4. Validation (Hamburg–Bayern broadcast clip)

Method: identical 5-minute gameplay segment of the HSV–Bayern Full HD broadcast (`hb_5min.mp4`, minutes 1–6), two runs on the worker differing ONLY in the flag: baseline (flat scale) vs `--pitch_homography`. Broadcast config (FPS 25, NEW_TRACK_THRESH 0.85).

Results (full numbers: `reports/2026-07-04/pitch_homography_ab.md`):
- **Mechanically sound end-to-end** — +1% runtime, per-frame H on 97.6% of frames once ready, baseline run confirms the code is inert without the flag.
- **Fit rate is the current blocker: 20%** (25/125 calibration frames; the rest reused a stale last-valid H, which mis-projects during camera pans). Successful fits used ~12 landmarks, so the fits themselves are solid — the failures are frames with <4 confident points.
- **Distance realism did not improve yet** — the per-frame 12 m/s speed gate plus ByteTrack fragmentation dominate distance error; with a stale H, pans read as world movement and get discarded. Projection is one layer of a stack.
- Verdict: stays **opt-in**. Promote only after the fit-rate fixes below and a re-run of this A/B against an event-level GT list.

## 5. Provenance

The approach and landmark schema originate from the contractor's v3 prototype evaluation (see `reports/2026-07-04/nabeel_v3_evaluation.md`): the one component of that codebase judged worth adopting. The estimator was re-implemented for this repo (meters convention, our Camera interface, counters, lazy model loading), unit-tested standalone, and wired into the stats engine — which the prototype did not do (it only fed a radar overlay).

## 6. Limitations & next steps

- **Fit rate first:** measured 20% on the HB broadcast clip. Two cheap levers before any retraining: sweep the keypoint confidence gate (0.50 → 0.30–0.45) and calibrate more often than every 60 source frames (a pan changes the camera pose in well under 2.4 s). Re-run the A/B after.
- **Model transfer:** `pitch_keypoints.pt` was trained on broadcast footage; on non-broadcast/training videos (webcam angles, small pitches without standard markings) the fit rate may be poor — the estimator then quietly falls back to the flat scale for unfitted stretches (or entirely). Check the fit-stats line before trusting spatial numbers.
- **Distance is a stack:** projection is one layer; the 12 m/s per-frame speed gate in `stats/event_logic.py` and ByteTrack fragmentation both suppress distance independently. Fix order: fit rate → speed gate under real projection → identity-merge rework.
- **Retraining path:** `train_pitch_keypoints.py` exists in the v3 archive; landmark annotation on our footage domains is the natural extension (annotation task — same workflow as ground-truth clips).
- **Interaction with xG:** shot distance/angle now change on broadcast footage; the xG model constants were calibrated on the flat scale, so xG shifts should be reviewed against verified shots once the admin workflow produces them.
- **Box-width convention:** the landmark grid uses the schema's 41.0 m penalty-box width; `stats/event_logic.py` uses 40.3 m in its own zone tests. Pre-existing ~0.7 m inconsistency, unchanged by this feature; worth unifying later.
