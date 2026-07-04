# Nabeel Pipeline v3 — Test Results & Progress Assessment

**Date:** 2026-07-04
**Source:** `football-analytics-v3.zip` (7.9 GB; 370 MB code+models extracted). New/changed code dated Jun 30 – Jul 3.
**Test:** same protocol as the v2 evaluation (2026-07-03) — isolated venv on the RTX 3070 worker, his code as-is (only the Windows `DATASET_PATH` patched), run on `babak_test_full.mp4` (14-min GT clip) and a 5-min trim.

---

## Verdict

**Output on our footage is unchanged from v2: an empty report.** Zero passes, zero events, zero identified players, and this time teams are literally labelled "Unknown" because his own calibration step found **1 player crop in 5 minutes of video** (it targets 250). v3 contains genuine, well-built new features — real homography above all — but they all sit downstream of the same detector (`best.pt`, byte-identical to v2) that cannot see players or the ball in wide-angle match footage. The identity/scaling core we flagged is untouched. **Recommendation stands: don't combine pipelines — port his homography module into ours.**

---

## 1. What v3 actually adds (code diff vs v2)

| Area | Change | Assessment |
|---|---|---|
| **Homography / radar** *(new)* | `p_best.pt` (32-landmark pitch keypoint model, trained Jul 2) + `view_transformer.py`: RANSAC `cv2.findHomography`, camera-px → pitch-cm, partial-pitch handling, last-valid-matrix fallback; `radar_renderer.py` minimap; `train_pitch_keypoints.py` training script included | **The standout.** Correctly engineered, exactly the piece our pipeline lacks (`vision/camera.py` uses a flat pixel→meter scale that distorts distance/xG). Directly portable. |
| **Team classification** *(rewritten)* | Fixed color dictionary → SigLIP (`google/siglip-base-patch16-224`) + UMAP + KMeans, unsupervised, with a 25-frame calibration pre-pass; per-frame batched prediction | Sound approach (we already run SigLIP clustering optionally), but it needs player crops to fit — see §3. |
| **Product plumbing** *(new)* | Live progress API (`/progress`, calibrating/processing/finalizing states), busy-guard against concurrent uploads, reworked `main.py` | Fine, not relevant to us. |
| **Identity / tracking** | `player_registry.py`, `matching_engine.py`, `tracker_manager.py` **unchanged since v2 (Jun 2)** | The scaling bug and locked-jersey-vote issues we identified are still there. |
| **Detection models** | `best.pt`, `jersey_best.pt` **byte-identical to v2 (May 12–18)** | The root-cause component was not retrained. |
| **Stats detectors** | pass/shot/possession/stats managers unchanged (May 23–28) | — |

## 2. Runtime on our 14-min test video

| Run | v2 (Jul 3) | v3 (Jul 4) |
|---|---|---|
| Full 14-min clip | killed at **85% after 1h35m** | killed at **54% after 1h00m** (stalled, 0 frames/30s at sampling) |
| 5-min trim | completed, 32.4 min | completed, **33.5 min** |
| Ours, full clip (reference) | 35–50 min total | — |

v3 is slower per frame — the pitch keypoint model runs **every frame** on top of detection, jersey OCR, ReID and pose — and the super-linear registry slowdown is unchanged, so a full match remains computationally out of reach.

## 3. Output on our footage (5-min trim, completed run)

Identical to v2: team stats all zero, all 12 event logs empty, zero players in possession/activity sections, 0/16 roster jersey slots. New in v3, his own logs now pinpoint the cause:

```
🎯 Calibrating teams — sampling every 240 frames (target 250 crops) ...
🎯 Calibration collected 1 player crops
⚠️  TeamClassifier.fit: need at least 2 valid crops, got 1. Not fitted.
⚠️  Team classifier could not be fitted. Teams will be 'Unknown'.
```

(Full 14-min run: 7 crops collected out of a 250 target.) His detector finds essentially no usable player crops on wide-angle footage — it was trained on close-up phone videos. No detections → no crops, no teams, no possession, no ball → no passes/shots/events → empty report. Every v3 feature inherits this ceiling.

## 4. Progress assessment (fair reading)

- **Real progress:** the homography stack is well done (correct algorithm, sensible confidence gating, partial-view handling, training script shipped). The SigLIP team classifier is the right direction and mirrors what we already use. He is clearly responsive to product requirements (progress bar, radar were "client requirements").
- **No progress on the blocking issues:** the detector domain gap (the reason both v2 and v3 output nothing on match footage) and the identity registry scaling bug (the reason neither version can finish a long video) are both untouched. These are the two things that decide whether his pipeline works at all on real matches.
- **Net:** v3 is feature progress on a foundation that still doesn't hold on our footage. Nothing in v3 changes the "do not switch" verdict.

## 5. What we should take, and what to ask him

**Adopt (worth real effort):** `p_best.pt` + `view_transformer.py` + `pitch_config.py` as the replacement for our flat homography in `vision/camera.py` — improves distance, possession zones and xG inputs across the board. We'd run it every N frames (as we already do for pitch calibration), not every frame.

**Evaluate later:** his pitch-keypoint *training script* — lets Jerome extend the landmark dataset with our footage if `p_best.pt` doesn't transfer.

**Questions to add to the existing list (see 2026-07-03 report §questions):**
1. What footage was `p_best.pt` trained on, how many annotated frames, and what's its keypoint accuracy on wide-angle/non-broadcast camera angles? Can we get the training data?
2. `best.pt` is unchanged since May while everything downstream depends on it — is a retrain on wide-angle match footage planned? What would he need from us?
3. Has v3 ever completed a full match (90 min) end-to-end? On what hardware and in how long?
4. The calibration step samples crops via `best.pt`; on our footage it collected 1–7 crops vs a 250 target — does he have a fallback when detection density is low?

## 6. Housekeeping

Test environment (code, models, venv, trim clip, tarball) deleted from the worker and locally after the test. The original `football-analytics-v3.zip` remains untouched in the dev repo root alongside the v2 archive.
