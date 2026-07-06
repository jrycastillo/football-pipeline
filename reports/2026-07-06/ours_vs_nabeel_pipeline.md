# Our Pipeline vs Nabeel's Pipeline — Consolidated Comparison

**Date:** 2026-07-06
**Basis:** hands-on tests of Nabeel v2 (2026-07-03) and v3 (2026-07-04) on our
hardware and our footage, against our pipeline's measured results on the same
clips. Detailed per-version reports: `reports/2026-07-03/nabeel_v2_evaluation.md`,
`reports/2026-07-04/nabeel_v3_evaluation.md`.

---

## Executive summary

On identical footage and identical hardware, our pipeline produces graded,
confidence-scored, per-player statistics in predictable time; Nabeel's
pipeline (both versions) produces **empty output** on match footage and
**cannot finish** long videos. His codebase still contributed two ideas we
adopted on our terms. There is no scenario in which switching or merging
pipelines makes sense; we cherry-pick components.

## Head-to-head on our test footage

| | **Ours** | **Nabeel v2** | **Nabeel v3** |
|---|---|---|---|
| 14-min GT clip, full | **35–50 min**, complete output | killed at 85% after **1h35m** | killed at 54% after **1h00m** |
| 5-min trim | ~20 min incl. annotated video | 32.4 min, **empty report** | 33.5 min, **empty report** |
| Stats vs Babak GT (count accuracy) | passes 50.7% · interceptions 56.2% · shots 100% · crosses 50% | all zeros | all zeros |
| Events with confidence | 469 per 14-min clip, all scored + verified/unverified labelled | none | none |
| Jersey numbers | 5/12 roster numbers matched (worst-case footage; 10 false) | 0 players identified | 0 players identified |
| Teams | 2 kits discovered (HSV or SigLIP), roster override supported | fixed color dictionary | SigLIP+UMAP, but fitted on **1–7 crops** (needs 250) — teams "Unknown" |
| Scaling behavior | linear in video length | super-linear (registry grows per frame) | same, plus extra per-frame model |

## Root causes on his side (measured, not speculation)

1. **Detector domain gap:** his `best.pt` (unchanged v2→v3) was trained on
   close-up phone clips. On wide-angle match footage it finds almost no
   players (1–7 crops per 5–14 min vs a 250 target) and no ball. All his
   stats hang off ball possession → no ball, no anything.
2. **Unbounded identity registry:** every detection is matched against every
   known player's embedding gallery, every frame, with no pruning — per-frame
   cost grows until throughput collapses (~0.2 fps by minute 10). A 90-minute
   match is out of reach on this design regardless of GPU.

## What we adopted from him (and its current state)

| Component | State in our pipeline |
|---|---|
| **Pitch keypoint homography** (v3's best work) | Ported as `vision/pitch_homography.py` + `--pitch_homography`; per-frame H feeds the stats engine. Fit rate on broadcast: 31% after our tuning — ceiling until the keypoint model is retrained on our footage (his `best.pt` problem repeating in miniature). |
| **Embedding gallery scoring** (his matching idea) | Used offline in our fragment-merge prototype (0.65·mean + 0.35·best over snapshots) — deliberately NOT per-frame online, avoiding his scaling failure. |
| **Multi-region ReID embedding** | Already in `vision/osnet_reid.py` before this evaluation. |
| Pose→LSTM activity model (dribble/tackle/foul) | Not adopted: trained on close-up single-player clips; unusable at our 30–80 px crops without retraining. |
| His full pipeline / stats / tracking | Rejected with measurements. |

## Where ours is honestly weak (so this report isn't a victory lap)

- Jersey identity is our weakest link: 5/12 on worst-case footage, and
  per-event identity confidence shows only ~19% of events carry a
  high-trust identity. The fragment-merge rework targets exactly this.
- Fouls: heuristic exists, 0/4 recall on training footage (no dead-ball
  signature); needs competitive footage or a retrained activity model.
- Distance/xG spatial quality is capped by homography fit rate (31%) and the
  speed-gate/fragmentation stack.

These weaknesses are all being worked with measured baselines — which is the
core difference: our pipeline's failures are quantified and addressable;
his pipeline's failures are architectural.
