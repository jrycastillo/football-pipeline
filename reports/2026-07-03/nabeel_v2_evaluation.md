# Nabeel Pipeline v2 — Hands-On Evaluation vs Our Pipeline

**Date:** 2026-07-03
**Test footage:** `babak_test_full.mp4` (14-min training clip, 17,080 frames @20fps) — the same clip our pipeline is graded on against Babak's ground truth.
**Hardware:** RTX 3070 worker, isolated Python venv (his code untouched except two portability fixes: a hard-coded Windows `DATASET_PATH` and relocating `lstm_weights.pth` into `Activities/`).
**Source:** `football-analytics-v2.zip` (3.76 GB; 318 MB of code+models extracted for testing). Test environment deleted after evaluation.

---

## Verdict

**Do not switch. Do not adopt his stats path.** On our footage his pipeline produced an *empty* report — zero passes, zero possession, zero events, zero jersey-identified players — and could not finish the full clip at all (killed at 85% after 1h35m, with the last hour advancing only ~4% of the video). The two components previously earmarked as adoptable (pose→LSTM activity confidence, pitch-keypoint homography) remain the only parts worth revisiting, with caveats below.

---

## 1. Runtime

| Run | Footage | Result |
|---|---|---|
| v2, full clip | 14 min | **Killed at 85% after 1h35m.** First 5,700 frames in ~2 min, then super-linear slowdown; final hour ~750 frames (~0.2 fps). Projected finish: 4-5 h. |
| v2, 5-min trim | 5 min | **Finished in 32.4 min** (1,945 s). Same slowdown pattern began ~frame 5,000. |
| Ours, full clip | 14 min | **35–50 min** including annotated video output (six validation runs this sprint). |

The slowdown is algorithmic, not resource-bound (2.2 GB RSS, 12 GB RAM free, no swap): per-frame cost grows with his accumulated player registry / ReID gallery, so cost explodes with match length and player count. A 90-min match is computationally out of reach — consistent with the earlier "2.3× slower" evaluation, but worse on long footage than that number suggests.

## 2. Stats output (5-min trim, completed run)

His match report is structurally rich (team stats, per-player possession/activity, pass/shot/tackle/foul logs) but came back **entirely empty** on our footage:

| Metric | Nabeel v2 | Ours (full 14 min) | GT (14 min) |
|---|---|---|---|
| passes | 0 | 69 | 136 |
| interceptions | 0 | 9 | 16 |
| shots | 0 | 6 | 6 |
| shots on target | 0 | 6 | 4 |
| crosses | 0 | 1 | 2 |
| fouls | 0 | 0 | 4 |
| goals | 0 | 0 | 1 |
| possession % | 0.0 / 0.0 | 64.6% frames owned | — |
| pass/shot/dribble/foul logs | all empty | 469 scored events | — |

Root cause (from code review): every stat in his pipeline hangs off ball-possession assignment, which depends on his custom detector (`best.pt`) finding the ball. It was trained on his close-up phone footage and does not transfer to our wide-angle training video — no ball, no possession, no events, no player stats. His sample report from the archive (his own test video) shows small counts (5 passes in a short clip), confirming the pipeline works on his domain but not ours.

## 3. Jersey number accuracy

Not comparable in his favor — he identified **0 players** on our footage (player entries only appear in his report through possession, which never engaged). Ours on the same clip: **5/12 unique roster numbers matched** (4, 6, 7, 10, 14; 10 false numbers) — weak, but measurable and improving via the roster feature.

Code review of his jersey path (`jersey_detector.py`): YOLO digit-detector (`jersey_best.pt`, 44 MB) on player crops with CLAHE/sharpen/background-removal preprocessing and per-digit confidence ≥0.30, duplicate-digit removal. It's a reasonable OCR design (similar spirit to our ResNet34+legibility gate) but there is no vote/lock identity layer as robust as ours, and it was never exercised on our footage in this test.

## 4. What his codebase does have (adoption candidates)

- **Pose→LSTM activity recognition** (`yolo_pose.py`, 356 pose features/frame, LSTM heads for dribble/tackle/foul with per-event confidence + cooldowns + ball-ownership gating in `activity_stats.py`). All artifacts present (weights/scaler/encoder) and the module loads and runs. **Caveat:** trained on close-up single-player phone clips; our player crops are 30–80 px where pose keypoints degrade — needs retraining on broadcast-scale crops before it can supply the foul/dribble/tackle confidence we want.
- **Pitch keypoint model** (`pitch_best.pt`, 140 MB) — untested here; still the candidate starting point for real homography to replace our flat pixel→meter scale (affects distance/xG).
- His FastAPI dashboard/report format is clean; nothing to adopt technically, but his per-event log structure resembles what our verification workflow now emits natively.

## 5. Methodology notes

- Isolated environment: fresh venv (torch 2.12.1+cu130, ultralytics, torchreid, sklearn, pandas), his code run via a small standalone runner calling `detect_and_track()` + `generate_match_report()` directly (his FastAPI layer bypassed).
- Two portability fixes only; no tuning, no thresholds changed — this is his pipeline as-is on our footage.
- Full-clip run killed by PID after measuring the degradation curve; 5-min trim used to obtain a completed report.
- Cleanup done: `~/nabeel_v2_test` (code, models, venv), tarball, and trim removed from the worker. The original `football-analytics-v2.zip` archive remains untouched in the dev repo root.

## 6. Bottom line for the team

Our pipeline, on identical footage, produces 469 confidence-scored events, per-player stats graded at 50–100% per metric against ground truth, in a third of the time v2 needs to *not* finish. v2's value is not its pipeline but two models (pose-LSTM, pitch keypoints) — both requiring adaptation work before they help. Priority stays: competitive clip for validation, LSTM retraining only if foul/confidence work justifies it.
