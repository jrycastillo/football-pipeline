# R14 Item A — Events Clipper Baseline (frozen reference)

**Date:** 2026-08-14 · **Clips:** `hb_5min.mp4` (HB, no GT) · `babak_test_full.mp4` (854s ≈ 14 min, Babak GT)
**Scope:** goals, shots, assists (client focus). save / goal_restart counted, not analysed.
**Runs:** `output/r14_clip_hb`, `output/r14_clip_babak` (stride 3, `--clip_events goal,assist,shot,save,goal_restart`).

## A1 — Events emitted vs clips written

| Clip | goal | shot | assist | save | goal_restart | clips written |
|------|------|------|--------|------|--------------|---------------|
| hb_5min | 0 | 7 | 0 | 0 | 0 | 7 (7 with player box) |
| babak_14min | 0 | 1 | 0 | 0 | 0 | 1 (1 with player box) |

Events emitted == clips written on both (no emit/write discrepancy).

## A2 — Clip integrity

Both clips: **0 failures** on all checks (file exists, non-zero, ffprobe opens + valid duration, decodes to final frame). All clips 6.0s (3.0s lead-in + 3.0s tail, `pad_s=3.0`).
`test_event_clipper_codec.py` covers only ffmpeg-command construction and codec fallback selection — **it covers none of these integrity checks** and would pass while a clip is missing/truncated.

## A3 — Temporal accuracy

Lead-in/tail is a **constant 3.0s / 3.0s** for every clip (the event sits centred, `pad_s=3.0`). The event is never near a clip edge. Matching to *true* event moments is not possible: `ground_truth/babak_14min_gt.json` holds **counts only, no timestamps**.

## A4 — Precision / recall (babak GT; hb has no GT)

GT (babak): goals=1 (scorer #10), shots=6 (on_target 4), assists not enumerated.

| Type | detected | GT | TP | FP | FN | recall | precision |
|------|----------|----|----|----|----|--------|-----------|
| goals | 0 | 1 | 0 | 0 | 1 | **0%** | n/a |
| shots | 1 | 6 | 1* | 0 | 5 | **~17%** | ~100%* |
| assists | 0 | (n/a) | 0 | 0 | — | **0%** | n/a |

No false positives — the clipper **under-detects** on this amateur clip. (*shot TP assumed: 1 of 6 real shots, unverifiable without GT timestamps.)

## A5 — Assist dependency (measured)

`assist` is emitted only inside the detected-`goal` branch (`stats/event_logic.py`). Babak: **0 goals detected → 0 assists possible → assist recall bounded at 0%.** The assist feature has **never been validated** (no clip has ever produced a detected goal here). 0 assist decisions consulted a team label, so the same-team check (downstream of the `team_map` defect, H1-H4) has not been exercised on these clips.

## Resolved contradictions

- **goals 0%** does NOT mean the clip has no goals — it has **1 real goal (scorer #10)** that the pipeline **missed** (recall 0/1). Not absence; a miss.
- **shots 100%** (docs/DAILY_PROGRESS) is **stale/contradicted** — this regenerated run gives shots recall **~17% (1/6)**.
- **A6:** `_detect_goal_restarts` docstring said ">=5 s"; live code uses **13 s** (`min_gap`). Docstring corrected to 13 s; behaviour unchanged.

## Bottom line

The clipper's *mechanics* are sound (0 integrity failures, correct windows). The problem is **detection recall on amateur footage** (goals 0/1, shots 1/6) — a detection packet, not a clipping one. hb_5min (broadcast HB) yields 7 shots but has no GT to score against.
