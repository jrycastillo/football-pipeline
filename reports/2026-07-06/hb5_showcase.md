# Hamburg–Bayern 5-min Showcase — Jersey Numbers, Stats, Annotated Video

**Date:** 2026-07-06
**Run:** `output/run_hb5_showcase` on the worker — `hb_5min.mp4` (minutes 1–6 of
the HSV–Bayern broadcast), full current feature set: roster priors
(`roster_hb.json`), `--pitch_homography` (tuned), `--clip_events goal,shot,save`
(H.264), `--dump_fragments`, ReID on. Broadcast config, ~25 min processing.
Auto-generated machine summary: `output/run_hb5_showcase/run_report.md`.

## Jersey numbers vs official rosters

Scored against the corrected rosters (Bayern black 14 players, HSV white 16;
9 numbers shared between teams — team+number must both match to count):

| Detected | Team label | Obs | Verdict |
|---|---|---|---|
| **7** | Black | 2467 | **MATCH** (Bayern #7) |
| 62 | Black | 2467 | FALSE — magnet number, on no roster |
| 21 | Black | 711 | cross-team (#21 is Hamburg's) |
| 28 | Black | 679 | cross-team (#28 is Hamburg's) |
| **6** | Black | 499 | **MATCH** (Bayern #6) |
| **19** | Black | 499 | **MATCH** (Bayern #19) |
| 93 / 31 / 55 | Black | 268/247/113 | FALSE |
| 32 / 35 / 56 / 50 | White | 141/108/85/58 | FALSE |
| 23 | — | 65 | FALSE |

**Summary: 14 players output — 3 team+number matches, 2 right-number/wrong-team,
9 false numbers; 5/21 unique roster numbers found.** Two caveats cut both ways:
(a) only a subset of the 30 rostered players appear readably in a 5-min window,
so 21 is not a fair denominator; (b) the roster itself is incomplete — the video
shows HSV **#18 Jatta**, who is not on our roster list. The dominant failure is
unchanged: a few "magnet" numbers (62, 33, 21, 7) absorb reads across many
tracks — the fragment-merge/denoising rework targets exactly this, and this
run's `fragments_dump.json` is its test data.

## Stats detected (5 minutes, no GT counts for this segment)

225 events, all confidence-scored and unverified-labelled:
passes 62 (avg identity-conf 0.77) · touches 119 · dribbles 17 · tackles 13 ·
shots 6 · crosses 2 · interceptions 5 · goals 1.

- **The goal event is a false positive** — the scoreboard stays 0-0 through
  this window. Its EVENT confidence is 0.79 (would pass a 0.75 bar), but its
  IDENTITY confidence is 0.50 — so a queue routed on min(event, identity)
  catches it. This is the concrete argument for gating on both scores, not
  event confidence alone.
- Confidence bands: 131 high (≥0.75) / 90 mid / 4 low → an 0.75 admin
  threshold would queue ~42% of events on this footage.
- Homography: 96 real fits (31% of 312 calibrations, ~one per 3 s of video).

## Deliverables on the worker (`output/run_hb5_showcase/`)

| File | What |
|---|---|
| `output_video_h264.mp4` (**72 MB**) | Annotated video: player boxes, track IDs, team color, jersey number where locked (raw `output_video.mp4` is 310 MB) |
| `run_report.md` | Auto-generated single-page run summary |
| `player_stats.json` / `raw_tracks.json` | Per-player stats / 225 scored events |
| `clips/` + `clips_manifest.json` | 7 H.264 clips (6 shots + the false goal), **14.5 MB total** — first production use of the H.264 path |
| `fragments_dump.json` | 289-fragment evidence dump (v2) for the identity rework |

Annotation note: boxes appear only for **tracked** players; broadcast cutaways
and close-ups are mostly untracked at broadcast spawn threshold (0.85), so
replay/bench shots are clean frames by design.

## vs Nabeel on this exact footage

His v2/v3 pipelines produce an empty report and identify zero players on this
video (see `reports/2026-07-06/ours_vs_nabeel_pipeline.md`). This run —
14 identified players, 225 scored events, clips, annotated video — is the
side-by-side answer.
