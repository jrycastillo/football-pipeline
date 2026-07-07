# Nabeel Deliverables vs Contract Requirements — Compliance Check

**Date:** 2026-07-07
**Basis:** hands-on testing of Nabeel's v2 (2026-07-03) and v3 (2026-07-04)
archives on our hardware + our footage (see `reports/2026-07-03/`,
`reports/2026-07-04/`). Requirements from the project brief ("Detailed
Requirement.pdf").

**Scope of this check:** I can directly assess the technical behavior I
measured (detection, jersey reading, stats, consistency, throughput). I did
NOT have access to his repo commits, database, or written documentation, so
those are marked "not verified" rather than pass/fail.

**CORRECTION (2026-07-07, after reviewing the annotated videos):** an earlier
version of this report said his pipeline produces "0 players / empty report on
our footage." That was an over-generalization from the Babak clip and the
stalled 5-min HB run. Reviewing the videos shows his detection is
FOOTAGE-DEPENDENT: on the sharp HD Hamburg clip it works well (players boxed,
teams A/B, GK/referee classes, some jersey numbers, radar; report identifies
9 players with numbers+teams). On the lower-quality Babak clip it fails
entirely (zero detection boxes). The rows below are corrected accordingly.

Legend: ✅ met · ⚠️ partial · ❌ not met (measured) · ❓ not verified

---

## Stage 1 (the gating stage — 2.6: MUST be satisfied before Stage 2)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Accept webm/mp4/avi/URL inputs | ⚠️ | Accepts mp4/URL (his `/upload-url` + downloader). Opened our 1080p mp4 fine. webm/avi not tested. |
| 2.1 | Detect Players, Referees, Pitch, Goals, Ball | ⚠️ | Footage-dependent. Sharp HD (Hamburg): detects players, referee, GK well + pitch keypoints. Lower-quality (Babak): zero detections. Ball detection weak/absent on both. |
| 2.2 | Classify players into teams by jersey color | ⚠️ | Works on the HD clip (TeamA/TeamB assigned in video + report). Fails on the Babak clip (no detections to classify). |
| 2.3 | Model detects jersey NUMBER | ⚠️ | Reads SOME numbers on the HD clip (visible in video + 9 numbered players in report). None on the Babak clip. |
| **2.4** | **≥70% jersey detection accuracy** | ❓ | **NOT measured.** He reads some numbers on HD footage but the clip that finished was only 60s (9 players) — too short to score against a roster, and longer runs stall before completing. Accuracy at scale is untested because the pipeline can't finish a full match. |
| 2.5 | Consistent/better across different full-match sources | ❌ | Fails: strong on the HD clip, zero on the Babak clip — the opposite of consistent across sources. And no source produces a COMPLETE full-match result (all long runs stalled). |
| 2.6 | Stage 1 satisfied FIRST before Stage 2 | ⚠️/❌ | Stage 1 works on select footage but not consistently, and never over a full match — so not demonstrably satisfied as the gate the contract requires. |

**Corrected root cause:** two separate issues. (1) The detector degrades on
lower-quality/more-distant footage (fine on HD, fails on the Babak clip) —
a training-domain gap. (2) Regardless of footage, per-frame cost grows until
the pipeline stalls, so it cannot process a full match (48 min for 60s; 5-
and 14-min runs stalled). The runtime issue is the harder blocker: even where
detection works, no usable full-match result is produced.

## Stage 2 (stats)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 3.1–3.3 | Compute basic (goal/passes/tackle/challenges/dribbles) + advanced (xG/xA) | ⚠️ code exists / ❌ output | The report schema HAS all these fields and his logs HAVE stat functions, but on our footage every counter returned 0 (no possession → no stats). On his own sample video it produces small non-zero numbers. |
| **3.4** | **All stats ≥80% accuracy** | ❌ / ❓ | Cannot be met while Stage 1 yields nothing on our footage. No accuracy measurement is possible without detections. Not validated against any published full-match result. |
| 3.5 | Result contains teams/players/numbers + basic+advanced stats (schema) | ⚠️ | Schema is correct and complete (team_stats, per-player, pass/shot/foul logs). It's just empty on our footage. |
| 3.6 | Dump result to a database | ❓ | Not verified — his report writes JSON files; no DB integration observed in the archive, but not confirmed absent. |
| 3.7 | QA First Pass — consistent across different full match videos | ❌ | Fails consistency (see 2.5). Cannot complete a full match on broadcast footage. |
| 3.8 | QA Second Pass — ≥3 concurrent, queue, notify, status interface | ⚠️/❓ | v3 added a busy-guard (blocks concurrent), a `/progress` status endpoint, and a dashboard — but the busy-guard is the OPPOSITE of "process 3 simultaneously." Queue/notify not verified. |

## Non-technical requirements

| # | Requirement | Status |
|---|---|---|
| 4.1 | Technical documentation | ❓ Not found in archive (no README/setup/docs; ships his whole Windows venv). |
| 4.2 | Test scenarios and results | ❓ Not found. |
| 5.1 | Source code committed to provided repo | ❓ Not verified (I tested the zip archives, not a repo). |
| 6 | Full stats list (green priority, GK column, grey) | ⚠️ Schema present incl. GK column; unverified in output (empty). |

---

## Bottom line

**Stage 1 partially works but is not demonstrably met as the contract defines
it.** His detector, team classifier, and jersey OCR function on sharp HD
footage (Hamburg) but fail on lower-quality footage (Babak) — so §2.5
(consistent across different full-match sources) fails. The ≥70% jersey bar
(§2.4) is untested at scale, because the only run that finished was 60s (9
players); every longer run stalled before completing. Since no complete
full-match result exists, Stage 2's ≥80% accuracy (§3.4) also can't be
demonstrated.

**Important fairness caveats before any payment decision:**
1. We tested his archive builds on OUR footage. If his GCP milestone was
   validated on specific footage he selected, his numbers there may differ —
   but requirement 2.5 explicitly demands consistency across *different*
   full-match sources, which is exactly what fails.
2. Requirement 10 notes no published full-match validation result exists yet,
   so "the" acceptance footage is not pinned down — worth resolving before
   judging, since it's the difference between "works on his demo" and "works
   on real matches."
3. Several contractual items (DB dump, docs, repo commits, QA passes) we
   could not inspect and are marked not-verified, not failed.

**Recommendation:** before releasing the GCP3 milestone (50%), require a live
demo of Stage 1 on a **full match video that Babak provides** (not Nabeel's
own footage), measured against the 70% bar. That single test settles
compliance objectively. Our own pipeline, for reference, produces per-player
stats and (with the corrected recognition model) 12/12–15/21 roster-number
coverage on the same footage class — so the bar is achievable; his pipeline as
delivered does not clear it on broadcast footage.
