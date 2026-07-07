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

Legend: ✅ met · ⚠️ partial · ❌ not met (measured) · ❓ not verified

---

## Stage 1 (the gating stage — 2.6: MUST be satisfied before Stage 2)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Accept webm/mp4/avi/URL inputs | ⚠️ | Accepts mp4/URL (his `/upload-url` + downloader). Opened our 1080p mp4 fine. webm/avi not tested. |
| 2.1 | Detect Players, Referees, Pitch, Goals, Ball | ❌ | On our broadcast footage his detector found ~1–7 player crops over 5–14 min (his own log: target 250). Effectively no player/ball detection. Pitch keypoints (v3) work. |
| 2.2 | Classify players into teams by jersey color | ❌ | v3 team classifier could not fit — his log: *"collected 1 player crops … Teams will be 'Unknown'."* No team assignment produced. |
| 2.3 | Model detects jersey NUMBER | ❌ | 0 jersey numbers identified on our footage (no player crops to read). |
| **2.4** | **≥70% jersey detection accuracy** | ❌ | **Measured 0% on our footage.** This is the hard gate for the whole project. |
| 2.5 | Consistent/better across different full-match sources | ❌ | Fails the core generalization test: works on his own close-up sample, produces empty reports on our broadcast footage, and cannot finish a full match (killed at 54–85%). |
| 2.6 | Stage 1 satisfied FIRST before Stage 2 | ❌ | Stage 1 not met on our footage, so the gate to Stage 2 is not cleared. |

**Root cause (measured, not opinion):** his detector `best.pt` was trained on
close-up single-player phone footage; it does not generalize to wide-angle
broadcast/match video. Every downstream stage depends on it.

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

**The gating requirement — Stage 1, ≥70% jersey detection accuracy (2.4),
consistent across full-match sources (2.5), satisfied before Stage 2 (2.6) —
is not met on the footage we tested (measured 0%).** Because Stage 1 gates
everything, Stage 2's ≥80% stat accuracy (3.4) cannot be demonstrated either.

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
