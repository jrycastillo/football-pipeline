# Nabeel Pipeline — Detailed Technical Feedback (for Babak)

**From:** Ronan
**Date:** 2026-07-08
**Basis:** hands-on testing of Nabeel's v2 (received ~Jun 28) and v3 (received
Jul 4) archive builds, run unmodified on our GPU worker against multiple
footage sources, checked against the signed requirements document
("Detailed Requirement" brief). An appendix at the end is written so you can
forward it to Nabeel directly.

---

## 1. Executive summary

Nabeel's pipeline is **real, well-designed work that functions on clean HD
footage — but it does not yet meet the contract's gating requirements**, for
two measured reasons:

1. **It cannot complete a full match.** Processing slows down as the match
   runs and stalls. Best case measured: 48 minutes to process a 60-second
   clip; every longer attempt (5 min, 14 min) stalled partway and never
   finished. A 90-minute match is out of reach by orders of magnitude.
2. **Results are not consistent across footage sources.** On a sharp HD
   broadcast clip, detection works well. On our lower-quality test match,
   the detector produced literally zero detections — empty output.

Because no full match has ever completed, the contract's central bar —
**70% jersey detection accuracy (§2.4)** — cannot be measured at all yet, on
any footage. That, plus the consistency requirement (§2.5) failing, means
**Stage 1 is not demonstrably satisfied**, and §2.6 makes Stage 1 a hard gate
for everything after it.

**Recommendation:** before approving the GCP3 milestone, require a live demo
of Stage 1 on a full-match video **that you provide** (not his own footage),
scored against the 70% bar. That single test settles compliance objectively.

---

## 2. How we tested (methodology)

- Ran his v2 and v3 archives **exactly as shipped** — his code, his models,
  his thresholds. Only two changes, both needed just to make his own code
  load on Linux (a hard-coded Windows path; a model file location).
- Fresh Python environment per version, all his dependencies installed,
  CUDA-enabled GPU (RTX 3070). His startup logs confirm every component
  loaded: ReID on CUDA, activity recognizer, team classifier.
- Called the same processing functions his own API endpoint calls.
- Footage used: (a) a sharp 1080p broadcast clip (Hamburg vs Bayern),
  (b) our 14-minute test match (the one we grade our own pipeline on),
  in 60-second, 5-minute, and full-length variants.
- Every claim below is backed by his own run logs and his own annotated
  output videos, which we can share.

## 3. What works — credit where due

- **On sharp HD footage, Stage 1 detection genuinely functions.** The
  Hamburg clip output shows players boxed and tracked, team A/B assignment,
  goalkeeper and referee classes, some jersey numbers read, and his
  radar/minimap overlay working. His report identified 9 players with
  numbers and teams on that 60-second clip.
- **The output schema is complete and correct** — team stats, per-player
  sections, pass/shot/foul logs, and the goalkeeper column from the stats
  sheet are all present and properly structured.
- **The pitch keypoint model (v3) is good work** — clean homography with
  sensible handling of partial pitch views.
- **The pose-based activity model** (dribbles/tackles/fouls with per-event
  confidence) is the right concept for the hardest stats.
- He is **responsive to product requirements**: v3 added a progress API,
  an upload guard, and the radar he marked "client requirement."

## 4. What fails — the two blockers (measured)

### 4.1 Runtime: cannot complete a full match
| Attempt | Footage | Result |
|---|---|---|
| 60-second clip | HD broadcast | completed — in **48.5 minutes** |
| 5-minute clip | HD broadcast | stalled at ~25% after 1 hour, never finished |
| 5-minute clip | our test match | completed (33 min) but output empty (see 4.2) |
| 14-minute clip | our test match | v2 stalled at 85% after 1h35m; v3 at 54% after 1h |

The slowdown is structural: per-frame player matching grows with the number
of accumulated player identities and has no pruning, so cost rises until
throughput collapses (~0.2 frames/sec). More players on screen = earlier
stall — broadcast footage stalls *sooner*. No hardware upgrade fixes a
super-linear loop; a 90-minute match at 25 fps is ~135,000 frames.

### 4.2 Consistency: detection is footage-dependent
- **HD broadcast clip:** detection works (see §3).
- **Our test match (more distant/lower quality):** zero detection boxes in
  the entire annotated video; his own calibration log explains it — the team
  classifier needs 250 player crops and collected **1–7** across the whole
  video, then reported "Teams will be 'Unknown'". Report came back all zeros.
- His own bundled sample video (close-range) works fine.

So: works on his footage and on clean HD; fails on harder real-world
sources. §2.5 requires *"consistently produce the same or better results
using different full-match sources"* — this is currently the opposite.

### 4.3 A note on stat quality (early signal, small sample)
On the one clip that completed (60s HD), the stats that did come out were
sparse and partly implausible — e.g. **4 fouls detected in one minute** of
a calm opening period, 1 pass total. Too small a sample to score against
the 80% bar (§3.4), but not an encouraging early signal.

## 5. Requirement-by-requirement scorecard

| Req | Requirement (abridged) | Verdict | Evidence |
|---|---|---|---|
| §1 | Accept webm/mp4/avi/URL | Partial | mp4 + URL work; webm/avi untested |
| §2.1 | Detect players/refs/pitch/goals/ball | **Partial** | Works on HD; zero on our test match; ball weak everywhere |
| §2.2 | Team classification by jersey color | **Partial** | Works on HD; "Unknown" on our test match |
| §2.3 | Jersey number detection | **Partial** | Some numbers on HD; none elsewhere |
| §2.4 | **≥70% jersey accuracy** | **Unmeasurable** | Only a 60s run ever completed — too short to score; longer runs stall |
| §2.5 | **Consistent across full-match sources** | **Fail** | Strong on HD, zero on our match; no full match ever completed |
| §2.6 | Stage 1 gate before Stage 2 | **Not satisfied** | Follows from 2.4/2.5 |
| §3.1–3.3 | Basic + advanced stat functions | Partial | Functions and schema exist; output sparse/noisy where produced, empty elsewhere |
| §3.4 | ≥80% stat accuracy | Unmeasurable | No complete full-match output exists to score |
| §3.5 | Result schema (teams/players/numbers/stats) | **Met (structure)** | Schema complete incl. GK column |
| §3.6 | Result dumped to database | Not verified | Report writes JSON; no DB integration observed in archive |
| §3.7 | QA pass: consistent across full matches | **Fail** | Same evidence as 2.5 |
| §3.8 | 3 videos simultaneously + queue + status | **Fail (direction)** | v3 added a busy-guard that BLOCKS concurrent processing — opposite of the requirement; status endpoint exists |
| §4 | Technical documentation, test scenarios | Not delivered in archive | No README/setup/docs; archive ships his whole environment folder |
| §5 | Source code in provided repo | Not verified | We received zip archives |

## 6. Payment/milestone recommendation

The Q&A attached to the contract (item 3) ties the first 50% (USD 750) to a
**PROCEED** from us at GCP3. On the evidence:

- Stage 1 is not demonstrably met (2.4 unmeasurable, 2.5 failing, 2.6 gate).
- His own contract answer (item 10) says no published full-match validation
  footage was agreed — so there is currently no neutral ground truth on
  which he can claim the bar is met, either.

**Suggested condition for PROCEED:** a live, timed demo of Stage 1 on a full
match video you supply, on his hardware, producing: (a) a completed run,
(b) jersey accuracy scored against the real team sheets, target ≥70%
(§2.4), (c) repeated on a second source for §2.5. This is exactly what the
contract already entitles you to, and it converts the dispute-prone
question "does it work?" into a number.

Worth knowing: the 70% bar is achievable on this footage class — our own
pipeline currently reads 12/12 and 15/21 roster numbers on the same two
test videos — so holding the line on §2.4 is reasonable, not punitive.

## 7. What would actually fix his pipeline (if he proceeds)

1. **The runtime stall** — bound the identity-matching cost (prune/limit the
   registry, or match against candidates instead of everyone). Without this,
   nothing else matters: no full match, no measurable accuracy.
2. **Detector robustness** — retrain his detector on wide-angle/lower-quality
   match footage (we can supply annotated frames). This addresses §2.5.
3. Then re-test §2.4/§3.4 on neutral footage.

---

## Appendix — feedback written for Nabeel (forwardable as-is)

Hey Nabeel — proper feedback after testing v2 and v3 hands-on. Good news
and a couple of real issues, all straight.

**What genuinely works well:** on sharp HD broadcast footage, your pipeline
holds up nicely. On a Hamburg–Bayern clip it detected the players, split
them into the two teams, tagged the goalkeepers and referee, read some of
the jersey numbers, and the radar/minimap looked good. Your output schema
is also complete and well-structured — team stats, per-player breakdowns,
the pass/shot/foul logs, the goalkeeper column. The design is solid.

**Two issues we need to flag:**

1. **It doesn't finish a full match.** Even on footage where detection
   works, processing took ~48 minutes for a 60-second clip, and on longer
   clips it slows further until it stalls partway. It looks like per-frame
   matching cost grows as players accumulate. One consequence: we couldn't
   measure jersey accuracy against the 70% target (§2.4), because the only
   run that completed was 60 seconds — too short to score. Getting a full
   match to complete is what unlocks that measurement.
2. **Detection drops off on lower-quality footage.** On a more distant,
   lower-resolution match video, the detector found almost nothing — versus
   the HD clip where it worked well. Results need to hold across different
   match sources (§2.5), because real uploads won't all be crisp broadcast
   feeds.

**Where this goes:** the detection foundation is clearly there on good
footage. The two things to solve are the runtime (so a full match
completes) and detector robustness on harder sources. If it helps, we can
send annotated frames from a range of match footage for retraining, and
we're happy to do a call to walk through the tests, share full logs, or set
up the environment so you can reproduce both behaviors yourself.

Short version: the pipeline works on clean footage — the gaps are finishing
a full match and holding up on harder video. Both look fixable.
