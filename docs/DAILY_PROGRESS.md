# Daily Progress Log — Football Analytics Pipeline

Single running log of daily work. Newest day at the bottom. Each entry:
what was done, what was measured, decisions, and what's next. Commit hashes
are in the dev repo (`football.git`, branch `production-v1.0`).

---

## 2026-07-01 — Handover baseline

- Took over the pipeline. Read the full agent guide (`docs/AGENT_INSTRUCTIONS.md`).
- Confirmed the working state: roster/user-input feature live (`--roster_file`),
  Layer 2 roster reconciliation shelved (`_ENABLE_ROSTER_RECONCILE=False`) because
  it dropped real players on unreliable-jersey footage.
- Commits: `2a29ee7` (disable Layer 2, restore working state), `7a8e4b3`, `1a86de4`.
- **Baseline accuracy** (14-min Babak GT clip): passes 50.7%, interceptions 56.2%,
  shots 100%, crosses 50%, fouls 0%, goals 0%. Jersey ~5/12 with 10 false numbers.

## 2026-07-02 — Sprint 1: the verification workflow

- **Per-event confidence scoring** (`bf8a9f9`): every event (pass, cross,
  interception, touch, dribble, tackle, shot, goal, save) emits a heuristic
  confidence in [0.05, 0.98]. Verified stat-neutral (identical counts).
- **Verified/Unverified labelling** (`5f5553b`): every event + player labelled;
  new `verification_summary.json` with confidence bands.
- **Event clipping** (`1a0c3c8`): `stats/event_clipper.py` — ±3s clips + manifest.
- **Foul detection** (`1dce7b4`, `e8af3b2`): heuristic (contact + stoppage + still
  ball). Honestly measured 0/4 on training footage — informal play restarts too
  fast to leave a dead-ball signature. Zero false positives.
- All validated on the worker: accuracy report identical to baseline.

## 2026-07-03 — Foul tuning + Nabeel v2 evaluation

- Foul gate fixes (`321ec81`, `5034986` diagnostics): raw-ball stillness,
  unmapped-track foulers, referee exclusion. Result stays 0 on training footage —
  documented as footage-limited, not a code gap.
- **Nabeel v2 hands-on test** (`b9639c3`, `reports/2026-07-03/`): on our 14-min
  clip his pipeline was killed at 85% after 1h35m; 5-min trim finished in 32 min
  with an EMPTY report. Root cause: his detector trained on close-up phone footage
  finds ~0 players/ball on wide-angle video. Verdict: do not switch; adopt only
  his homography + gallery ideas.

## 2026-07-04 — Homography port + multi-agent work

- **Pitch keypoint homography** ported from Nabeel v3 (`df2005d` estimator +
  test, `07ff57e` pipeline integration, `abaf3dd` tuning, `075ca84` cadence fix,
  `e59f96c` docs). Real px→meter projection replacing the flat scale. A/B on HB
  clip: fit rate 20%→31% after tuning; distance still capped by the speed-gate +
  fragmentation stack. Kept opt-in (`--pitch_homography`).
- **Nabeel v3 test** (`10feb2a`): same empty-report outcome as v2; v3 added a good
  homography stack + SigLIP teams but never retrained the detector, so no change
  on our footage. Slower (killed at 54% after 1h).
- **Multi-agent round 1** (Codex + Gemini, all reviewed/committed by me):
  per-event identity confidence (`b9e4981`), penalty-box constants unification
  (`21beb46`), event-level GT eval harness (`f889a6e`), derived stats (`8cb8b32`).
- `--dump_fragments` (`9cbd017`): per-track evidence export for the identity rework.
- `AGENTS.md` (`f07b7b1`): rules file for coding agents (dev repo only).

## 2026-07-05 — Back-testing gate + evidence upgrades

- **Back-testing gate** (`40936c5`): `tools/backtest_gate.py` — PASS/FAIL verdict
  against configurable baselines (band-based; over-detection also fails). Live-
  tested on the real accuracy report. Thresholds placeholder pending Babak.
- **H.264 clip encoding** (`fd0a300`): ffmpeg with per-clip mp4v fallback; clips
  now ~2 MB and browser-playable.
- **Fragment dump v2** (`43ff969`): timestamped jersey reads + embedding galleries
  per track — needed after finding 20% of fragments carry mid-track ID switches.

## 2026-07-06 — Comparison, showcase, clean-repo push

- **Consolidated pipeline comparison** (`27bdc24`, `reports/2026-07-06/`): ours
  vs Nabeel v2/v3, head-to-head with measurements.
- **HB 5-min showcase** (`4d53e27`): full-featured run with roster — 14 players
  (3 exact matches, 9 false — the magnet-number problem), 225 scored events
  including a false-positive goal correctly routed to admin review (event conf
  0.79 / identity conf 0.50). 7 H.264 clips (14.5 MB), annotated video.
- **Highlights uploader** (`3df062b`): `tools/upload_highlights.py` posts clips to
  Jhan's ScoutBridge endpoint with metadata (dry-run validated; not yet live).
- **Fragment splitter** (`d19e89e`) + **run-report generator** (`b52a261`) —
  agent round-3 work, reviewed/committed.
- **Pushed the clean shared repo** (`football-pipeline`, `6aed902..e3ea9c8`) —
  16 curated commits for Jhan & Jerome; reports/AGENTS/tasks excluded.
- Video comparison package delivered locally (`output/videos_for_review/`): ours
  vs Nabeel on the identical HB 60-s clip (ours 8 min / 79 events / 10 players;
  his 48.5 min / 11 noisy events / 0 players).

## 2026-07-07 — JNR root cause + PARSeq fix (in progress)

- **Found the JNR root cause** (flagged by Ronan): the pipeline defaults to
  `resnet34_clean.pt`, which scores **2.1% end-to-end** on labeled HB crops — it
  learned jersey appearance, not digit reading. This is the source of the magnet
  numbers, the 9 false numbers, and the low identity confidence.
- **PARSeq was already wired** (`--jnr_backend parseq`) but never selected.
  Evaluated all checkpoints on 3,000 HB crops through the production path:
  ResNet 2.1% · v6 46% (held-out) · v8 71% (match-specialized, trained on HB).
- **Video A/B result (HB 5-min, scored vs official roster)** — v6 is held out
  (did NOT train on this match), so this is an honest generalization number:

  | metric | ResNet (old default) | PARSeq v6 |
  |---|---|---|
  | exact team+number matches | 3 | **13** |
  | false numbers | 9 | **1** (and it's #18 Jatta, a real player missing from our roster) |
  | roster numbers found | 5/21 | **15/21** |

  A 4× jump in correct identities and near-elimination of the magnet/false-number
  problem — the root issue behind low identity confidence and the fragment-merge
  difficulty.
- **Made PARSeq v6 the config default** (`5010728`). `--jnr_backend resnet`
  restores the old model; v8 available for match-specialized use.
- **In progress**: Babak-clip validation with v6 (fully held out — no PARSeq
  version trained on it) vs the 5/12 ResNet baseline.
- Implication: this likely lifts every downstream identity metric and makes the
  fragment-merge rework easier (cleaner reads → less denoising). Highest-value
  change of the week; jumps ahead of the merge prototype in priority.

---

### Open items carried forward
- Babak decisions: confidence threshold, xG ownership, clip types, default view,
  go-live bar (all have supporting data ready).
- Need from Babak: a competitive clip + complete lineups (gates jersey accuracy,
  fouls, and a fair accuracy read).
- Jhan: live upload confirmation + `identity_confidence` field in his schema.
- Jerome: timestamped GT events per `docs/GT_EVENTS_FORMAT.md`.
- Unpushed: dev repo `production-v1.0` (backup push, user's token).
