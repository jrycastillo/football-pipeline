# Plan — User-Guided Ground-Truth Input

**Status:** Plan for review. Not yet implemented. Backup of the current pipeline
taken first (branch `backup/pre-userinput-*`, tag `backup-pre-userinput-*`,
archive in `~/pipeline_backups/`).

---

## 1. The idea

Before the AI processes a video, the **user provides known facts** as input:

- **Team colors** — e.g. Team A = Red, Team B = White
- **Roster (jersey numbers per team)** — e.g. Red: 1,4,7,10…  White: 1,3,9,11…
- *(optional)* **Known match facts** — final score / goals, for anchoring

The pipeline then analyzes the video **guided by** that input, instead of having
to discover everything blindly. The user's input acts as **ground-truth priors**
that constrain and correct the AI's output.

This is a **human-in-the-loop, prior-guided** design. It directly attacks the
problems we keep hitting (jersey misreads, team mis-clustering, the "magnet"
over-assignment, even numberless situations) by *telling* the system the answers
it currently has to guess.

---

## 2. Why this is high value

It fixes our biggest weaknesses at the source:

| Current problem | How user input fixes it |
|-----------------|-------------------------|
| Jersey misreads (open-set: model can output any number) | Constrain to the **known roster** — a read of "31" snaps to the nearest valid roster number, or is rejected if no team has it |
| "Magnet" over-assignment (many tracks → one jersey) | Roster gives a fixed set of slots; can't collapse 5 players onto one number that doesn't exist for that team |
| Team mis-clustering (e.g. lumping players as "Goalkeeper") | Team colors are **given**, not discovered — no K-means guessing |
| Wrong team colors on odd footage | Fixed up front |
| No accuracy baseline | The same input format doubles as a ground-truth file for evaluation |

---

## 3. Where each input plugs into the pipeline

Three clean anchor points — none require rebuilding the core:

**(a) Team colors → team assignment**
- File: `vision/team_clustering.py` / `vision/color_classifier.py` / kit discovery
- Today: HSV K-means *discovers* the two team colors.
- Change: if user supplies colors, **seed/override** the discovery with them.
  K-means becomes "assign each player to the nearest *given* team color."

**(b) Roster → jersey recognition (JNR) + identity**
- File: `vision/resnet_recognition.py` + IdentityManager (in `pipeline_consolidated.py`)
- Today: open-set — the model can lock any number it reads.
- Change: **closed-set constraint.** A recognized number must exist in that
  team's roster. Reads are mapped to the nearest valid roster number; numbers
  not in any roster are rejected. This also bounds the magnet bug — one roster
  slot per real player.

**(c) Known facts → validation / anchoring**
- File: `stats/metrics.py` / post-processing
- Today: nothing to check against.
- Change: if the user gives known goals/score, flag mismatches; and the input
  file feeds the evaluation harness directly as ground truth.

---

## 4. User input format (proposed)

A single JSON (or simple form/CSV) the user fills before processing:

```json
{
  "teams": {
    "Red":   { "color": "red",   "roster": [1, 4, 7, 10, 11, 23] },
    "White": { "color": "white", "roster": [1, 3, 6, 9, 14, 21] }
  },
  "known_facts": {            // optional
    "final_score": {"Red": 2, "White": 1}
  }
}
```

Passed via a new CLI flag, e.g. `--roster_file roster.json`. If omitted, the
pipeline behaves exactly as today (fully automatic) — so this is **additive and
backward-compatible.**

---

## 5. Implementation phases (incremental, each testable)

**Phase 1 — Input plumbing — DONE (commit a6d7755)**
- `vision/roster.py` (RosterPrior loader + validation) and `--roster_file` flag.
- Backward-compatible: no file -> None -> fully automatic.

**Phase 2 — Team colors as priors — DONE (commit b72cdbd)**
- User colors override KitCoordinator discovery (`forced_player_colors`).
- `canonical_color()` maps free-text colors to the classifier vocabulary.

**Phase 3 — Roster-constrained jersey recognition — DONE (commit 6bb780c)**
- `RosterPrior.snap()`: misread JNR reads snap to the nearest valid roster
  number (visually-confusable digit pairs only); off-roster reads admitted.
- Wired at the JNR registration point in `pipeline_consolidated.py`.

**Phase 4 — Validation & anchoring — PENDING**
- Compare output to `known_facts` (e.g. final score vs detected goals); surface
  mismatches. Wire the input format into the evaluation harness as ground truth.
- Lower priority; depends on the evaluation harness.

Phases 1-3 unit-tested and deployed to the worker. Next: end-to-end run with a
roster file on numbered match footage to validate live behavior.

---

## 6. Locked decisions (confirmed)

1. **Scope of user input:** **Roster + team colors only.** The user provides
   team colors and jersey rosters as priors; the AI produces all stats. No
   user-entered stats/events.
2. **Roster strictness:** **Soft constraint.** Map reads to the nearest valid
   roster number; allow an off-roster number through only at low confidence.
   Robust to incomplete/imperfect rosters.
3. **Input method:** **JSON file via `--roster_file` CLI flag.** Backward-
   compatible — omitting it runs the pipeline fully automatically as today.

---

## 7. Relationship to the team's other ideas

This is the foundation the other suggestions build on:
- **Jerome's spatial encoding** — user input anchors *who* players are; spatial
  encoding anchors *where* they are. Complementary.
- **Jhan's human-in-the-loop** — same philosophy; this is HITL at the *input*
  stage, his is at the *review* stage.
- **Nabeel's confidence-scored events** — roster constraint raises confidence on
  identity, which makes the flag-uncertain-events workflow cleaner.
