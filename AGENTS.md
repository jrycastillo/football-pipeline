# Agent Instructions (Codex / any coding agent)

Read `docs/AGENT_INSTRUCTIONS.md` in full before doing anything — it is the
single source of truth (vision, architecture, history, constraints,
validation, communication). This file is only the hard-rules summary.

## Hard rules (non-negotiable)

1. **Commits:** each change is a SEPARATE commit, authored as
   `John Ronan Castillo <jr.yonzon@icloud.com>`. NEVER add Co-Authored-By,
   "Generated with", or any AI/tool attribution to commits, code, comments,
   or docs. Do not push — the maintainer pushes with an inline token.
2. **This dev repo only.** Never copy this file, CLAUDE.md, reports/, or
   backups into the clean shared repo (`football-pipeline`) — collaborators
   there must not see AI tooling or internal reports.
3. **Never commit `.env`** (DB password + ScoutBridge token). Never store
   tokens in git remotes.
4. **Do NOT re-enable** `_ENABLE_ROSTER_RECONCILE` in `stats/metrics.py`
   without the full rework described in `docs/AGENT_INSTRUCTIONS.md` §5.
5. **Validation before "done":** run on the GPU worker with a roster file
   containing `known_stats` and compare `accuracy_report.json` against the
   previous baseline. No accuracy regressions. Local machine has no GPU and
   its cv2 cannot load (libGL) — tests that need cv2 must run on the worker.
6. **Checker mode:** the AI's output is labelled unverified until the admin
   workflow verifies it. Do not change that default.

## Environment facts

- Main pipeline: `pipeline_consolidated.py`; stats: `stats/metrics.py`,
  `stats/event_logic.py`; identity: inline in `pipeline_consolidated.py`
  (the `vision/identity_manager.py` file is legacy/unused).
- GPU worker: `ronan@100.76.11.68`, repo at `~/work/football`, venv at
  `~/work/football/.venv`. Deploy = scp changed files. Do not
  `pkill -f pipeline_consolidated.py` (it matches your own ssh command);
  kill by exact PID.
- Config per footage: broadcast = FPS 25 / NEW_TRACK_THRESH 0.85;
  non-broadcast = FPS 20 / 0.45. Restore config.yaml after a run.
- Model weights (`models/*.pt`) are gitignored; ship via scp.

## Style

- Match surrounding code style; plain comments that state constraints, not
  narration. No emoji in code.
- Reports go in `reports/YYYY-MM-DD/`; docs in `docs/`.
