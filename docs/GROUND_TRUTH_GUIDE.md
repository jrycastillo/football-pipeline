# Ground-Truth Reference Guide

## Why we're doing this

Our pipeline outputs per-player match statistics (`player_stats.json`). To know
how *accurate* it is, we need a human-verified "truth" to compare against.

There are **two separate things** you'll produce — they are not the same activity:

1. **Jersey-number reference (annotation):** identify each player on the numbered
   team by shirt number. This is labeling — matching a player to their number.
2. **Manual match stat-keeping (tallying):** watch the clip like a match analyst
   and count what each player does — passes, tackles, shots, etc. Stats are not
   "annotated"; they are *observed and tallied* in real time (with rewinds).

You do **not** need to read the pipeline code for this. You need football
knowledge + the definitions below. Consistency matters more than anything: use
the same definition every time.

---

## The video to annotate

Use the **14-minute test clip** (shared separately). Important constraints for
this specific video:

- It is a **training session**, not a competitive match — play is looser and
  less structured. Annotate it as best you can.
- **One team wears numberless training bibs.** There are no jersey numbers to
  read for them, so **only annotate the team with numbered jerseys.** For the
  bib team, just note "numberless bibs — not annotatable."
- Don't try to do all 14 minutes at once. Start with the **first 5 minutes**,
  get it fully right, then continue. Manual annotation runs ~3-5× real time.

This clip gives us a *partial* evaluation — engine accuracy on the players we can
identify, plus how many we may be missing. A future competitive clip (both teams
numbered) will give a fuller picture.

---

## Step 1 — Jersey-number reference (the annotation part)

For the **numbered team only**, list every visible player's jersey number. This
alone is high value — our jersey recognition is a known weak point.

| Team | Jersey # | Notes (position, GK, etc.) |
|------|----------|----------------------------|
| Blue | 1        | Goalkeeper |
| Blue | 10       | |

Record the team colors as you see them. For the bib team, just note the color
and that the players have no readable numbers.

---

## Step 2 — Manual match stat-keeping (the tallying part)

For the numbered team's players, count what each player does as you watch the
clip — like a match analyst. Use one row per player, one column per stat. Tally
in real time and rewind freely.

These are exactly the stats our pipeline produces, so your tallies compare
directly to its output:

| Team | # | Goals | Shots (on target) | Passes | Accurate passes | Crosses | Tackles | Interceptions | Dribbles | Fouls |
|------|---|-------|-------------------|--------|-----------------|---------|---------|---------------|----------|-------|

**Stat definitions — use these exactly** (they match how the pipeline counts):

- **Goal** — ball fully crosses the goal line into the net. Credit the scorer.
- **Shot (on target)** — a deliberate attempt toward goal that would go in or
  forces a save. Don't count blocked-by-defender or clearly wide attempts.
- **Pass** — player deliberately plays the ball to a teammate. Count it for the
  passer regardless of success.
- **Accurate pass** — a pass that reaches a teammate (same team keeps the ball).
- **Cross** — a pass played from a wide area into the penalty box.
- **Tackle** — a defender dispossesses an opponent who was in control of the ball.
  Only count when the opponent clearly had possession first.
- **Interception** — a player cuts out a pass intended for an opponent (ball was
  in flight between two opposing players).
- **Dribble** — a player takes on and beats an opponent while carrying the ball.
- **Foul** — an infringement the referee penalizes (or clearly should). Credit
  the player who committed it.

Skip distance and xG — those can't be eyeballed; the pipeline estimates them and
we evaluate those a different way.

---

## Step 3 — Deliver in this format

Save as a spreadsheet (CSV or Google Sheet) with the columns above, plus a
header noting: **video name, clip start/end time, your name, date.**

One row per player. Team totals (sum) at the bottom are helpful too.

Example CSV:
```
team,jersey,goals,shots,passes,accurate_passes,crosses,tackles,interceptions,dribbles,fouls
Blue,7,1,3,28,22,2,2,1,4,1
Blue,10,0,2,35,30,0,0,3,6,0
Blue,9,2,5,19,14,1,1,0,2,2
```

---

## Tips

- **Rewind often.** Accuracy > speed. A contested 5-minute clip can take 20-30 min.
- **When unsure, be consistent.** If you're not sure something is a tackle vs an
  interception, pick one rule and apply it the same way every time.
- **Flag ambiguous moments** in a notes column rather than guessing silently.
- **One clip, fully done** beats three clips half-done.

---

## What happens next

We load your spreadsheet and the pipeline's `player_stats.json` for the same clip
and compute a per-metric accuracy table (e.g. "passes: 82% of ground truth,
tackles: 140%"). That tells us exactly which stats are reliable and which need
work — which is the whole point.
