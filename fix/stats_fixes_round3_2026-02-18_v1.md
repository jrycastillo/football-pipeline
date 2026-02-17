# Stats Fixes Round 3 — 2026-02-18 v1

## Context

Based on the H100 comparison (Feb 14 vs Feb 17), switching to `VID_STRIDE=3` dramatically
improved pass detection (585 total, +88%) and pass accuracy (Red #18 hit 89%), but revealed
five critical regressions. Total xG exploded from 3.85→9.77 (+154%), Video 2 collapsed
entirely, and interception/tackle counts inflated significantly.

**Files changed:**
- `stats/event_logic.py` — Section 4 xG removal, recovery half fix, tackle debounce, interception window
- `stats/metrics.py` — Pass GK set to analyzer, team imbalance warning

---

## Fixes

### P0-1: Section 4 xG Double-Counting (event_logic.py)

**Problem:** Total xG exploded from 3.85→9.77 (+154%). Video 3 alone has 9.36 xG for a
2-2 match (realistic: 2-4 total). Root cause: xG is accumulated in TWO independent places:

- **Section 5** (line 432): Shot detection with velocity check → `calculate_xg()` → adds
  to `xg_foot_*` counters. This is correct — xG triggered by actual shot events.
- **Section 4** (line 676): Touch-in-box → `calculate_xg()` → adds to **the same**
  `xg_foot_*` counters. This gives xG for EVERY ownership segment ending in the penalty
  box, regardless of whether the player shoots. Defensive clearances, passes through the
  box, dribbles in the box — all generate xG.

With `VID_STRIDE=3`, there are ~1.67x more ownership segments than VID_STRIDE=5. Each
segment ending in the box adds xG. The double-counting was partially masked at stride 5
but is now obvious.

**Evidence:**
- Red #1 (GK): 0 shots, 1.50 xG — ALL from section 4 (box touches from goal kicks/catches)
- Red #18: 3 shots, 2.42 xG → 0.81 xG/shot average (real-world: ~0.10/shot)
- Section 4 is contributing the majority of xG for most players

**Fix:** Remove xG accumulation from section 4 entirely. xG should ONLY come from section 5
(actual shot events with velocity verification). Section 4 remains useful for tracking
penalty box touches but should NOT add to xG counters.

**Impact:** Total xG should drop from 9.77 → ~1.5-3.0 (realistic for 16 shots across 3 videos).
GK xG drops to 0.00.

### P0-2: Ball Recovery Half Classification — Pixel vs Meter Bug (event_logic.py)

**Problem:** In the interception logic (line 411), `end_pos[0] > 52.5` compares the ball's
**pixel** X coordinate to 52.5 **meters** (half the 105m pitch). Since video resolution is
typically 1920 pixels wide, `end_pos[0]` is almost always >52.5, making virtually every
interception classify as "recovery in opponent half."

**Evidence:** In the Feb 17 report, every player has `Int == Rec` (interceptions equals
opponent-half recoveries). This is statistically impossible in real football.

**Fix:** Project `end_pos` to meters using `self.camera.project_point()` before comparing
to 52.5m. This correctly classifies recoveries by pitch half.

### P1-1: Section 3 Tackle Debounce (event_logic.py)

**Problem:** Tackles increased from 13→48 (+270%). Section 3 (defensive tackles from
ownership transitions) has no per-player debounce. With VID_STRIDE=3 producing more
frame transitions, rapid back-and-forth ownership in contested areas generates multiple
tackle events per second.

**Fix:** Add per-player debounce to section 3: max 1 tackle per 5-second window per player.
Mirrors the existing interception debounce pattern.

### P1-2: Increase Interception Debounce Window (event_logic.py)

**Problem:** Despite the Round 2 debounce (3s window), Blue #4 still has 78 interceptions,
Blue #35 has 67. The root cause is team clustering errors (Blue team has only 6 players vs
Red's 15 — many Blue players are likely misclassified as Red, causing same-team passes to
register as cross-team → incomplete → interception).

While the upstream fix is team clustering, the stats layer can mitigate by increasing the
debounce window from 3 seconds to 10 seconds. This matches the typical duration of a
football possession sequence and caps interceptions at a more realistic rate.

**Fix:** Increase interception debounce from `EFF_FPS * 3` to `EFF_FPS * 10` (10 seconds).

### P1-3: Team Size Imbalance Warning (metrics.py)

**Problem:** Video 3 has 6 Blue players vs 15 Red players. Real football has 11v11.
This 2.5:1 ratio indicates severe team clustering errors that cascade into pass accuracy
(Blue at 9-16%) and interception inflation.

**Fix:** Add a log warning in `_cluster_teams()` when team sizes are imbalanced (>1.8:1
ratio). This helps operators identify clustering failures before they corrupt stats.

### P2-1: Video 2 Complete Collapse (Investigation — No Code Fix)

**Problem:** Video 2 collapsed entirely in Feb 17: 12 passes (vs 74), 0 shots (vs 14),
0 goals (vs 5), team color changed from Green→Yellow. This is NOT a stats formula issue.

**Root cause hypothesis:** Color classifier instability with VID_STRIDE=3. The change in
temporal resolution may affect the color sampling window or initialization behavior,
causing the classifier to assign different colors to the same kits.

**Action:** Requires separate investigation of color classifier and ball tracking behavior
on Video 2 specifically. Not addressable through stats formula changes.

---

## Expected Impact

| Metric         | Feb 17 | Expected After |
|---------------|--------|----------------|
| Total xG       | 9.77   | 1.5-3.0        |
| GK xG (Red #1) | 1.50   | 0.00           |
| Tackles        | 48     | 20-30          |
| Interceptions  | ~300+  | ~60-100        |
| Int == Rec     | always | ~40-60% match  |

---

## Verification

```bash
python3 -c "import py_compile; py_compile.compile('stats/event_logic.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('stats/metrics.py', doraise=True)"
```

---

## Files Changed

- `stats/event_logic.py` — P0-1, P0-2, P1-1, P1-2
- `stats/metrics.py` — P1-3
