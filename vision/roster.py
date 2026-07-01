"""
vision/roster.py — User-provided roster & team-color priors (ground-truth input).

Phase 1 of the user-guided-input feature (see docs/PLAN_user_guided_input.md).
Loads and validates a roster JSON the user fills in before processing. The
user's input acts as ground-truth priors that later phases use to constrain
team assignment and jersey-number recognition.

Backward-compatible: if no file is given (or it is missing/invalid),
RosterPrior.load() returns None and the pipeline runs fully automatically.

Schema:
{
  "teams": {
    "Red":   {"color": "red",   "roster": [1, 4, 7, 10, 11]},
    "White": {"color": "white", "roster": [1, 3, 9, 14, 21]}
  },
  "known_facts": {"final_score": {"Red": 2, "White": 1}}   # optional
}
"""

import json
import os

# Map free-text user colors onto the classifier's canonical football vocabulary
# (Red, Blue, Green, Yellow, White, Black). Mirrors color_classifier._FOOTBALL_MERGE
# plus common spelling variants the user might type.
_COLOR_SYNONYMS = {
    "maroon": "Red", "pink": "Red", "orange": "Red", "crimson": "Red",
    "navy": "Blue", "cyan": "Blue", "teal": "Blue", "sky": "Blue", "skyblue": "Blue",
    "lime": "Green",
    "gold": "Yellow",
    "silver": "White", "gray": "White", "grey": "White",
}


def canonical_color(c):
    """Normalize a user-typed color to the classifier's canonical label."""
    c = (c or "").strip().lower()
    if not c:
        return None
    if c in _COLOR_SYNONYMS:
        return _COLOR_SYNONYMS[c]
    return c.capitalize()  # red->Red, white->White, black->Black, ...


class RosterPrior:
    """Holds the user-provided team colors and per-team jersey rosters."""

    def __init__(self, data):
        self._data = data
        self.teams = {}                   # team_name -> {"color": str, "roster": set[int]}
        self.color_to_team = {}           # raw user color -> team_name
        self.canonical_color_to_team = {} # canonical color (Red/White...) -> team_name
        self._all_numbers = set()

        for name, info in (data.get("teams") or {}).items():
            color = (info.get("color") or "").strip().lower()
            roster = set()
            for n in (info.get("roster") or []):
                try:
                    roster.add(int(n))
                except (TypeError, ValueError):
                    continue
            self.teams[name] = {"color": color, "roster": roster}
            if color:
                self.color_to_team[color] = name
                ccol = canonical_color(color)
                if ccol:
                    self.canonical_color_to_team[ccol] = name
            self._all_numbers |= roster

        self.known_facts = data.get("known_facts", {}) or {}

        # Phase 4: optional user-provided ground-truth stats per team, e.g.
        #   "known_stats": {"passes": {"Blue": 62, "Green": 74}, "fouls": {...}}
        # Used to auto-generate an accuracy report after analysis.
        self.known_stats = data.get("known_stats", {}) or {}

    # Map ground-truth metric names -> pipeline stat field(s). A field of None
    # means it's derived (see compare_stats).
    _STAT_FIELD_MAP = {
        "passes": "passes_total",
        "passes_accurate": "passes_accurate",
        "interceptions": "ball_interceptions_total",
        "shots_on_target": "shots_on_target_total",
        "shots": None,   # on_target + wide
        "crosses": "crosses_total",
        "fouls": "fouls_total",
        "goals": "goals_total",
        "dribbles": "dribbles_total",
        "tackles": "tackles_total",
    }

    def gt_total(self, metric):
        """Sum of the ground-truth values across teams for a metric, or None."""
        vals = self.known_stats.get(metric)
        if not isinstance(vals, dict):
            return None
        return sum(v for v in vals.values() if isinstance(v, (int, float)))

    def compare_stats(self, player_stats):
        """Compare pipeline output to the user-provided known_stats.

        player_stats: {player_key: {"stats": {...}, ...}}.
        Returns a list of dicts: {metric, pipeline, ground_truth, accuracy_pct}.
        Empty if no known_stats were provided.
        """
        if not self.known_stats:
            return []

        def field_total(field):
            return sum(p.get("stats", {}).get(field, 0) or 0 for p in player_stats.values())

        rows = []
        for metric, gt in self.known_stats.items():
            gtt = self.gt_total(metric)
            if gtt is None:
                continue
            field = self._STAT_FIELD_MAP.get(metric, metric + "_total")
            if metric == "shots":
                pv = field_total("shots_on_target_total") + field_total("shots_wide_total")
            else:
                pv = field_total(field)
            acc = round(100.0 * pv / gtt, 1) if gtt else (0.0 if pv else 100.0)
            rows.append({"metric": metric, "pipeline": round(pv, 2),
                         "ground_truth": gtt, "accuracy_pct": acc})
        return rows

    # --- helpers used by later phases (team assignment, JNR constraint) ---

    def team_colors(self):
        """{team_name: color} as typed by the user."""
        return {name: t["color"] for name, t in self.teams.items()}

    def canonical_team_colors(self):
        """{team_name: canonical_color} mapped to the classifier vocabulary (Phase 2)."""
        return {name: canonical_color(t["color"]) for name, t in self.teams.items()}

    def all_numbers(self):
        """Set of every valid jersey number across both teams (Phase 3)."""
        return set(self._all_numbers)

    def numbers_for_team(self, team_name):
        return set(self.teams.get(team_name, {}).get("roster", set()))

    def is_valid_number(self, num, team_name=None):
        """True if num is on the roster (of team_name, or any team if None)."""
        try:
            num = int(num)
        except (TypeError, ValueError):
            return False
        if team_name is not None and team_name in self.teams:
            return num in self.teams[team_name]["roster"]
        return num in self._all_numbers

    # Digit pairs that are commonly confused when reading blurred/angled jersey
    # numbers. Snapping only fires on these, so a genuinely different off-roster
    # number (e.g. 50) is admitted rather than force-corrected to 10.
    _CONFUSABLE = {
        frozenset("17"), frozenset("38"), frozenset("39"), frozenset("89"),
        frozenset("08"), frozenset("06"), frozenset("68"), frozenset("56"),
        frozenset("59"), frozenset("27"), frozenset("49"), frozenset("58"),
    }

    @classmethod
    def _plausible_misread(cls, num, cand):
        """True if cand is a plausible OCR misread of num: same digit count,
        exactly one differing digit, and that digit pair is visually confusable
        (e.g. 21<->27 via 1/7, 38<->88 via 3/8)."""
        a, b = str(num), str(cand)
        if len(a) != len(b):
            return False
        diffs = [(x, y) for x, y in zip(a, b) if x != y]
        if len(diffs) != 1:
            return False
        return frozenset(diffs[0]) in cls._CONFUSABLE

    def _nearest_roster(self, num, valid):
        """Nearest roster number that is a plausible misread of num, or None."""
        cands = [c for c in valid if self._plausible_misread(num, c)]
        if not cands:
            return None
        return min(cands, key=lambda c: abs(c - num))

    def snap(self, num, conf=None, team_name=None):
        """Soft roster constraint. Returns (mapped_num, status):
          - 'on_roster'  : read is a valid roster number (full trust)
          - 'snapped'    : read corrected to a near roster number (OCR misread)
          - 'off_roster' : no close roster match; kept as-is (admitted, unverified)

        A read valid for ANY team is accepted as on_roster even if the track's
        (possibly wrong/unsettled) color points at a different team. The
        team-specific roster only *prefers* a snap target. This avoids the
        chicken-and-egg failure where an early color misclassification rejects a
        perfectly valid number (e.g. Blue's #44 admitted as off-roster because
        the track was momentarily coloured Green)."""
        try:
            num = int(num)
        except (TypeError, ValueError):
            return num, "off_roster"
        if not self._all_numbers:
            return num, "off_roster"

        # Valid for some team -> accept (the number itself is real, regardless
        # of the current color assignment).
        if num in self._all_numbers:
            return num, "on_roster"

        # Not a real number anywhere -> try to snap. Prefer the assigned team's
        # roster, then fall back to all numbers.
        team_valid = self.numbers_for_team(team_name) if (team_name and team_name in self.teams) else set()
        cand = self._nearest_roster(num, team_valid) if team_valid else None
        if cand is None:
            cand = self._nearest_roster(num, self._all_numbers)
        if cand is not None:
            return cand, "snapped"
        return num, "off_roster"

    def summary(self):
        parts = []
        for name, t in self.teams.items():
            parts.append(f"{name} ({t['color']}): {sorted(t['roster'])}")
        return " | ".join(parts)

    @classmethod
    def load(cls, path):
        """Load a roster JSON. Returns a RosterPrior, or None if no/invalid file
        (so callers can treat None as 'run fully automatic')."""
        if not path:
            return None
        if not os.path.exists(path):
            print(f"⚠️ [Roster] file not found: {path} — running without roster priors")
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ [Roster] failed to parse {path}: {e} — running without roster priors")
            return None
        if not data.get("teams"):
            print(f"⚠️ [Roster] no 'teams' key in {path} — running without roster priors")
            return None
        prior = cls(data)
        print(f"✅ [Roster] Loaded user priors: {prior.summary()}")
        if prior.known_facts:
            print(f"✅ [Roster] Known facts: {prior.known_facts}")
        return prior
