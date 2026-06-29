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
        self.teams = {}          # team_name -> {"color": str, "roster": set[int]}
        self.color_to_team = {}  # color -> team_name
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
            self._all_numbers |= roster

        self.known_facts = data.get("known_facts", {}) or {}

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
