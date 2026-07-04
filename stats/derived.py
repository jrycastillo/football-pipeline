"""
Derived statistics calculation module.

Spatial stats (distance, possession, xG) require tracking and are explicitly out of scope — they cannot be derived from counts.
"""

import json
import os
import sys
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-player derived statistics from event logs and player stats.")
    parser.add_argument("--events", type=str, required=True, help="Path to input event list JSON (raw_tracks.json).")
    parser.add_argument("--player_stats", type=str, required=True, help="Path to input player stats JSON (player_stats.json).")
    parser.add_argument("--output", type=str, default="derived_stats.json", help="Path to write the output derived stats JSON. Default 'derived_stats.json'.")
    parser.add_argument("--verified_only", action="store_true", help="If set, only compute stats using events whose status is 'verified'.")
    parser.add_argument("--match_minutes", type=float, default=90.0, help="Match duration in minutes for per-90 normalizations. Default 90.0.")
    return parser.parse_args()

def load_json(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        sys.exit(1)

def get_primary_actor(event):
    """
    Extract the primary actor ID from the event.
    """
    t = event.get("type", "")
    if t in ("pass", "cross"):
        return event.get("from")
    elif t in ("tackle", "foul", "interception"):
        return event.get("by")
    else:
        return event.get("player")

def compute_derived_stats(events, player_stats, match_minutes=90.0, verified_only=False):
    """
    Computes per-player derived statistics.
    """
    # Initialize counters for all players in player_stats
    # Each player will have counters:
    # {pid: {total_passes: int, completed_passes: int, shots: int, goals: int, total_dribbles: int, successful_dribbles: int, tackles: int}}
    counters = defaultdict(lambda: {
        "total_passes": 0,
        "completed_passes": 0,
        "shots": 0,
        "goals": 0,
        "total_dribbles": 0,
        "successful_dribbles": 0,
        "tackles": 0
    })

    # Pre-populate counters with player keys from player_stats so they are always evaluated
    for pid in player_stats:
        _ = counters[str(pid)]

    for event in events:
        # Filter by status if verified_only is requested
        if verified_only and event.get("status") != "verified":
            continue

        raw_type = event.get("type", "")
        if not raw_type:
            continue

        t = raw_type.strip().lower()
        actor = get_primary_actor(event)
        if actor is None:
            continue

        pid_str = str(actor)

        if t in ("pass", "cross"):
            counters[pid_str]["total_passes"] += 1
            if event.get("complete") is True:
                counters[pid_str]["completed_passes"] += 1
        elif t == "shot":
            counters[pid_str]["shots"] += 1
        elif t == "goal":
            counters[pid_str]["goals"] += 1
        elif t == "dribble":
            counters[pid_str]["total_dribbles"] += 1
            if event.get("successful") is True:
                counters[pid_str]["successful_dribbles"] += 1
        elif t == "tackle":
            counters[pid_str]["tackles"] += 1

    # Compute final metrics
    derived_output = {}
    
    # Process both pre-existing players and any unknown players found in events
    all_player_ids = sorted(list(set(player_stats.keys()) | set(counters.keys())))

    for pid in all_player_ids:
        pid_str = str(pid)
        c = counters[pid_str]

        # Calculate percentages
        pass_acc = None
        if c["total_passes"] > 0:
            pass_acc = round((c["completed_passes"] / c["total_passes"]) * 100.0, 2)

        shot_conv = None
        if c["shots"] > 0:
            shot_conv = round((c["goals"] / c["shots"]) * 100.0, 2)

        dribble_succ = None
        if c["total_dribbles"] > 0:
            dribble_succ = round((c["successful_dribbles"] / c["total_dribbles"]) * 100.0, 2)

        # Calculate per-90 normalizations
        passes_90 = 0.0
        tackles_90 = 0.0
        if match_minutes > 0:
            passes_90 = round((c["total_passes"] / match_minutes) * 90.0, 2)
            tackles_90 = round((c["tackles"] / match_minutes) * 90.0, 2)

        # Get base player info
        base_info = player_stats.get(pid_str, {})
        player_entry = base_info.copy()

        # Update metadata if not present
        if "player_name" not in player_entry:
            player_entry["player_name"] = f"Unknown Player {pid_str}"
        if "jersey_number" not in player_entry:
            if pid_str.isdigit():
                player_entry["jersey_number"] = int(pid_str)
            else:
                player_entry["jersey_number"] = None
        if "team" not in player_entry:
            player_entry["team"] = "Unknown"

        # Inject derived stats
        player_entry["derived_stats"] = {
            "pass_accuracy_pct": pass_acc,
            "shot_conversion_pct": shot_conv,
            "dribble_success_rate_pct": dribble_succ,
            "passes_per_90": passes_90,
            "tackles_per_90": tackles_90
        }

        derived_output[pid_str] = player_entry

    return derived_output

def main():
    args = parse_args()
    events = load_json(args.events)
    player_stats = load_json(args.player_stats)

    derived = compute_derived_stats(
        events=events,
        player_stats=player_stats,
        match_minutes=args.match_minutes,
        verified_only=args.verified_only
    )

    try:
        with open(args.output, "w") as f:
            json.dump(derived, f, indent=2)
        print(f"Derived stats successfully saved to {args.output}")
    except Exception as e:
        print(f"Error saving derived stats to {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
