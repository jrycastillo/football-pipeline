#!/usr/bin/env python3
"""
Generate a formatted player stats report from processed video outputs.
Creates a dated report directory with the markdown report + raw JSON copies.

Usage:
    python3 generate_report.py
    python3 generate_report.py --date 2026-02-14
    python3 generate_report.py --video_ids 162b6abe208946b,14c0f4e8c4af40d,69a33466fc234db
"""

import json
import os
import shutil
import argparse
from datetime import datetime
from collections import defaultdict

DEFAULT_VIDEO_IDS = ["162b6abe208946b", "14c0f4e8c4af40d", "69a33466fc234db"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def load_video_data(vid):
    """Load player_stats.json and match_kits.json for a video."""
    stats_path = os.path.join(OUTPUT_DIR, vid, "player_stats.json")
    kits_path = os.path.join(OUTPUT_DIR, vid, "match_kits.json")

    if not os.path.exists(stats_path):
        return None, None

    with open(stats_path) as f:
        stats = json.load(f)
    kits = {}
    if os.path.exists(kits_path):
        with open(kits_path) as f:
            kits = json.load(f)
    return stats, kits


def get_xg(s):
    return (s.get("xg_foot_no_opponent", 0) + s.get("xg_foot_opponent_present", 0) +
            s.get("xg_header_no_opponent", 0) + s.get("xg_header_opponent_present", 0))


def get_shots(s):
    return s.get("shots_on_target_total", 0) + s.get("shots_wide_total", 0)


def get_file_size_str(vid):
    """Get approximate video size from known values."""
    sizes = {
        "162b6abe208946b": "314 MB",
        "14c0f4e8c4af40d": "1.3 GB",
        "69a33466fc234db": "3.4 GB",
    }
    return sizes.get(vid, "Unknown")


def get_processing_time(vid):
    """Try to extract processing time from log, otherwise return N/A."""
    log_path = os.path.join(OUTPUT_DIR, vid, "player_stats.json")
    if os.path.exists(log_path):
        mtime = os.path.getmtime(log_path)
        # We can't determine processing time from the file alone
        return "N/A"
    return "N/A"


def format_team_table(players):
    """Format a team's players as a markdown table."""
    lines = []
    lines.append("| # | Pos | Obs | Dist | Touch | ToB | G | S | xG | Drb | Pass | P% | Tkl | Chal | Int | Rec |")
    lines.append("|---|-----|----:|-----:|------:|----:|--:|--:|---:|----:|-----:|---:|----:|-----:|----:|----:|")

    for p in sorted(players, key=lambda x: x["jersey_number"] or 0):
        s = p["stats"]
        pos = "GK" if p["position"] == "GK" else "Pla"
        xg = get_xg(s)
        shots = get_shots(s)
        lines.append(
            f"| {p['jersey_number']} | {pos} | {p['observations']} | {s['total_distance']:.0f} "
            f"| {s['touch_frames']} | {s['time_on_ball_s']:.1f} "
            f"| {s['goals_total']} | {shots} | {xg:.2f} "
            f"| {s['dribbles_total']} | {s['passes_total']} | {s['accurate_passes_percent']:.0f} "
            f"| {s['tackles_total']} | {s['challenges_total']} "
            f"| {s['ball_interceptions_total']} | {s.get('ball_recoveries_opp_half', 0)} |"
        )
    return "\n".join(lines)


def generate_report(video_ids, date_str, pipeline_info=""):
    """Generate the full markdown report."""
    lines = []
    lines.append(f"# Full Player Stats Report — {date_str}")
    lines.append("")
    if pipeline_info:
        lines.append(f"**Pipeline:** {pipeline_info}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Status:** ✅ All {len(video_ids)} Videos Completed")
    lines.append("")
    lines.append("---")

    for i, vid in enumerate(video_ids, 1):
        stats, kits = load_video_data(vid)
        if stats is None:
            lines.append(f"\n## Video {i} — `{vid}`")
            lines.append("**Status:** ❌ No player_stats.json found")
            lines.append("")
            continue

        # Group players by team (skip Unknown_ prefixed)
        teams = defaultdict(list)
        for pid, p in stats.items():
            if pid.startswith("Unknown"):
                continue
            teams[p["team"]].append(p)

        total_players = sum(len(v) for v in teams.values())
        team_names = sorted(teams.keys())

        # Calculate match totals
        all_players = [p for pl in teams.values() for p in pl]
        total_goals = sum(p["stats"]["goals_total"] for p in all_players)
        total_shots = sum(get_shots(p["stats"]) for p in all_players)
        total_xg = sum(get_xg(p["stats"]) for p in all_players)
        total_passes = sum(p["stats"]["passes_total"] for p in all_players)
        total_tackles = sum(p["stats"]["tackles_total"] for p in all_players)
        total_dribbles = sum(p["stats"]["dribbles_total"] for p in all_players)
        total_saves = sum(p["stats"].get("shots_saved_total", 0) for p in all_players)

        # Calculate score per team
        team_goals = {}
        for team, players in teams.items():
            team_goals[team] = sum(p["stats"]["goals_total"] for p in players)

        # Format score
        if len(team_names) == 2:
            score_str = f"{team_names[0]} {team_goals.get(team_names[0], 0)}–{team_goals.get(team_names[1], 0)} {team_names[1]}"
        else:
            score_str = f"Total Goals: {total_goals}"

        file_size = get_file_size_str(vid)

        lines.append(f"\n## Video {i} — `{vid}` ({file_size})")
        lines.append(f"**Teams:** {', '.join(team_names)} ({total_players} players total)")
        lines.append(f"**Score:** {score_str}")
        lines.append(f"**Match Totals:** Shots: {total_shots} | xG: {total_xg:.2f} | Passes: {total_passes} | Tackles: {total_tackles} | Dribbles: {total_dribbles} | Saves: {total_saves}")

        for team in team_names:
            players = teams[team]
            lines.append(f"\n### Team {team} ({len(players)} players)\n")
            lines.append(format_team_table(players))

        lines.append("")
        lines.append("---")

    # Legend
    lines.append("")
    lines.append("## Legend")
    lines.append("| Abbr | Full Name |")
    lines.append("|------|-----------|")
    lines.append("| Pos | Position (GK/Pla) |")
    lines.append("| Obs | Frame observations |")
    lines.append("| Dist | Total distance (meters) |")
    lines.append("| Touch | Ball touches |")
    lines.append("| ToB | Time on Ball (s) |")
    lines.append("| G | Goals |")
    lines.append("| S | Shots (on target + wide) |")
    lines.append("| xG | Expected Goals |")
    lines.append("| Drb | Dribbles |")
    lines.append("| Pass | Passes |")
    lines.append("| P% | Pass accuracy % |")
    lines.append("| Tkl | Tackles |")
    lines.append("| Chal | Challenges |")
    lines.append("| Int | Interceptions |")
    lines.append("| Rec | Ball recoveries (opp half) |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate player stats report")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date label for the report (default: today)")
    parser.add_argument("--video_ids", default=",".join(DEFAULT_VIDEO_IDS),
                        help="Comma-separated video IDs")
    parser.add_argument("--pipeline", default="H100 Optimized | VID_STRIDE=5",
                        help="Pipeline description for the report header")
    args = parser.parse_args()

    video_ids = [v.strip() for v in args.video_ids.split(",")]
    date_str = args.date

    # Generate report
    report = generate_report(video_ids, date_str, args.pipeline)

    # Create dated report directory
    report_dir = os.path.join(REPORTS_DIR, date_str)
    os.makedirs(report_dir, exist_ok=True)

    # Write report
    report_path = os.path.join(report_dir, f"full_stats_report_{date_str}.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✅ Report saved: {report_path}")

    # Copy raw JSON files
    for vid in video_ids:
        src = os.path.join(OUTPUT_DIR, vid, "player_stats.json")
        dst = os.path.join(report_dir, f"player_stats_{vid}.json")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Copied: {dst}")
        else:
            print(f"⚠️  Not found: {src}")

        # Also copy match_kits.json
        kits_src = os.path.join(OUTPUT_DIR, vid, "match_kits.json")
        kits_dst = os.path.join(report_dir, f"match_kits_{vid}.json")
        if os.path.exists(kits_src):
            shutil.copy2(kits_src, kits_dst)

    # Also save to project root for easy access
    root_report = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"full_stats_report_{date_str.replace('-', '')[-4:]}.md")
    with open(root_report, "w") as f:
        f.write(report)
    print(f"✅ Also saved to: {root_report}")

    print(f"\n📁 Report directory: {report_dir}")
    print(f"📄 Files:")
    for f in sorted(os.listdir(report_dir)):
        size = os.path.getsize(os.path.join(report_dir, f))
        print(f"   {f} ({size // 1024}K)")


if __name__ == "__main__":
    main()
