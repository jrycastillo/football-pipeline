#!/usr/bin/env python3
import json
import os
import sys
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a markdown summary report for a pipeline run.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the pipeline run output directory.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the generated markdown report. Default: <run_dir>/run_report.md")
    parser.add_argument("--title", type=str, default=None, help="Custom title for the report.")
    return parser.parse_args()

def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def parse_accuracy_report(run_dir):
    """
    Parse accuracy_report.json.
    Returns list of dicts or None if missing/invalid.
    """
    path = os.path.join(run_dir, "accuracy_report.json")
    data = load_json(path)
    if not isinstance(data, list):
        return None
    return data

def parse_backtest_verdict(run_dir):
    """
    Parse backtest_verdict.md.
    Returns the first line of the file, or None if missing.
    """
    path = os.path.join(run_dir, "backtest_verdict.md")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    return line.strip()
    except Exception:
        pass
    return None

def parse_verification_summary(run_dir):
    """
    Parse verification_summary.json.
    Returns the loaded dict or None if missing/invalid.
    """
    path = os.path.join(run_dir, "verification_summary.json")
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    return data

def parse_player_stats(run_dir):
    """
    Parse player_stats.json.
    Returns a sorted list of players or None if missing/invalid.
    """
    path = os.path.join(run_dir, "player_stats.json")
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    
    players = []
    for pid, pinfo in data.items():
        if not isinstance(pinfo, dict):
            continue
        stats = pinfo.get("stats", {})
        players.append({
            "id": pid,
            "player_name": pinfo.get("player_name", f"Player {pid}"),
            "jersey_number": pinfo.get("jersey_number"),
            "team": pinfo.get("team", "Unknown"),
            "observations": pinfo.get("observations", 0),
            "verification_status": pinfo.get("verification_status", "unverified"),
            "passes": stats.get("passes_total", 0),
            "shots_on_target": stats.get("shots_on_target_total", 0),
            "tackles": stats.get("tackles_total", 0),
            "distance": stats.get("total_distance", 0.0)
        })
    
    # Sort by observations descending
    players.sort(key=lambda p: p["observations"], reverse=True)
    return players

def parse_clips_manifest(run_dir):
    """
    Parse clips_manifest.json.
    Returns summary dict or None if missing/invalid.
    """
    path = os.path.join(run_dir, "clips_manifest.json")
    data = load_json(path)
    if not isinstance(data, list):
        return None
    
    total_clips = len(data)
    count_by_type = {}
    total_size_bytes = 0
    admin_queue_size = 0
    
    for clip in data:
        if not isinstance(clip, dict):
            continue
        ctype = clip.get("type", "unknown")
        count_by_type[ctype] = count_by_type.get(ctype, 0) + 1
        total_size_bytes += clip.get("size_bytes", 0)
        
        conf = clip.get("confidence")
        if conf is None or conf < 0.75:
            admin_queue_size += 1
            
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    return {
        "total_clips": total_clips,
        "count_by_type": count_by_type,
        "total_size_mb": total_size_mb,
        "admin_queue_size": admin_queue_size
    }

def parse_diagnostics_log(run_dir):
    """
    Search pipeline.log (then launch.log) for diagnostic one-liners.
    Returns list of verbatim matching lines, or None if no log file exists.
    """
    log_files = ["pipeline.log", "launch.log"]
    log_path = None
    for f in log_files:
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            log_path = p
            break
            
    if not log_path:
        return None
        
    targets = [
        "[PitchHomography] fit stats",
        "[FoulDebug]",
        "[PassDebug]",
        "Fragment dump:",
        "[Option A]",
        "[StatsEngine] Ball track"
    ]
    
    matches = []
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                for target in targets:
                    if target in line_str:
                        matches.append(line_str)
                        break # Avoid duplicate addition if multiple targets match
    except Exception:
        pass
        
    return matches

def build_markdown_report(run_dir, title=None):
    title = title or "Pipeline Run Report"
    run_dir_name = os.path.basename(os.path.abspath(run_dir))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md = []
    
    # 1. Header
    md.append(f"# {title}")
    md.append(f"**Run Directory:** {run_dir_name}")
    md.append(f"**Generated At:** {timestamp}")
    md.append("")
    
    # 2. Accuracy vs ground truth
    md.append("## Accuracy vs Ground Truth")
    acc_data = parse_accuracy_report(run_dir)
    if acc_data:
        md.append("| Metric | Pipeline | Ground Truth | Accuracy |")
        md.append("| --- | --- | --- | --- |")
        for row in acc_data:
            metric = row.get("metric", "unknown")
            pipeline = row.get("pipeline", 0)
            gt = row.get("ground_truth", 0)
            acc_pct = row.get("accuracy_pct", 0.0)
            md.append(f"| {metric} | {pipeline} | {gt} | {acc_pct:.1f}% |")
    else:
        md.append("Not available")
    md.append("")
    
    # 3. Back-test verdict
    md.append("## Back-test Verdict")
    verdict = parse_backtest_verdict(run_dir)
    if verdict:
        md.append(f"{verdict} ([backtest_verdict.md](backtest_verdict.md))")
    else:
        md.append("Not available")
    md.append("")
    
    # 4. Event summary
    md.append("## Event Summary")
    ev_summary = parse_verification_summary(run_dir)
    if ev_summary:
        verified = ev_summary.get("verified_events", 0)
        unverified = ev_summary.get("unverified_events", 0)
        total = verified + unverified
        bands = ev_summary.get("confidence_bands", {})
        
        md.append(f"- **Total Events:** {total}")
        md.append(f"- **Verified Events:** {verified}")
        md.append(f"- **Unverified Events:** {unverified}")
        md.append("- **Confidence Bands:**")
        md.append(f"  - High (>= 0.75): {bands.get('high_0.75+', 0)}")
        md.append(f"  - Mid (0.50 - 0.75): {bands.get('mid_0.50-0.75', 0)}")
        md.append(f"  - Low (< 0.50): {bands.get('low_<0.50', 0)}")
        md.append("")
        
        by_type = ev_summary.get("by_event_type", {})
        if by_type:
            md.append("| Event Type | Count | Avg Confidence | Avg Identity Confidence |")
            md.append("| --- | --- | --- | --- |")
            for etype in sorted(by_type.keys()):
                info = by_type[etype]
                count = info.get("count", 0)
                
                avg_conf = info.get("avg_confidence")
                avg_conf_str = f"{avg_conf:.4f}" if isinstance(avg_conf, (int, float)) else "N/A"
                
                avg_id = info.get("avg_identity_confidence")
                avg_id_str = f"{avg_id:.4f}" if isinstance(avg_id, (int, float)) else "N/A"
                
                md.append(f"| {etype} | {count} | {avg_conf_str} | {avg_id_str} |")
        else:
            md.append("No event details available")
    else:
        md.append("Not available")
    md.append("")
    
    # 5. Players
    md.append("## Top Players by Observations")
    players = parse_player_stats(run_dir)
    if players:
        md.append("| Jersey | Team | Name | Observations | Verification Status | Passes | Shots on Target | Tackles | Distance (m) |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for p in players[:10]:
            jersey = p["jersey_number"]
            j_str = str(jersey) if jersey is not None else "-"
            dist_str = f"{p['distance']:.2f}"
            md.append(f"| {j_str} | {p['team']} | {p['player_name']} | {p['observations']} | {p['verification_status']} | {p['passes']} | {p['shots_on_target']} | {p['tackles']} | {dist_str} |")
    else:
        md.append("Not available")
    md.append("")
    
    # 6. Clips
    md.append("## Clips Summary")
    clips = parse_clips_manifest(run_dir)
    if clips:
        md.append(f"- **Total Clips:** {clips['total_clips']}")
        md.append(f"- **Total Size:** {clips['total_size_mb']:.2f} MB")
        md.append(f"- **Admin Queue Size (Conf < 0.75):** {clips['admin_queue_size']}")
        md.append("- **Clips by Type:**")
        by_type = clips.get("count_by_type", {})
        if by_type:
            for ctype in sorted(by_type.keys()):
                md.append(f"  - {ctype}: {by_type[ctype]}")
        else:
            md.append("  - None")
    else:
        md.append("Not available")
    md.append("")
    
    # 7. Diagnostics
    md.append("## Diagnostics")
    logs = parse_diagnostics_log(run_dir)
    if logs is None:
        md.append("Not available")
    elif len(logs) == 0:
        md.append("No matching diagnostic logs found")
    else:
        md.append("```")
        for line in logs:
            md.append(line)
        md.append("```")
    md.append("")
    
    return "\n".join(md)

def main():
    args = parse_args()
    report = build_markdown_report(args.run_dir, title=args.title)
    
    output_path = args.output
    if not output_path:
        output_path = os.path.join(args.run_dir, "run_report.md")
        
    try:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Run report successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving run report: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
