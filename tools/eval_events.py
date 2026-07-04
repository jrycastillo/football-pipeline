#!/usr/bin/env python3
import json
import os
import sys
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate event-level precision and recall against ground truth.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth JSON file.")
    parser.add_argument("--det", type=str, required=True, help="Path to detected events JSON (raw_tracks.json).")
    parser.add_argument("--player_stats", type=str, default=None,
                        help="Path to player stats JSON (player_stats.json). If omitted, searches in the directory of --det.")
    parser.add_argument("--fps", type=float, default=25.0, help="Video frame rate. Default 25.0.")
    parser.add_argument("--vid_stride", type=int, default=3, help="Video frame stride. Default 3.")
    parser.add_argument("--tolerance", type=float, default=3.0, help="Default matching time tolerance in seconds. Default 3.0.")
    parser.add_argument("--type_tolerances", type=str, default=None,
                        help="Comma-separated custom time tolerances per type (e.g. pass:2,shot:3,goal:5).")
    parser.add_argument("--strictness", type=str, default="type-only",
                        choices=["type-only", "type+team", "type+player"],
                        help="Strictness mode for matching. Default 'type-only'.")
    parser.add_argument("--min_conf", type=float, default=0.5,
                        help="Minimum confidence threshold for detected events. Default 0.5.")
    parser.add_argument("--sweep", action="store_true", help="Enable sweep mode across confidence thresholds (0.3 to 0.9).")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation metrics report as JSON.")
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

def parse_type_tolerances(type_tolerances_str, default_tolerance):
    tolerances = defaultdict(lambda: default_tolerance)
    if not type_tolerances_str:
        return tolerances
    try:
        parts = type_tolerances_str.split(",")
        for part in parts:
            if ":" in part:
                t_type, t_val = part.split(":")
                tolerances[t_type.strip().lower()] = float(t_val.strip())
    except Exception as e:
        print(f"Warning: Failed to parse type_tolerances '{type_tolerances_str}': {e}. Using default values.", file=sys.stderr)
    return tolerances

def resolve_player_info(det_player_raw, player_stats):
    """
    Resolves the jersey number and team color from raw player identifiers in the events.
    """
    if det_player_raw is None:
        return None, None

    det_key = str(det_player_raw)
    
    # 1. Look up in player_stats.json if loaded
    if player_stats and det_key in player_stats:
        info = player_stats[det_key]
        return info.get("jersey_number"), info.get("team")

    # 2. Try parsing digit IDs
    if det_key.isdigit():
        return int(det_key), None

    # 3. Fallback team extraction (e.g. GK_Red)
    if det_key.startswith("GK_"):
        parts = det_key.split("_")
        if len(parts) > 1:
            return None, parts[1]

    return None, None

def get_primary_actor(det_event):
    """
    Extract the primary actor ID from the detected event.
    """
    t = det_event.get("type")
    if t in ("pass", "cross"):
        return det_event.get("from")
    elif t in ("tackle", "foul", "interception"):
        return det_event.get("by")
    else:
        return det_event.get("player")

def team_matches(gt_team, det_team):
    if not gt_team:
        return True
    if not det_team:
        return False
    return gt_team.strip().lower() == det_team.strip().lower()

def player_matches(gt_player, det_jersey):
    if gt_player is None:
        return True
    if det_jersey is None:
        return False
    try:
        return int(gt_player) == int(det_jersey)
    except (ValueError, TypeError):
        return False

def normalize_event_type(t):
    t_lower = t.strip().lower()
    if t_lower == "cross":
        return "pass"
    return t_lower

def match_events(gt_events, det_events, player_stats, fps, vid_stride, tolerances, strictness, min_conf):
    """
    Performs greedy one-to-one matching of ground-truth events to detected events.
    """
    # 1. Filter and prepare detected events
    prepared_dets = []
    for det in det_events:
        conf = det.get("confidence", 1.0)
        if conf < min_conf:
            continue

        raw_type = det.get("type", "")
        norm_type = normalize_event_type(raw_type)
        frame = det.get("frame", 0)
        time_s = frame * vid_stride / fps

        actor_raw = get_primary_actor(det)
        jersey, team = resolve_player_info(actor_raw, player_stats)

        prepared_dets.append({
            "raw": det,
            "type": norm_type,
            "time_s": time_s,
            "jersey": jersey,
            "team": team,
            "confidence": conf
        })

    # 2. Sort both by timestamp
    gt_sorted = sorted(gt_events, key=lambda x: x["time_s"])
    det_sorted = sorted(prepared_dets, key=lambda x: x["time_s"])

    matched_det_indices = set()
    matches = []
    unmatched_gt = []

    for gt in gt_sorted:
        gt_type = normalize_event_type(gt["type"])
        gt_time = gt["time_s"]
        gt_team = gt.get("team")
        gt_player = gt.get("player")

        best_idx = None
        best_diff = float("inf")

        type_tolerance = tolerances[gt_type]

        for idx, det in enumerate(det_sorted):
            if idx in matched_det_indices:
                continue

            if det["type"] != gt_type:
                continue

            diff = abs(det["time_s"] - gt_time)
            if diff > type_tolerance:
                continue

            # Strictness verification
            if strictness == "type+team":
                if not team_matches(gt_team, det["team"]):
                    continue
            elif strictness == "type+player":
                if not team_matches(gt_team, det["team"]):
                    continue
                if not player_matches(gt_player, det["jersey"]):
                    continue

            if diff < best_diff:
                best_diff = diff
                best_idx = idx

        if best_idx is not None:
            matched_det_indices.add(best_idx)
            matches.append((gt, det_sorted[best_idx]))
        else:
            unmatched_gt.append(gt)

    # Detections that were not matched
    unmatched_det = [det for idx, det in enumerate(det_sorted) if idx not in matched_det_indices]

    # Calculate statistics
    tp_per_type = defaultdict(int)
    fp_per_type = defaultdict(int)
    fn_per_type = defaultdict(int)

    for gt, det in matches:
        tp_per_type[gt["type"]] += 1

    for gt in unmatched_gt:
        fn_per_type[normalize_event_type(gt["type"])] += 1

    for det in unmatched_det:
        fp_per_type[det["type"]] += 1

    # Get union of all event types in gt and det
    all_types = sorted(list(set(normalize_event_type(e["type"]) for e in gt_events) | set(d["type"] for d in det_sorted)))

    metrics = {}
    for t in all_types:
        tp = tp_per_type[t]
        fp = fp_per_type[t]
        fn = fn_per_type[t]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[t] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Micro-average overall metrics
    total_tp = sum(tp_per_type.values())
    total_fp = sum(fp_per_type.values())
    total_fn = sum(fn_per_type.values())
    
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0

    metrics["overall"] = {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": total_precision,
        "recall": total_recall,
        "f1": total_f1
    }

    return metrics, unmatched_gt, unmatched_det

def print_metrics_table(metrics):
    print("\nEvent-Level Metrics Summary:")
    print(f"{'Event Type':<15} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 75)
    for key, m in sorted(metrics.items()):
        if key == "overall":
            continue
        print(f"{key:<15} | {m['true_positives']:<5} | {m['false_positives']:<5} | {m['false_negatives']:<5} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f}")
    
    print("-" * 75)
    m = metrics["overall"]
    print(f"{'OVERALL (micro)':<15} | {m['true_positives']:<5} | {m['false_positives']:<5} | {m['false_negatives']:<5} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f}\n")

def print_unmatched_tables(unmatched_gt, unmatched_det):
    if unmatched_gt:
        print("Unmatched Ground Truth (Missed Detections):")
        print(f"{'Time (s)':<10} | {'Type':<12} | {'Team':<10} | {'Player':<6}")
        print("-" * 45)
        for gt in sorted(unmatched_gt, key=lambda x: x["time_s"]):
            print(f"{gt['time_s']:<10.2f} | {gt['type']:<12} | {str(gt.get('team', '')):<10} | {str(gt.get('player', '')):<6}")
        print()

    if unmatched_det:
        print("Unmatched Detections (False Positives):")
        print(f"{'Time (s)':<10} | {'Type':<12} | {'Team':<10} | {'Player':<6} | {'Conf':<6}")
        print("-" * 55)
        for det in sorted(unmatched_det, key=lambda x: x["time_s"]):
            print(f"{det['time_s']:<10.2f} | {det['type']:<12} | {str(det.get('team', '')):<10} | {str(det.get('jersey', '')):<6} | {det['confidence']:<6.2f}")
        print()

def run_sweep(gt_events, det_events, player_stats, fps, vid_stride, tolerances, strictness):
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\nConfidence Threshold Sweep (Overall Micro Metrics):")
    print(f"{'Threshold':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 67)
    
    sweep_results = []
    for thresh in thresholds:
        metrics, _, _ = match_events(gt_events, det_events, player_stats, fps, vid_stride, tolerances, strictness, thresh)
        m = metrics["overall"]
        print(f"{thresh:<10.1f} | {m['true_positives']:<5} | {m['false_positives']:<5} | {m['false_negatives']:<5} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f}")
        sweep_results.append({
            "threshold": thresh,
            "metrics": metrics
        })
    print()
    return sweep_results

def main():
    args = parse_args()

    # Load inputs
    gt_events = load_json(args.gt)
    det_events = load_json(args.det)

    # Determine player_stats.json path
    player_stats_path = args.player_stats
    if not player_stats_path:
        det_dir = os.path.dirname(os.path.abspath(args.det))
        potential_path = os.path.join(det_dir, "player_stats.json")
        if os.path.exists(potential_path):
            player_stats_path = potential_path
            print(f"Auto-detected player stats file at: {player_stats_path}")

    player_stats = None
    if player_stats_path:
        player_stats = load_json(player_stats_path)

    # Parse tolerances
    tolerances = parse_type_tolerances(args.type_tolerances, args.tolerance)

    if args.sweep:
        sweep_results = run_sweep(gt_events, det_events, player_stats, args.fps, args.vid_stride, tolerances, args.strictness)
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"sweep": sweep_results}, f, indent=2)
            print(f"Sweep results saved to {args.output}")
    else:
        metrics, unmatched_gt, unmatched_det = match_events(
            gt_events, det_events, player_stats, args.fps, args.vid_stride, tolerances, args.strictness, args.min_conf
        )

        print_metrics_table(metrics)
        print_unmatched_tables(unmatched_gt, unmatched_det)

        if args.output:
            report = {
                "strictness": args.strictness,
                "min_conf": args.min_conf,
                "tolerance": args.tolerance,
                "metrics": metrics,
                "unmatched_gt_count": len(unmatched_gt),
                "unmatched_det_count": len(unmatched_det)
            }
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {args.output}")

if __name__ == "__main__":
    main()
