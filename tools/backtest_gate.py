#!/usr/bin/env python3
import json
import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Automated back-testing quality gate for football metrics pipeline.")
    parser.add_argument("--accuracy_report", type=str, required=True, help="Path to input count accuracy report JSON.")
    parser.add_argument("--eval_events", type=str, default=None, help="Path to event-level evaluation report JSON.")
    parser.add_argument("--baselines", type=str, default="tools/backtest_baselines.json", help="Path to baselines threshold JSON.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory where the verdict markdown is written. Default current directory.")
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

def run_gate(accuracy_report_path, eval_events_path, baselines_path, output_dir):
    # 1. Load inputs
    baselines = load_json(baselines_path)
    acc_report = load_json(accuracy_report_path)
    
    # Map accuracy report by metric
    acc_map = {r["metric"]: r for r in acc_report}
    
    global_pass = True
    count_rows = []
    
    # 2. Process count-accuracy thresholds
    count_baselines = baselines.get("count_accuracy", {})
    for metric, config in count_baselines.items():
        low = config.get("low", 0.0)
        high = config.get("high", float("inf"))
        is_optional = config.get("optional", False)
        
        if metric in acc_map:
            val = acc_map[metric]["accuracy_pct"]
            passed = (low <= val <= high)
            status_str = "PASS" if passed else "FAIL"
            value_str = f"{val:.1f}%"
        else:
            passed = False
            status_str = "MISSING"
            value_str = "N/A"
            
        required = not is_optional
        if required and not passed:
            global_pass = False
            
        count_rows.append({
            "metric": metric,
            "type": "Count Accuracy",
            "value": value_str,
            "threshold": f"{low:.1f}% - {high:.1f}%",
            "required": "Yes" if required else "No",
            "status": status_str
        })
        
    # 3. Process event-level F1 thresholds (if report provided)
    event_rows = []
    event_skipped = False
    
    if eval_events_path:
        eval_events = load_json(eval_events_path)
        metrics = eval_events.get("metrics", {})
        
        event_baselines = baselines.get("event_f1", {})
        for etype, config in event_baselines.items():
            min_f1 = config.get("min", 0.0)
            is_optional = config.get("optional", False)
            
            # Lookup F1 score
            f1 = metrics.get(etype, {}).get("f1")
            
            if f1 is not None:
                passed = (f1 >= min_f1)
                status_str = "PASS" if passed else "FAIL"
                value_str = f"{f1:.4f}"
            else:
                passed = False
                status_str = "MISSING"
                value_str = "N/A"
                
            required = not is_optional
            if required and not passed:
                global_pass = False
                
            event_rows.append({
                "metric": etype,
                "type": "Event F1 Score",
                "value": value_str,
                "threshold": f">= {min_f1:.4f}",
                "required": "Yes" if required else "No",
                "status": status_str
            })
    else:
        event_skipped = True
        
    # 4. Generate Verdict Markdown Report
    os.makedirs(output_dir, exist_ok=True)
    verdict_path = os.path.join(output_dir, "backtest_verdict.md")
    
    headline = "PASS" if global_pass else "FAIL"
    
    md_content = []
    md_content.append(f"# Back-test Verdict: {headline}\n")
    
    # Table headers
    headers = ["Metric / Event Type", "Category", "Value", "Baseline Threshold", "Required", "Status"]
    md_content.append("| " + " | ".join(headers) + " |")
    md_content.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Table rows
    for row in count_rows + event_rows:
        row_str = f"| {row['metric']} | {row['type']} | {row['value']} | {row['threshold']} | {row['required']} | **{row['status']}** |"
        md_content.append(row_str)
        
    md_content.append("")
    if event_skipped:
        md_content.append("*Event-level F1 evaluation was skipped (no --eval_events input provided).*")
        
    # Write to file
    try:
        with open(verdict_path, "w") as f:
            f.write("\n".join(md_content) + "\n")
        print(f"Back-test verdict report saved to {verdict_path}")
    except Exception as e:
        print(f"Error writing verdict markdown to {verdict_path}: {e}", file=sys.stderr)
        
    # Print status to stdout
    print(f"Verdict: {headline}")
    for row in count_rows + event_rows:
        print(f"  {row['type']} - {row['metric']}: {row['value']} (Threshold: {row['threshold']}, Required: {row['required']}) -> {row['status']}")
        
    return 0 if global_pass else 1

def main():
    args = parse_args()
    exit_code = run_gate(
        accuracy_report_path=args.accuracy_report,
        eval_events_path=args.eval_events,
        baselines_path=args.baselines,
        output_dir=args.output_dir
    )
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
