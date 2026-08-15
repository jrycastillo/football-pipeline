#!/usr/bin/env python3
"""R16 Item 2: timestamped shot/goal/assist ground-truth schema, validator, scorer.
GT schema (JSON): {"clip": str, "fps": float, "vid_stride": int,
  "events": [{"type": "shot"|"goal"|"assist", "frame": int (source-frame), "time_s": float,
              "team": str, "player": int|null, "unattributable": bool, "note": str}]}
Temporal matching: a detected event matches a GT event of the same type if
|t_det - t_gt| <= tol_s (default 2.0s -- one clip lead/tail unit; a scout tolerates
~2s slack around a highlight). One-to-one greedy nearest match."""
import json, sys

REQUIRED = {"type", "frame", "time_s"}
VALID_TYPES = {"shot", "goal", "assist"}

def validate(gt):
    errs = []
    if "events" not in gt: errs.append("missing 'events'")
    for i, e in enumerate(gt.get("events", [])):
        miss = REQUIRED - set(e)
        if miss: errs.append(f"event {i}: missing {miss}")
        if e.get("type") not in VALID_TYPES: errs.append(f"event {i}: bad type {e.get('type')}")
        if not isinstance(e.get("frame"), (int, float)): errs.append(f"event {i}: frame not numeric")
        if e.get("player") is None and not e.get("unattributable", False):
            errs.append(f"event {i}: player null but not marked unattributable")
    return errs

def score(det, gt, etype, tol_s=2.0):
    """det/gt: lists of dicts with time_s. Returns tp, fp, fn, matches."""
    G = sorted([e for e in gt if e.get("type") == etype], key=lambda x: x["time_s"])
    D = sorted([e for e in det if e.get("type") == etype], key=lambda x: x["time_s"])
    used = set(); tp = 0; matches = []
    for d in D:
        best = None
        for j, g in enumerate(G):
            if j in used: continue
            dt = abs(d["time_s"] - g["time_s"])
            if dt <= tol_s and (best is None or dt < best[1]): best = (j, dt)
        if best is not None:
            used.add(best[0]); tp += 1; matches.append((d["time_s"], G[best[0]]["time_s"], round(best[1], 2)))
    fp = len(D) - tp; fn = len(G) - tp
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return dict(type=etype, tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, tol_s=tol_s, matches=matches)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "validate":
        gt = json.load(open(sys.argv[2])); errs = validate(gt)
        print("VALID" if not errs else "INVALID: " + "; ".join(errs))
    elif cmd == "score":
        det = json.load(open(sys.argv[2])); gt = json.load(open(sys.argv[3]))
        tol = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
        de = det if isinstance(det, list) else det.get("events", det.get("clips", []))
        ge = gt.get("events", [])
        for et in ("shot", "goal", "assist"):
            print(json.dumps(score(de, ge, et, tol)))
