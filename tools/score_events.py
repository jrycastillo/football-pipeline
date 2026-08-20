#!/usr/bin/env python3
"""Score a run's detected events against timestamped ground truth.

Phase 4 of the events-detector plan: make "is it good?" a measurement rather
than an impression. Detected and ground-truth events are matched greedily by
time (nearest first) within a per-type tolerance, then precision / recall / F1
are reported per event type.

Tolerances differ by event because the detectors differ in what they timestamp:
a goal is anchored on the paired shot (or the score change, which the broadcast
graphic delays), so it needs a wide window; a shot is anchored on the strike
itself and should land within a couple of seconds.

Ground truth format (seconds into the source video):

    {
      "match": "hb_full",
      "goals":   [{"time_s": 2031, "team": "home", "scorer": 24},  ...],
      "shots":   [{"time_s": 352,  "player": 24}, ...],
      "assists": [{"time_s": 2025, "player": 8}, ...]
    }

Usage:
    python tools/score_events.py --run output/full_hb --gt ground_truth/hb_full_gt.json
    python tools/score_events.py --run output/r23_hbgoals --gt ... --offset -1920
"""
import argparse
import json
import os

# Per-type match tolerance in seconds (see module docstring).
TOLERANCE = {"goal": 45.0, "shot": 4.0, "assist": 8.0}


def _load_events(run_dir):
    """Events from a run directory (raw_tracks.json is a flat event list)."""
    path = os.path.join(run_dir, "raw_tracks.json")
    if not os.path.exists(path):
        raise SystemExit(f"no raw_tracks.json in {run_dir}")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("events", [])
    return [e for e in data if isinstance(e, dict) and "type" in e]


def _match(detected, truth, tol):
    """Greedy nearest-first matching. Returns (pairs, unmatched_det, unmatched_gt).

    Nearest-first rather than in-order: when two events sit close together, the
    order they appear in the list should not decide which one is 'correct'.
    """
    pairs = []
    dets = sorted(detected, key=lambda e: e["t"])
    gts = sorted(truth, key=lambda e: e["t"])
    cand = []
    for di, d in enumerate(dets):
        for gi, g in enumerate(gts):
            dt = abs(d["t"] - g["t"])
            if dt <= tol:
                cand.append((dt, di, gi))
    cand.sort()
    used_d, used_g = set(), set()
    for dt, di, gi in cand:
        if di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
        pairs.append((dets[di], gts[gi], dt))
    return (pairs,
            [d for i, d in enumerate(dets) if i not in used_d],
            [g for i, g in enumerate(gts) if i not in used_g])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run output dir (holds raw_tracks.json)")
    ap.add_argument("--gt", required=True, help="ground-truth JSON")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds to ADD to detected times (for a clip cut from a "
                         "longer match, pass the clip's start offset as negative)")
    ap.add_argument("--types", default="goal,shot,assist")
    args = ap.parse_args()

    events = _load_events(args.run)
    with open(args.gt) as f:
        gt = json.load(f)

    print(f"run: {args.run}")
    print(f"gt : {args.gt} ({gt.get('match', '?')})")
    if args.offset:
        print(f"offset applied to detected times: {args.offset:+.0f}s")
    print()
    header = f"{'type':<9}{'det':>5}{'gt':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>8}{'rec':>8}{'F1':>8}"
    print(header)
    print("-" * len(header))

    overall = {"tp": 0, "fp": 0, "fn": 0}
    detail = {}
    for etype in [t.strip() for t in args.types.split(",") if t.strip()]:
        det = [{"t": (e.get("time_s") or 0) + args.offset, "e": e}
               for e in events if e.get("type") == etype and e.get("time_s") is not None]
        truth = [{"t": g["time_s"], "g": g} for g in gt.get(etype + "s", [])]
        if not det and not truth:
            continue
        pairs, fps_, fns = _match(det, truth, TOLERANCE.get(etype, 5.0))
        tp, fp, fn = len(pairs), len(fps_), len(fns)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"{etype:<9}{len(det):>5}{len(truth):>5}{tp:>5}{fp:>5}{fn:>5}"
              f"{prec:>7.0%}{rec:>8.0%}{f1:>8.2f}")
        overall["tp"] += tp
        overall["fp"] += fp
        overall["fn"] += fn
        detail[etype] = (pairs, fps_, fns)

    t, f, n = overall["tp"], overall["fp"], overall["fn"]
    p = t / max(1, t + f)
    r = t / max(1, t + n)
    print("-" * len(header))
    print(f"{'OVERALL':<9}{'':>5}{'':>5}{t:>5}{f:>5}{n:>5}{p:>7.0%}{r:>8.0%}"
          f"{2*p*r/max(1e-9,p+r):>8.2f}")

    for etype, (pairs, fps_, fns) in detail.items():
        if not (fps_ or fns):
            continue
        print(f"\n--- {etype} misses ---")
        for d in fps_:
            e = d["e"]
            print(f"  FALSE POSITIVE  {int(d['t']//60):>3}:{int(d['t']%60):02d}  "
                  f"player={e.get('player')} conf={e.get('confidence')}")
        for g in fns:
            print(f"  MISSED          {int(g['t']//60):>3}:{int(g['t']%60):02d}  {g['g']}")


if __name__ == "__main__":
    main()
