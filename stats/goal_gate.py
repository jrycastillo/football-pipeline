"""
stats/goal_gate.py — reject false "shot" events using visual goal detection.

Our shot detector gates on a homography-derived distance to goal (see
stats/event_logic.py). When the homography mis-fits — which it does on a large
fraction of broadcast frames — a hard midfield pass gets mis-located as "near
goal" and logged as a shot. The pipeline never had a visual goal anchor
(pipeline_consolidated.py: "Only feasible if we detect 'Goal' regions.
skipping for now").

This module supplies that anchor. For each shot it runs a goal detector on the
shot frame and confirms a real goal is visible near the ball. A fast forward
ball movement with no goal nearby is almost always a pass/clearance, so it is
flagged goal_confirmed=False. Non-destructive: it only ADDS the flag; callers
decide whether to clip/count unconfirmed shots.
"""
import math


def _load_cv2():
    import importlib
    return importlib.import_module("cv2")


def _positions(all_frames, vid_stride):
    """src_frame -> {actor: (cx, cy)}: every box keyed by its id, plus 'ball'
    for the ball box. The pipeline writes the ball as cls 32 (NOT 0 — it
    re-tags the ball model's hits "Force Class 32 for EventDetector
    compatibility", pipeline_consolidated.py); the zoom re-acquisition ball is a
    possession-rescue heuristic, not a clean position, so it is skipped.
    all_frames is indexed by processed frame, so source_frame = (i+1)*vid_stride."""
    stride = max(1, int(vid_stride or 1))
    pos = {}
    for i, f in enumerate(all_frames):
        d = {}
        for b in f.get("boxes", []):
            xy = b.get("xyxy")
            if not xy:
                continue
            c = ((xy[0] + xy[2]) / 2.0, (xy[1] + xy[3]) / 2.0)
            if b.get("cls") == 32 and not b.get("zoom"):
                d["ball"] = c
            if b.get("id") is not None:
                d[b["id"]] = c
        if d:
            pos[(i + 1) * stride] = d
    return pos


def _ref_point(pos, src, shooter, window=8):
    """Where the shot was taken from: the ball if detected near src, else the
    shooter's box (players are detected far more reliably than the ball).
    Returns (point, source_label) or (None, None)."""
    for key, label in (("ball", "ball"), (shooter, "shooter")):
        for d in range(0, window + 1):
            for s in ((src,) if d == 0 else (src - d, src + d)):
                if s in pos and key in pos[s]:
                    return pos[s][key], label
    return None, None


def gate_shots(events, all_frames, video_path, goal_weights, vid_stride=1,
               goal_conf=0.35, ball_goal_frac=0.30, device="cuda"):
    """Add goal_confirmed to every 'shot' event.

    A shot is confirmed when a Goal is detected in its frame within
    ball_goal_frac * frame_width pixels of the ball. Returns (events, stats).
    """
    from ultralytics import YOLO
    cv2 = _load_cv2()

    model = YOLO(goal_weights)
    try:
        model.to(device)
    except Exception:
        pass
    goal_ids = [k for k, v in model.names.items() if str(v).lower() == "goal"]
    goal_id = goal_ids[0] if goal_ids else 1

    positions = _positions(all_frames, vid_stride)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    max_dist = ball_goal_frac * width
    dev = 0 if str(device).startswith("cuda") else "cpu"

    out = []
    stats = {"shots": 0, "confirmed": 0, "rejected": 0, "no_ref": 0, "no_goal": 0}
    for ev in events:
        if not (isinstance(ev, dict) and ev.get("type") == "shot"):
            out.append(ev)
            continue
        stats["shots"] += 1
        src = (int(ev.get("frame", 0)) + 1) * max(1, int(vid_stride or 1))
        ref, ref_src = _ref_point(positions, src, ev.get("player"))
        confirmed, gdist, goal_seen = False, None, False

        cap.set(cv2.CAP_PROP_POS_FRAMES, src)
        ok, frame = cap.read()
        if ok and ref is not None:
            res = model(frame, conf=goal_conf, imgsz=640, device=dev, verbose=False)[0]
            for b in res.boxes:
                if int(b.cls[0]) != goal_id:
                    continue
                goal_seen = True
                gx1, gy1, gx2, gy2 = (float(v) for v in b.xyxy[0])
                gc = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)
                d = math.hypot(ref[0] - gc[0], ref[1] - gc[1])
                gdist = d if gdist is None else min(gdist, d)
                if d <= max_dist:
                    confirmed = True
                    break

        ev = dict(ev)
        ev["goal_confirmed"] = confirmed
        ev["gate_ref"] = ref_src
        if gdist is not None:
            ev["ball_goal_px"] = round(gdist, 1)
        out.append(ev)
        stats["confirmed" if confirmed else "rejected"] += 1
        if ref is None:
            stats["no_ref"] += 1
        elif not goal_seen:
            stats["no_goal"] += 1

    cap.release()
    return out, stats


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Goal-gate shot events (visual goal anchor)")
    ap.add_argument("--events", required=True, help="raw_tracks.json")
    ap.add_argument("--all_frames", required=True, help="debug_all_frames.json")
    ap.add_argument("--video", required=True)
    ap.add_argument("--goal_weights", required=True, help="goal detector (.pt) with a 'Goal' class")
    ap.add_argument("--vid_stride", type=int, default=1)
    ap.add_argument("--goal_conf", type=float, default=0.35)
    ap.add_argument("--ball_goal_frac", type=float, default=0.30)
    ap.add_argument("--out", default=None, help="write gated events here (default: stdout summary only)")
    args = ap.parse_args()

    events = json.load(open(args.events))
    all_frames = json.load(open(args.all_frames))
    gated, stats = gate_shots(events, all_frames, args.video, args.goal_weights,
                              vid_stride=args.vid_stride, goal_conf=args.goal_conf,
                              ball_goal_frac=args.ball_goal_frac)
    print(f"[goal_gate] shots={stats['shots']} confirmed={stats['confirmed']} "
          f"rejected={stats['rejected']} (no_ref={stats['no_ref']}, no_goal_in_frame={stats['no_goal']})")
    for ev in gated:
        if isinstance(ev, dict) and ev.get("type") == "shot":
            print(f"  frame={ev.get('frame')} player={ev.get('player')} "
                  f"goal_confirmed={ev.get('goal_confirmed')} "
                  f"ref={ev.get('gate_ref')} ball_goal_px={ev.get('ball_goal_px')}")
    if args.out:
        json.dump(gated, open(args.out, "w"), indent=1)
        print(f"[goal_gate] wrote {args.out}")
