"""
stats/event_clipper.py — extract short video clips around detected events.

Feeds the verification workflow: events (with their confidence + status) are
clipped so an admin can watch and approve/correct/reject them, and verified
key events become player-portal highlights. Clip storage/retrieval and the
admin UI live outside this repo; this module only produces the clips and a
clips_manifest.json describing them.

Event "frame" fields are PROCESSED-frame indices: the pipeline samples one
video frame every vid_stride, so source_frame = (index + 1) * vid_stride
(the pipeline's frame counter starts at 1 and keeps multiples of the stride).

Standalone use (re-clip from a finished run without re-running the pipeline):
    python3 -m stats.event_clipper --video match.mp4 --events output/run/raw_tracks.json \
        --output_dir output/run --types goal,shot,save --vid_stride 3
"""

import json
import os

import cv2

# Default clip-worthy types: the confirmed highlight events. Ball touches are
# also clippable (--types touch) but produce hundreds of clips per match, so
# they stay opt-in until the product decision on portal clip types is made.
DEFAULT_CLIP_TYPES = ("goal", "shot", "save")


def _event_player(event):
    """Best-effort primary player for an event (schema varies by type)."""
    for key in ("player", "from", "by"):
        if event.get(key) is not None:
            return event[key]
    return None


def clip_events(video_path, events, output_dir, vid_stride=1, pad_s=3.0,
                event_types=DEFAULT_CLIP_TYPES, min_conf=None, max_conf=None,
                max_clips=200):
    """Write pad_s-padded clips for selected events + clips_manifest.json.

    events: the pipeline event list (raw_tracks.json content).
    min_conf/max_conf: optional confidence window — e.g. max_conf=0.75 clips
    only the low-confidence events that need admin review.
    Returns the manifest (list of dicts); empty list if the video is missing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Clipper] Cannot open video: {video_path} — no clips written")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pad_frames = int(round(pad_s * fps))

    selected = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") not in event_types:
            continue
        conf = ev.get("confidence")
        if min_conf is not None and (conf is None or conf < min_conf):
            continue
        if max_conf is not None and conf is not None and conf > max_conf:
            continue
        selected.append(ev)
    # Sort by frame so the capture only ever seeks forward
    selected.sort(key=lambda e: e.get("frame", 0))
    if len(selected) > max_clips:
        print(f"[Clipper] {len(selected)} events selected, capping at {max_clips}")
        selected = selected[:max_clips]

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    manifest = []
    for idx, ev in enumerate(selected):
        src_frame = (int(ev.get("frame", 0)) + 1) * max(1, vid_stride)
        start = max(0, src_frame - pad_frames)
        end = src_frame + pad_frames
        if total_frames > 0:
            end = min(end, total_frames - 1)
        if end <= start:
            continue

        player = _event_player(ev)
        name = f"{idx:03d}_{ev['type']}_p{player}_f{src_frame}.mp4"
        clip_path = os.path.join(clips_dir, name)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        written = 0
        for _ in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        writer.release()

        if written == 0:
            os.remove(clip_path)
            continue

        manifest.append({
            "clip": os.path.join("clips", name),
            "type": ev["type"],
            "player": player,
            "confidence": ev.get("confidence"),
            "identity_confidence": ev.get("identity_confidence"),
            "identity_confidence_receiver": ev.get("identity_confidence_receiver"),
            "status": ev.get("status", "unverified"),
            "event_frame": ev.get("frame"),
            "source_frame": src_frame,
            "time_s": round(src_frame / fps, 2),
            "clip_start_s": round(start / fps, 2),
            "clip_end_s": round(end / fps, 2),
        })

    cap.release()

    manifest_path = os.path.join(output_dir, "clips_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Clipper] Wrote {len(manifest)} clip(s) to {clips_dir} + clips_manifest.json")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clip events from a processed match")
    parser.add_argument("--video", required=True, help="Source match video")
    parser.add_argument("--events", required=True, help="Event list JSON (raw_tracks.json)")
    parser.add_argument("--output_dir", required=True, help="Run output dir (clips/ goes here)")
    parser.add_argument("--types", default=",".join(DEFAULT_CLIP_TYPES),
                        help="Comma-separated event types to clip")
    parser.add_argument("--vid_stride", type=int, default=1,
                        help="vid_stride the pipeline ran with (frame index mapping)")
    parser.add_argument("--pad_s", type=float, default=3.0, help="Seconds of padding each side")
    parser.add_argument("--min_conf", type=float, default=None)
    parser.add_argument("--max_conf", type=float, default=None,
                        help="e.g. 0.75 → only low-confidence events (admin review queue)")
    args = parser.parse_args()

    with open(args.events) as f:
        event_list = json.load(f)
    clip_events(args.video, event_list, args.output_dir,
                vid_stride=args.vid_stride, pad_s=args.pad_s,
                event_types=tuple(t.strip() for t in args.types.split(",") if t.strip()),
                min_conf=args.min_conf, max_conf=args.max_conf)
