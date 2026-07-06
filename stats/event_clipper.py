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

# Default clip-worthy types: the confirmed highlight events. Ball touches are
# also clippable (--types touch) but produce hundreds of clips per match, so
# they stay opt-in until the product decision on portal clip types is made.
DEFAULT_CLIP_TYPES = ("goal", "shot", "save")
_FFMPEG_PROBED = False
_FFMPEG_PATH = None
_FFMPEG_UNAVAILABLE_WARNED = False


def _load_cv2():
    import importlib
    return importlib.import_module("cv2")


def _event_player(event):
    """Best-effort primary player for an event (schema varies by type)."""
    for key in ("player", "from", "by"):
        if event.get(key) is not None:
            return event[key]
    return None


def _ffmpeg_path():
    global _FFMPEG_PROBED, _FFMPEG_PATH
    if not _FFMPEG_PROBED:
        import shutil
        _FFMPEG_PATH = shutil.which("ffmpeg")
        _FFMPEG_PROBED = True
    return _FFMPEG_PATH


def _warn_ffmpeg_unavailable_once():
    global _FFMPEG_UNAVAILABLE_WARNED
    if not _FFMPEG_UNAVAILABLE_WARNED:
        print("[Clipper] ffmpeg not found; falling back to OpenCV mp4v encoding")
        _FFMPEG_UNAVAILABLE_WARNED = True


def _select_codec(codec):
    codec = (codec or "h264").lower()
    if codec not in ("h264", "mp4v"):
        raise ValueError("codec must be 'h264' or 'mp4v'")
    if codec == "mp4v":
        return "mp4v", None

    ffmpeg = _ffmpeg_path()
    if ffmpeg:
        return "h264", ffmpeg

    _warn_ffmpeg_unavailable_once()
    return "mp4v", None


def _build_ffmpeg_command(ffmpeg_path, clip_path, fps, width, height):
    return [
        ffmpeg_path,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "veryfast",
        "-crf", "26",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        clip_path,
    ]


def _remove_file_quiet(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _write_clip_ffmpeg(cap, clip_path, start, end, fps, width, height, ffmpeg_path):
    import subprocess

    cmd = _build_ffmpeg_command(ffmpeg_path, clip_path, fps, width, height)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    written = 0
    try:
        for _ in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            proc.stdin.write(frame.tobytes())
            written += 1
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr else b""
        return_code = proc.wait()
    except Exception:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        proc.wait()
        raise

    if return_code != 0:
        msg = stderr.decode("utf-8", errors="replace").strip()
        if msg:
            raise RuntimeError(f"ffmpeg exited {return_code}: {msg}")
        raise RuntimeError(f"ffmpeg exited {return_code}")
    return written


def _write_clip_mp4v(cv2, cap, clip_path, start, end, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    written = 0
    for _ in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    return written


def _write_clip(cv2, cap, clip_path, start, end, fps, width, height,
                selected_codec, ffmpeg_path):
    if selected_codec == "h264" and ffmpeg_path:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            written = _write_clip_ffmpeg(
                cap, clip_path, start, end, fps, width, height, ffmpeg_path)
            if written > 0:
                return written, "h264"
            print(f"[Clipper] ffmpeg wrote no frames for {os.path.basename(clip_path)}; "
                  "falling back to mp4v")
        except Exception as e:
            print(f"[Clipper] ffmpeg failed for {os.path.basename(clip_path)} ({e}); "
                  "falling back to mp4v")
        _remove_file_quiet(clip_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = _write_clip_mp4v(cv2, cap, clip_path, start, end, fps, width, height)
    return written, "mp4v"


def clip_events(video_path, events, output_dir, vid_stride=1, pad_s=3.0,
                event_types=DEFAULT_CLIP_TYPES, min_conf=None, max_conf=None,
                max_clips=200, codec="h264"):
    """Write pad_s-padded clips for selected events + clips_manifest.json.

    events: the pipeline event list (raw_tracks.json content).
    min_conf/max_conf: optional confidence window — e.g. max_conf=0.75 clips
    only the low-confidence events that need admin review.
    Returns the manifest (list of dicts); empty list if the video is missing.
    """
    codec = (codec or "h264").lower()
    if codec not in ("h264", "mp4v"):
        raise ValueError("codec must be 'h264' or 'mp4v'")

    cv2 = _load_cv2()

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

    selected_codec, ffmpeg_path = ("mp4v", None)
    if selected:
        selected_codec, ffmpeg_path = _select_codec(codec)

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

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

        written, actual_codec = _write_clip(
            cv2, cap, clip_path, start, end, fps, width, height,
            selected_codec, ffmpeg_path)

        if written == 0:
            _remove_file_quiet(clip_path)
            continue

        manifest.append({
            "clip": os.path.join("clips", name),
            "type": ev["type"],
            "player": player,
            "confidence": ev.get("confidence"),
            "identity_confidence": ev.get("identity_confidence"),
            "identity_confidence_receiver": ev.get("identity_confidence_receiver"),
            "codec": actual_codec,
            "size_bytes": os.path.getsize(clip_path) if os.path.exists(clip_path) else 0,
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
    parser.add_argument("--codec", choices=("h264", "mp4v"), default="h264",
                        help="Clip encoder: h264 via ffmpeg when available, or OpenCV mp4v")
    args = parser.parse_args()

    with open(args.events) as f:
        event_list = json.load(f)
    clip_events(args.video, event_list, args.output_dir,
                vid_stride=args.vid_stride, pad_s=args.pad_s,
                event_types=tuple(t.strip() for t in args.types.split(",") if t.strip()),
                min_conf=args.min_conf, max_conf=args.max_conf, codec=args.codec)
