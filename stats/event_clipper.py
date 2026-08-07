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

import bisect
import json
import os

# Default clip-worthy types: the confirmed highlight events. Ball touches are
# also clippable (--types touch) but produce hundreds of clips per match, so
# they stay opt-in until the product decision on portal clip types is made.
DEFAULT_CLIP_TYPES = ("goal", "assist", "shot", "save", "goal_restart")
_FFMPEG_PROBED = False
_FFMPEG_PATH = None
_FFMPEG_UNAVAILABLE_WARNED = False


def _load_cv2():
    import importlib
    return importlib.import_module("cv2")


def _fmt_clock(seconds):
    """Seconds -> mm:ss (the event/clip timestamp Babak wants labelled)."""
    s = max(0, int(round(seconds)))
    return f"{s // 60:02d}:{s % 60:02d}"


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


def _write_clip_ffmpeg(cap, clip_path, start, end, fps, width, height, ffmpeg_path,
                       draw_hook=None):
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
        for off in range(end - start + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if draw_hook is not None:
                out = draw_hook(frame, start + off)
                if out is not None:
                    frame = out
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


def _write_clip_mp4v(cv2, cap, clip_path, start, end, fps, width, height, draw_hook=None):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    written = 0
    for off in range(end - start + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if draw_hook is not None:
            out = draw_hook(frame, start + off)
            if out is not None:
                frame = out
        writer.write(frame)
        written += 1
    writer.release()
    return written


def _write_clip(cv2, cap, clip_path, start, end, fps, width, height,
                selected_codec, ffmpeg_path, draw_hook=None):
    if selected_codec == "h264" and ffmpeg_path:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            written = _write_clip_ffmpeg(
                cap, clip_path, start, end, fps, width, height, ffmpeg_path, draw_hook)
            if written > 0:
                return written, "h264"
            print(f"[Clipper] ffmpeg wrote no frames for {os.path.basename(clip_path)}; "
                  "falling back to mp4v")
        except Exception as e:
            print(f"[Clipper] ffmpeg failed for {os.path.basename(clip_path)} ({e}); "
                  "falling back to mp4v")
        _remove_file_quiet(clip_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = _write_clip_mp4v(cv2, cap, clip_path, start, end, fps, width, height, draw_hook)
    return written, "mp4v"


def _build_frame_boxes(all_frames, vid_stride):
    """Map source-frame index -> {player_id: (x1,y1,x2,y2)} from the pipeline's
    per-frame box data. all_frames is indexed by processed frame; the source
    frame is (index+1)*vid_stride (see module docstring)."""
    fb = {}
    for idx, f in enumerate(all_frames or []):
        src = (idx + 1) * max(1, vid_stride)
        boxes = {}
        for b in f.get("boxes", []):
            pid = b.get("id")
            xy = b.get("xyxy")
            if pid is not None and xy:
                boxes[pid] = tuple(int(v) for v in xy)
        if boxes:
            fb[src] = boxes
    return fb


def _load_track_jersey(output_dir):
    """Read track_jersey.json (written by the pipeline) -> {track_id: label},
    where label is '#<jersey>' when the track locked a number, else ''. Lets the
    clipper tag EVERY player box with its jersey, not just the actor. Returns {}
    when the file is absent (older runs) — boxes then show only 'T<id>'."""
    import json
    path = os.path.join(output_dir or "", "track_jersey.json")
    if not os.path.isfile(path):
        return {}
    try:
        raw = json.load(open(path))
    except Exception:
        return {}
    out = {}
    for tid, info in (raw or {}).items():
        try:
            jn = info.get("jersey") if isinstance(info, dict) else info
            out[int(tid)] = f"#{jn}" if jn is not None else ""
        except (TypeError, ValueError):
            continue
    return out


def _build_frame_balls(all_frames, vid_stride):
    """Map source-frame index -> ball center (cx, cy) from the cls-32 ball boxes
    the pipeline writes per frame. Zoom re-acquisition balls are skipped — they
    are a possession-rescue heuristic, not a clean position (see metrics.py). At
    most one ball per frame: the highest-confidence detection."""
    fb = {}
    for idx, f in enumerate(all_frames or []):
        src = (idx + 1) * max(1, vid_stride)
        best = None  # (conf, cx, cy)
        for b in f.get("boxes", []):
            if b.get("cls") != 32 or b.get("zoom"):
                continue
            xy = b.get("xyxy")
            if not xy:
                continue
            conf = b.get("conf") or 0.0
            if best is None or conf > best[0]:
                best = (conf, (xy[0] + xy[2]) * 0.5, (xy[1] + xy[3]) * 0.5)
        if best is not None:
            fb[src] = (best[1], best[2])
    return _stabilize_ball_track(fb, max(1, vid_stride))


def _build_frame_class_boxes(all_frames, vid_stride, cls):
    """Map source-frame index -> highest-confidence box (x1,y1,x2,y2) of a given
    class. Used for the detected goal (cls 33) and goalkeeper (cls 1) so shot
    clips can show the ball heading to a keeper-defended goal."""
    fb = {}
    for idx, f in enumerate(all_frames or []):
        src = (idx + 1) * max(1, vid_stride)
        best = None  # (conf, box)
        for b in f.get("boxes", []):
            if b.get("cls") != cls:
                continue
            xy = b.get("xyxy")
            if not xy:
                continue
            conf = b.get("conf") or 0.0
            if best is None or conf > best[0]:
                best = (conf, tuple(float(v) for v in xy))
        if best is not None:
            fb[src] = best[1]
    return fb


def _nearest_class_box(frame_boxes, src_idx, window):
    """The class box at the sampled frame nearest src_idx within +/- window
    source frames (goals/keepers are near-static, so a small window is fine)."""
    if not frame_boxes:
        return None
    best = None
    for k, box in frame_boxes.items():
        d = abs(k - src_idx)
        if d <= window and (best is None or d < best[0]):
            best = (d, box)
    return best[1] if best else None


def _stabilize_ball_track(fb, stride):
    """Reject implausible-jump ball detections so the drawn ball moves smoothly.

    The ball detector occasionally fires on a white shirt or a pitch line far
    from the real ball for a single frame — the marker then teleports across the
    pitch and back ("jumping"). A real ball can't move more than a bounded
    pixel distance between detections, so we walk the detections in time order
    and drop any that jump faster than that ceiling from the last accepted
    position. The track re-seeds after a long stale gap so a genuine relocation
    (throw-in, restart on the far side) is not rejected forever."""
    if len(fb) < 3:
        return fb
    MAX_PX_PER_SRC_FRAME = 55.0    # generous ceiling for real ball motion @1080p
    BASE_PX = 45.0                 # slack for jitter on a near-static ball
    reseed_gap = 30 * stride       # src frames with no accepted ball -> trust next
    out, last_k, last_p, dropped = {}, None, None, 0
    for k in sorted(fb.keys()):
        p = fb[k]
        if last_p is None:
            out[k] = p; last_k, last_p = k, p; continue
        gap = k - last_k
        d = ((p[0] - last_p[0]) ** 2 + (p[1] - last_p[1]) ** 2) ** 0.5
        if d <= BASE_PX + MAX_PX_PER_SRC_FRAME * gap or gap > reseed_gap:
            out[k] = p; last_k, last_p = k, p
        else:
            dropped += 1  # teleport -> false positive, keep the smooth track
    if dropped:
        print(f"[Clipper] ball track: dropped {dropped} jump/outlier detection(s) "
              f"of {len(fb)} for a stable overlay")
    return out


def _make_ball_lookup(frame_balls, stride):
    """Return ball_at(src_idx) -> (x, y, is_real) or None. Interpolates between
    detections (matching the pipeline's ball track), but only across gaps up to
    the pipeline's MAX_GAP (25 processed frames): a larger gap means the ball is
    genuinely lost, and we draw nothing rather than inventing a position. is_real
    is True when the bracketing detections are consecutive samples (real tracked
    motion) and False when interpolated across a detection gap."""
    if not frame_balls:
        return None
    bkeys = sorted(frame_balls.keys())
    max_gap = 25 * max(1, stride)

    def ball_at(src_idx):
        if src_idx in frame_balls:
            x, y = frame_balls[src_idx]
            return (x, y, True)
        lo = hi = None
        for k in bkeys:
            if k <= src_idx:
                lo = k
            else:
                hi = k
                break
        if lo is None or hi is None or hi - lo > max_gap:
            return None
        t = (src_idx - lo) / (hi - lo)
        (lx, ly), (hx, hy) = frame_balls[lo], frame_balls[hi]
        return (lx + (hx - lx) * t, ly + (hy - ly) * t, hi - lo <= stride)

    return ball_at


def _make_draw_hook(cv2, frame_boxes, player_id, stride, label, event_src=None,
                    zoom=False, frame_balls=None, frame_goals=None, frame_gks=None,
                    track_jersey=None, draw_all_boxes=False, fps=None):
    """Return a draw_hook(frame, src_idx) that boxes the event player, LOCKED to
    a single moving target. When zoom=True the hook also crops tight on the
    player's torso (from the real box coordinates, so it is immune to the
    coloured advertising boards that fool pixel-based re-zooming) and upscales
    to the clip's frame size — returning the transformed frame.

    When frame_balls is supplied the hook also overlays the tracked ball (green
    marker) and a line from the event player to it, so the clip visually proves
    the event is ball-grounded. A solid marker is a real detection at that frame;
    a hollow ring is interpolated across a detection gap; nothing is drawn when
    the ball is genuinely lost (gap > pipeline MAX_GAP).

    The box is anchored on the ACTOR — the box nearest the ball at the event
    frame, which is exactly the owner the event engine credited. This is robust
    to the id-space mismatch (debug_all_frames boxes carry raw ByteTrack track
    ids; events carry remapped jersey numbers) that made id-matching land on the
    wrong player or none. It then follows that box by spatial nearest-neighbour
    across the clip, interpolating gaps, so the highlight stays glued to the
    actor. When no ball is available it falls back to id-matching (correct when
    box ids already ARE jersey numbers, i.e. after the production remap).
    """
    if not frame_boxes or player_id is None:
        # No single actor to box (e.g. a goal_restart / kickoff marker). Still make
        # the clip legible: a big event-LABEL banner at the top (so it clearly reads
        # as a KICKOFF, not raw footage), every player boxed (the kickoff formation),
        # the ball, and the running clock.
        if fps or frame_balls or frame_boxes:
            _ball_at0 = _make_ball_lookup(frame_balls, stride)
            _keys0 = sorted(frame_boxes.keys()) if frame_boxes else []

            def _minimal_hook(frame, src_idx):
                _h, _w = frame.shape[:2]
                # every player boxed (cyan T{id} #{jersey}) at the nearest sample
                if draw_all_boxes and _keys0:
                    _pos = bisect.bisect_left(_keys0, src_idx)
                    _cand = [_keys0[i] for i in (_pos - 1, _pos) if 0 <= i < len(_keys0)]
                    _nk = min(_cand, key=lambda k: abs(k - src_idx)) if _cand else None
                    if _nk is not None and abs(_nk - src_idx) <= stride:
                        for _tid, _b in frame_boxes[_nk].items():
                            bx1, by1, bx2, by2 = (int(round(v)) for v in _b)
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 2)
                            _lab = f"T{_tid}"
                            _j = track_jersey.get(_tid, "") if track_jersey else ""
                            if _j:
                                _lab += f" {_j}"
                            cv2.putText(frame, _lab, (bx1, max(10, by1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
                # ball
                if _ball_at0 is not None:
                    bp = _ball_at0(src_idx)
                    if bp is not None:
                        bx, by = int(round(bp[0])), int(round(bp[1]))
                        cv2.circle(frame, (bx, by), 10, (0, 0, 0), -1 if bp[2] else 2, cv2.LINE_AA)
                        cv2.circle(frame, (bx, by), 7, (60, 245, 60), -1 if bp[2] else 2, cv2.LINE_AA)
                        _minimal_hook.drew["ball"] += 1
                # prominent event-label banner (top-centre) so it reads as a KICKOFF
                (_tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                _lx = max(10, (_w - _tw) // 2)
                cv2.rectangle(frame, (_lx - 14, 18), (_lx + _tw + 14, 34 + _th + 14), (0, 0, 0), -1)
                cv2.putText(frame, label, (_lx, 34 + _th), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 3, cv2.LINE_AA)
                # running clock
                if fps:
                    _c = _fmt_clock(src_idx / fps)
                    cv2.putText(frame, _c, (12, _h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, _c, (12, _h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                return None
            _minimal_hook.drew = {"any": True, "ball": 0}
            return _minimal_hook
        return None
    keys = sorted(frame_boxes.keys())
    if not keys:
        return None
    drew = {"any": False, "ball": 0}
    ball_at = _make_ball_lookup(frame_balls, stride)

    def _center(b):
        return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

    def _feet(b):
        return ((b[0] + b[2]) * 0.5, b[3])   # bottom-centre = where the ball is

    def _dist(a, b):
        (ax, ay), (bx, by) = _center(a), _center(b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def _dist_pt(box, pt):
        cx, cy = _center(box)
        return ((cx - pt[0]) ** 2 + (cy - pt[1]) ** 2) ** 0.5

    def _dist_feet(box, pt):
        fx, fy = _feet(box)
        return ((fx - pt[0]) ** 2 + (fy - pt[1]) ** 2) ** 0.5

    # 1. Anchor on the ACTOR: the box nearest the ball at the event frame. That
    # is precisely the "owner" the event engine credited (ownership = nearest
    # player to the ball), so it is robust to the box-id / jersey-number mismatch
    # — debug_all_frames boxes carry raw ByteTrack track ids, but events carry
    # remapped jersey numbers, so matching player_id against box ids lands on the
    # wrong player (or none). Nearest-to-ball needs no id agreement.
    anchor_src = event_src if event_src is not None else keys[len(keys) // 2]
    anchor_i, anchor_box = None, None
    if ball_at is not None:
        for k in sorted(keys, key=lambda kk: abs(kk - anchor_src)):
            bp = ball_at(k)
            if bp is None:
                continue
            best, bestd = None, 1e18
            for b in frame_boxes[k].values():
                d = _dist_feet(b, bp)   # the carrier's FEET are on the ball, not their torso
                if d < bestd:
                    best, bestd = b, d
            # Ownership only fires when a player is near the ball; a huge nearest
            # distance means no plausible owner in this sampled frame — keep
            # scanning outward for one before giving up.
            if best is not None and bestd <= 500.0:
                anchor_i, anchor_box = keys.index(k), best
                break
    # 1b. Fallback (no ball, or no owner found): the old id-match anchor. Works
    # when box ids ARE jersey numbers (production, orchestrator remap applied).
    if anchor_box is None:
        for k in sorted(keys, key=lambda kk: abs(kk - anchor_src)):
            if player_id in frame_boxes[k]:
                anchor_i, anchor_box = keys.index(k), frame_boxes[k][player_id]
                break
    if anchor_box is None:
        return None  # no owner and player never detected -> draw nothing

    # Jump gate scaled to the player's size: a player won't move more than a
    # couple of body-lengths between sampled frames, so a larger step is an id
    # collision, not real motion.
    max_jump = max(120.0, 2.2 * max(anchor_box[3] - anchor_box[1], 40))

    # 2. Walk out from the anchor in both directions, tracking by proximity.
    track = {keys[anchor_i]: anchor_box}
    for step in (1, -1):
        prev = anchor_box
        j = anchor_i + step
        while 0 <= j < len(keys):
            boxes = frame_boxes[keys[j]]
            if player_id in boxes and _dist(boxes[player_id], prev) <= max_jump:
                cand = boxes[player_id]                    # trust the label when plausible
            else:
                cand, bestd = None, max_jump               # else nearest box of any id
                for b in boxes.values():
                    d = _dist(b, prev)
                    if d < bestd:
                        cand, bestd = b, d
            if cand is not None:
                track[keys[j]] = cand
                prev = cand
            j += step

    tkeys = sorted(track.keys())

    def _interp_box(src_idx):
        if src_idx <= tkeys[0]:
            return track[tkeys[0]]
        if src_idx >= tkeys[-1]:
            return track[tkeys[-1]]
        lo = max(k for k in tkeys if k <= src_idx)
        hi = min(k for k in tkeys if k >= src_idx)
        if hi == lo:
            return track[lo]
        a, b = track[lo], track[hi]
        t = (src_idx - lo) / (hi - lo)
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))

    # Constant zoom window sized to the player: ~3x the median tracked box
    # height, keeping the clip's aspect ratio. Recomputed per clip.
    _bhs = sorted(b[3] - b[1] for b in track.values())
    crop_h0 = (_bhs[len(_bhs) // 2] if _bhs else 300.0) * 3.0

    # Temporal smoothing state so the box and ball GLIDE instead of jittering
    # frame-to-frame (raw detections wobble a few px each frame). EMA: a lower
    # alpha is smoother but lags more; the ball resets on loss so it does not
    # slide from a stale position when it re-appears.
    _sm = {"box": None, "ball": None}
    _BOX_ALPHA, _BALL_ALPHA = 0.4, 0.5

    def hook(frame, src_idx):
        box = _interp_box(src_idx)
        if not box:
            return None
        if _sm["box"] is None:
            _sm["box"] = list(box)
        else:
            _sm["box"] = [_BOX_ALPHA * box[k] + (1 - _BOX_ALPHA) * _sm["box"][k]
                          for k in range(4)]
        box = _sm["box"]
        x1, y1, x2, y2 = (int(round(v)) for v in box)

        # Context boxes: every OTHER tracked player in this frame, drawn in cyan
        # (distinct from the actor's amber) and labelled "T{track_id} #{jersey}"
        # — jersey only when that track locked a number. Drawn first so the
        # actor's amber box lands on top. Uses the nearest sampled frame (boxes
        # exist only every `stride` frames); the actor's own box is skipped by
        # nearest-centre match so it isn't double-drawn under the highlight.
        if draw_all_boxes and keys:
            _pos = bisect.bisect_left(keys, src_idx)
            _cand = [keys[i] for i in (_pos - 1, _pos) if 0 <= i < len(keys)]
            _nk = min(_cand, key=lambda k: abs(k - src_idx)) if _cand else None
            if _nk is not None and abs(_nk - src_idx) <= stride:
                _ex, _ey = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                for _tid, _b in frame_boxes[_nk].items():
                    bx1, by1, bx2, by2 = (int(round(v)) for v in _b)
                    if abs((bx1 + bx2) * 0.5 - _ex) < 30 and abs((by1 + by2) * 0.5 - _ey) < 30:
                        continue  # this is the actor — drawn amber below
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 2)
                    _lab = f"T{_tid}"
                    _j = track_jersey.get(_tid, "") if track_jersey else ""
                    if _j:
                        _lab += f" {_j}"
                    cv2.putText(frame, _lab, (bx1, max(10, by1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 3)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
        drew["any"] = True

        # Shot context: draw the detected goal (red) and goalkeeper (pink) so a
        # shot clip reads as "ball -> keeper-defended goal" — the exact thing the
        # shot gate now requires. Only when supplied (shot clips), so other
        # events stay uncluttered.
        if frame_goals:
            gb = _nearest_class_box(frame_goals, src_idx, 6 * stride)
            if gb:
                gx1, gy1, gx2, gy2 = (int(round(v)) for v in gb)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (40, 40, 235), 2)
                cv2.putText(frame, "GOAL", (gx1, max(12, gy1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 235), 2, cv2.LINE_AA)
        if frame_gks:
            kb = _nearest_class_box(frame_gks, src_idx, 6 * stride)
            if kb:
                kx1, ky1, kx2, ky2 = (int(round(v)) for v in kb)
                cv2.rectangle(frame, (kx1, ky1), (kx2, ky2), (203, 92, 255), 2)
                cv2.putText(frame, "GK", (kx1, max(12, ky1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (203, 92, 255), 2, cv2.LINE_AA)

        # Ball overlay: draw the tracked ball, so the clip visually proves the
        # event is ball-grounded. Drawn in full-frame coords BEFORE any zoom
        # crop, so it scales with the crop. Solid marker = ball really tracked
        # here; hollow ring = interpolated across a detection gap.
        # The ownership line (player -> ball) is drawn ONLY while the ball is
        # close enough to be owned; a long cross-pitch line would just mean the
        # ball is elsewhere (before/after possession), which is noise, not signal.
        if ball_at is not None:
            bp = ball_at(src_idx)
            if bp is not None:
                # EMA-smooth the drawn ball so it glides. A big jump (a re-acquire
                # after a loss, or a real fast kick) snaps instead of lagging.
                _rx, _ry, is_real = bp[0], bp[1], bp[2]
                if _sm["ball"] is None or ((_rx - _sm["ball"][0]) ** 2 + (_ry - _sm["ball"][1]) ** 2) ** 0.5 > 120:
                    _sm["ball"] = [_rx, _ry]
                else:
                    _sm["ball"] = [_BALL_ALPHA * _rx + (1 - _BALL_ALPHA) * _sm["ball"][0],
                                   _BALL_ALPHA * _ry + (1 - _BALL_ALPHA) * _sm["ball"][1]]
                bx, by = int(round(_sm["ball"][0])), int(round(_sm["ball"][1]))
                pcx, pcy = (x1 + x2) // 2, (y1 + y2) // 2
                own_px = max(220.0, 2.5 * max(y2 - y1, 40))
                if _dist_pt(box, (bx, by)) <= own_px:
                    cv2.line(frame, (pcx, pcy), (bx, by), (0, 215, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 10, (0, 0, 0),
                           -1 if is_real else 2, cv2.LINE_AA)
                cv2.circle(frame, (bx, by), 7, (60, 245, 60),
                           -1 if is_real else 2, cv2.LINE_AA)
                drew["ball"] += 1
            else:
                _sm["ball"] = None  # ball genuinely lost -> reset so it doesn't slide on return

        # Running match clock (mm:ss) bottom-left — the event/clip timestamp so
        # the admin can locate + compare. Drawn last, in fixed screen coords, on
        # whichever frame is returned (full frame, or the zoom crop) so it is
        # never cropped out.
        def _put_clock(target):
            if not fps:
                return
            _c = _fmt_clock(src_idx / fps)
            _h = target.shape[0]
            cv2.putText(target, _c, (12, _h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(target, _c, (12, _h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if not zoom:
            _put_clock(frame)
            return None
        # Crop tight on the torso (upper third of the box) and upscale to the
        # clip's frame size. Uses the box coords, not colour detection.
        h, w = frame.shape[:2]
        crop_h = int(min(max(crop_h0, 200), h))
        crop_w = int(min(max(crop_h * w / float(h), 200), w))
        cx = (x1 + x2) / 2.0
        cy = y1 + 0.30 * max(y2 - y1, 40)
        left = int(min(max(cx - crop_w / 2.0, 0), w - crop_w))
        top = int(min(max(cy - crop_h / 2.0, 0), h - crop_h))
        crop = frame[top:top + crop_h, left:left + crop_w]
        out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        _put_clock(out)
        return out

    hook.drew = drew  # inspect after writing to tally clips that got a box
    return hook


def clip_events(video_path, events, output_dir, vid_stride=1, pad_s=3.0,
                event_types=DEFAULT_CLIP_TYPES, min_conf=None, max_conf=None,
                max_clips=200, codec="h264", frame_boxes=None, zoom=False,
                valid_players=None, require_box=False, frame_balls=None,
                clean_passes=True, frame_goals=None, frame_gks=None,
                track_jersey=None, draw_all_boxes=False,
                pad_before_s=None, pad_after_s=None):
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
    # Asymmetric padding: pad_before_s / pad_after_s override pad_s per side, so a
    # clip can lead in with the build-up and run on to show the outcome — enough
    # to read the whole event, not just the instant.
    _pb = pad_before_s if pad_before_s is not None else pad_s
    _pa = pad_after_s if pad_after_s is not None else pad_s
    pad_before_frames = int(round(_pb * fps))
    pad_after_frames = int(round(_pa * fps))

    selected = []
    _dropped_pass = 0
    _dropped_trackid = 0
    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") not in event_types:
            continue
        conf = ev.get("confidence")
        if min_conf is not None and (conf is None or conf < min_conf):
            continue
        if max_conf is not None and conf is not None and conf > max_conf:
            continue
        # Drop events involving an unresolved track id (>99 — not a jersey) in
        # ANY role (actor OR duel partner OR receiver). Jersey numbers are <=99,
        # so a value above that is a raw ByteTrack fragment that never resolved
        # to a player; one such fragment otherwise leaks into several events it
        # cannot be attributed to (e.g. #108 -> a pass, a dribble, an interception).
        if any(isinstance(v, (int, float)) and v > 99
               for v in (ev.get(k) for k in ("player", "by", "from", "against", "on", "to"))):
            _dropped_trackid += 1
            continue
        # Clean-pass filter: a "pass" highlight must be a genuine, completed pass
        # to a teammate. Drop the misleading cases —
        #   - complete is False -> the ball was intercepted; that moment is
        #     already emitted as an interception by the opponent, so clipping it
        #     as a "pass" double-represents it and looks false.
        #   - the receiver is an unresolved track id (>99, not a jersey) -> we
        #     can't say who received it, so it isn't a trustworthy pass clip.
        if clean_passes and ev.get("type") in ("pass", "cross"):
            if ev.get("complete") is False:
                _dropped_pass += 1
                continue
            _to = ev.get("to")
            if isinstance(_to, (int, float)) and _to > 99:
                _dropped_pass += 1
                continue
        selected.append(ev)
    if _dropped_pass:
        print(f"[Clipper] dropped {_dropped_pass} pass/cross events "
              "(intercepted or unresolved receiver)")
    if _dropped_trackid:
        print(f"[Clipper] dropped {_dropped_trackid} events with an "
              "unresolved track-id participant")

    # Duel dedup: a FAILED dribble emits BOTH a "dribble" and a mirror "tackle"
    # at the same frame (event_logic) — one physical duel between the same two
    # players. Clipping both makes two clips of the same moment that look like a
    # false duplicate. Keep the higher-confidence one of each duel.
    _kept, _duel_at = [], {}
    _dropped_dup = 0
    for ev in selected:
        et = ev.get("type")
        if et in ("dribble", "tackle"):
            pair = frozenset((ev.get("player") if et == "dribble" else ev.get("by"),
                              ev.get("against") if et == "dribble" else ev.get("on")))
            key = (ev.get("frame"), pair)
            prev = _duel_at.get(key)
            if prev is not None:
                if (ev.get("confidence") or 0) > (_kept[prev].get("confidence") or 0):
                    _kept[prev] = ev
                _dropped_dup += 1
                continue
            _duel_at[key] = len(_kept)
        _kept.append(ev)
    selected = _kept
    if _dropped_dup:
        print(f"[Clipper] deduped {_dropped_dup} duplicate duel events "
              "(same-frame dribble+tackle -> kept the higher-confidence one)")

    # Content filter: only clip events attributed to a valid (roster) player.
    # Drops referee / track-id / off-roster attributions — e.g. a "tackle"
    # credited to #97 that is really a referee, or a track id that never
    # resolved to a jersey number.
    if valid_players is not None:
        def _rostered(e):
            try:
                return int(_event_player(e)) in valid_players
            except (TypeError, ValueError):
                return False
        _before = len(selected)
        selected = [e for e in selected if _rostered(e)]
        if _before - len(selected):
            print(f"[Clipper] dropped {_before - len(selected)} events with "
                  "off-roster / unresolved player ids")

    # Sort by frame so the capture only ever seeks forward
    selected.sort(key=lambda e: e.get("frame", 0))
    if len(selected) > max_clips:
        print(f"[Clipper] {len(selected)} events selected, capping at {max_clips}")
        selected = selected[:max_clips]

    selected_codec, ffmpeg_path = ("mp4v", None)
    if selected:
        selected_codec, ffmpeg_path = _select_codec(codec)

    # Guard against the silent-no-box failure: frame_boxes is keyed by the box
    # "id", which is only the jersey number AFTER the pipeline's Option A remap
    # rewrites all_frames in place. If frame_boxes is supplied but no event
    # player matches any box id, boxes would render on zero clips — warn loudly
    # instead (usually means all_frames still holds raw ByteTrack track IDs).
    if frame_boxes:
        _box_ids = set()
        for _b in frame_boxes.values():
            _box_ids.update(_b.keys())
        _event_players = {_event_player(ev) for ev in selected}
        _event_players.discard(None)
        if _event_players and not (_event_players & _box_ids):
            print("[Clipper] WARNING: frame_boxes supplied but no event player "
                  "matches any box id — highlight boxes will NOT render. Are "
                  "all_frames box IDs still raw track IDs (Option A remap not applied)?")

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    manifest = []
    clips_with_box = 0
    clips_with_ball = 0
    kept = 0
    for ev in selected:
        src_frame = (int(ev.get("frame", 0)) + 1) * max(1, vid_stride)
        start = max(0, src_frame - pad_before_frames)
        end = src_frame + pad_after_frames
        if total_frames > 0:
            end = min(end, total_frames - 1)
        if end <= start:
            continue

        player = _event_player(ev)

        # Label the box with the event + its timestamp (mm:ss) so the admin can
        # locate and compare it (Babak: "timestamp of events should be labelled").
        # Prefer the event's own time_s if present; else derive from the frame.
        _ts = ev.get("time_s")
        if _ts is None and fps:
            _ts = src_frame / fps
        _clk = f"  {_fmt_clock(_ts)}" if _ts is not None else ""
        label = (f"#{player} {ev['type']}" if player is not None else ev["type"]) + _clk
        if ev.get("type") == "shot" and "goal_confirmed" in ev:
            label += " [ON GOAL]" if ev["goal_confirmed"] else " [NO GOAL NEARBY]"
        # Surface the goal-restart cross-check right on the clip label.
        if ev.get("type") == "goal_restart":
            label = ("KICKOFF - CHECK GOAL" + _clk) if ev.get("goal_confirmation") \
                else ("KICKOFF - POSSIBLE MISSED GOAL" + _clk)

        # Highlight the event player with a bounding box (keeps the full-frame
        # context so the admin can verify the play, not just the player).
        # Shot clips also get the goal + goalkeeper drawn, so the shot reads as
        # "ball -> keeper-defended goal" (Babak: focus on true shots on goal).
        _is_shot = ev.get("type") == "shot"
        draw_hook = _make_draw_hook(
            cv2, frame_boxes, player, max(1, vid_stride),
            label=label, event_src=src_frame, zoom=zoom,
            frame_balls=frame_balls,
            frame_goals=frame_goals if _is_shot else None,
            frame_gks=frame_gks if _is_shot else None,
            track_jersey=track_jersey, draw_all_boxes=draw_all_boxes, fps=fps)

        # With require_box on, skip events whose player is never tracked in the
        # clip window (draw_hook is None) — no highlightable subject, so a
        # full-frame clip the admin can't act on. Removes the "no box" clips.
        if require_box and draw_hook is None:
            continue

        name = f"{kept:03d}_{ev['type']}_p{player}_f{src_frame}.mp4"
        clip_path = os.path.join(clips_dir, name)

        written, actual_codec = _write_clip(
            cv2, cap, clip_path, start, end, fps, width, height,
            selected_codec, ffmpeg_path, draw_hook)

        if written == 0:
            _remove_file_quiet(clip_path)
            continue

        _drew = getattr(draw_hook, "drew", {}) if draw_hook is not None else {}
        boxed = bool(_drew.get("any"))
        ball_frames = int(_drew.get("ball", 0))
        if boxed:
            clips_with_box += 1
        if ball_frames:
            clips_with_ball += 1

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
            "player_boxed": boxed,
            "ball_frames": ball_frames,
        })
        kept += 1

    cap.release()

    manifest_path = os.path.join(output_dir, "clips_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    box_note = f", {clips_with_box} with player box" if frame_boxes else ""
    ball_note = f", {clips_with_ball} showing the ball" if frame_balls else ""
    print(f"[Clipper] Wrote {len(manifest)} clip(s){box_note}{ball_note} to {clips_dir} "
          "+ clips_manifest.json")
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
    parser.add_argument("--pad_before_s", type=float, default=None,
                        help="Seconds BEFORE the event (overrides --pad_s for the lead-in)")
    parser.add_argument("--pad_after_s", type=float, default=None,
                        help="Seconds AFTER the event (overrides --pad_s for the outcome)")
    parser.add_argument("--min_conf", type=float, default=None)
    parser.add_argument("--max_conf", type=float, default=None,
                        help="e.g. 0.75 → only low-confidence events (admin review queue)")
    parser.add_argument("--codec", choices=("h264", "mp4v"), default="h264",
                        help="Clip encoder: h264 via ffmpeg when available, or OpenCV mp4v")
    parser.add_argument("--all_frames", default=None,
                        help="debug_all_frames.json — enables the locked highlight box "
                             "on the event player (omit to clip without boxes)")
    parser.add_argument("--zoom", action="store_true",
                        help="crop tight on the event player (needs --all_frames)")
    parser.add_argument("--max_clips", type=int, default=200,
                        help="cap the number of clips (per invocation)")
    parser.add_argument("--roster", default=None,
                        help="roster_input.json — only clip events for players on the roster")
    parser.add_argument("--require_box", action="store_true",
                        help="skip events whose player is not tracked at the event frame")
    parser.add_argument("--no_ball", action="store_true",
                        help="do NOT overlay the tracked ball + ownership line (needs --all_frames)")
    parser.add_argument("--include_incomplete_passes", action="store_true",
                        help="also clip intercepted / unresolved-receiver passes (off by default)")
    parser.add_argument("--all_boxes", action="store_true",
                        help="box EVERY tracked player (cyan T{id} #{jersey}), not just the "
                             "actor (amber); reads track_jersey.json from --output_dir for numbers")
    args = parser.parse_args()

    valid_players = None
    if args.roster:
        _rj = json.load(open(args.roster))
        valid_players = set()
        for _t in (_rj.get("teams") or {}).values():
            for _n in (_t.get("roster") or []):
                try:
                    valid_players.add(int(_n))
                except (TypeError, ValueError):
                    pass
        print(f"[Clipper] roster filter: {len(valid_players)} valid jersey numbers")

    with open(args.events) as f:
        event_list = json.load(f)
    frame_boxes = None
    frame_balls = None
    frame_goals = None
    frame_gks = None
    track_jersey = None
    if args.all_frames:
        with open(args.all_frames) as f:
            _all = json.load(f)
        frame_boxes = _build_frame_boxes(_all, args.vid_stride)
        if not args.no_ball:
            frame_balls = _build_frame_balls(_all, args.vid_stride)
        # goal (cls 33) + goalkeeper marker (cls 34) boxes for shot-clip context
        frame_goals = _build_frame_class_boxes(_all, args.vid_stride, 33)
        frame_gks = _build_frame_class_boxes(_all, args.vid_stride, 34)
    if args.all_boxes:
        track_jersey = _load_track_jersey(args.output_dir)
        print(f"[Clipper] all-boxes on: {sum(1 for v in track_jersey.values() if v)} "
              f"tracks carry a jersey number")
    clip_events(args.video, event_list, args.output_dir,
                vid_stride=args.vid_stride, pad_s=args.pad_s,
                pad_before_s=args.pad_before_s, pad_after_s=args.pad_after_s,
                event_types=tuple(t.strip() for t in args.types.split(",") if t.strip()),
                min_conf=args.min_conf, max_conf=args.max_conf, codec=args.codec,
                frame_boxes=frame_boxes, zoom=args.zoom, max_clips=args.max_clips,
                valid_players=valid_players, require_box=args.require_box,
                frame_balls=frame_balls, clean_passes=not args.include_incomplete_passes,
                frame_goals=frame_goals, frame_gks=frame_gks,
                track_jersey=track_jersey, draw_all_boxes=args.all_boxes)
