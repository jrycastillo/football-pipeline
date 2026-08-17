import numpy as np
import yaml

# Load Config
try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    CLASS_BALL = CONFIG["classes"].get("ball", 0)
except:
    CLASS_BALL = 0

def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

class BallTracker:
    def __init__(self, frame_width=1920, trajectory_select=False):
        self.tracks = {} # frame_idx -> (x, y)
        self.raw_detections = {} # frame_idx -> box
        self.max_gap = 30 # Round 2 fix: 50→30 (50 frames at VID_STRIDE=5 = 10s, too long for linear interp)
        # R20: trajectory-consistent ball selection. The ball model emits 2-6
        # spurious "balls" per frame (heads / boots / noise) on many clips, and
        # highest-confidence alone often picks one of the false balls, wrecking
        # the track (and blocking goal detection). We instead prefer the detection
        # nearest the position predicted from recent motion, gated by a plausible
        # per-frame travel distance.
        self.frame_width = frame_width or 1920
        self.trajectory_select = trajectory_select
        self._gate_frac = 0.18            # max plausible ball travel / frame (of width)
        self._reappear_conf = 0.45        # conf needed to accept a ball outside the gate
        self._hist = []                   # recent accepted (frame_idx, (x, y))

    def update(self, frame_idx, boxes):
        """Update ball tracker with detections from a frame, choosing the
        trajectory-consistent detection rather than merely the highest-conf one."""
        balls = [b for b in boxes if b["cls"] in [0, 32]]
        if not balls:
            return

        cands = [(bbox_center(b["xyxy"]), float(b.get("conf", 0.0)), b) for b in balls]

        if not self.trajectory_select or not self._hist:
            best = max(cands, key=lambda x: x[1])          # highest confidence
            self._commit(frame_idx, best[0], best[2])
            return

        # Predict from the last one/two accepted positions.
        pf, ppos = self._hist[-1]
        if len(self._hist) >= 2:
            pf2, ppos2 = self._hist[-2]
            dt = max(1, pf - pf2)
            vx = (ppos[0] - ppos2[0]) / dt
            vy = (ppos[1] - ppos2[1]) / dt
        else:
            vx = vy = 0.0
        gap = max(1, frame_idx - pf)
        px, py = ppos[0] + vx * gap, ppos[1] + vy * gap
        gate = self.frame_width * self._gate_frac * gap

        best = None
        for c, conf, b in cands:
            d = ((c[0] - px) ** 2 + (c[1] - py) ** 2) ** 0.5
            if d <= gate:
                score = d - 30.0 * conf                    # small pull toward confident boxes
                if best is None or score < best[0]:
                    best = (score, c, b, conf)
        if best is not None:
            self._commit(frame_idx, best[1], best[2])
            return

        # Nothing on the predicted path: either the ball genuinely jumped (a
        # confident re-appearance) or these are only spurious balls. Accept a
        # high-confidence detection; otherwise SKIP this frame so the ball reads
        # as gone (linear interp bridges short gaps; a real disappearance near
        # the goal now survives as the goal signal instead of being masked).
        top = max(cands, key=lambda x: x[1])
        if top[1] >= self._reappear_conf:
            self._hist.clear()                             # reset track after a jump
            self._commit(frame_idx, top[0], top[2])

    def _commit(self, frame_idx, pos, box):
        self.raw_detections[frame_idx] = box
        self.tracks[frame_idx] = pos
        self._hist.append((frame_idx, pos))
        if len(self._hist) > 6:
            self._hist.pop(0)

    def interpolate(self, total_frames):
        """
        Fill gaps in ball tracking using linear interpolation.
        Returns a list of (x, y) or None for each frame.
        """
        final_track = [None] * total_frames
        
        # Fill knowns
        for idx, pos in self.tracks.items():
            if idx < total_frames:
                final_track[idx] = pos
                
        # Interpolate
        # Use pandas-like ffill/bfill logic or simple loop
        # For strict physics, maybe Kalman, but Linear is fine for stats
        
        last_idx = -1
        for i in range(total_frames):
            if final_track[i] is not None:
                if last_idx != -1 and (i - last_idx) <= self.max_gap:
                    # Interpolate
                    start_pos = final_track[last_idx]
                    end_pos = final_track[i]
                    steps = i - last_idx
                    dx = (end_pos[0] - start_pos[0]) / steps
                    dy = (end_pos[1] - start_pos[1]) / steps
                    
                    for j in range(1, steps):
                        final_track[last_idx + j] = (start_pos[0] + dx * j, start_pos[1] + dy * j)
                last_idx = i
                
        return final_track
