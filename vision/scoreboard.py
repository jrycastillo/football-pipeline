"""Scoreboard-based goal detection.

Reads the on-screen score from broadcast footage and emits a goal on every score
change — the only unambiguous goal signal on a single camera. The vision methods
(ball trajectory / ball vanishing into the net) all depend on seeing the moment
broadcast handles worst, and measured 0-1 goals per match, all low confidence.

Scope: footage that HAS a scorebug. Amateur footage without one has no score to
read and keeps the vision path.

    PRIMARY — `GlyphScoreboardReader` (use this).
        Detects a score change as a change in the *shape* of the rendered digit,
        so it never has to decide which digit it is. No OCR, no model, no torch:
        just OpenCV. Validated on the full Hamburg-Bayern broadcast (truth 2-2):
        4 goals, 4 correct, 0 false positives, running score 1-0/1-1/1-2/2-2.

            goals = GlyphScoreboardReader().detect_goals("match.mp4")
            # [(time_s, 'home'|'away', (home, away)), ...]

    LEGACY — `ScoreboardReader` (PARSeq digit OCR).
        Kept for reference/diagnostics. It reads every NON-ZERO digit correctly
        but confuses "0" with "8": the weights are jersey-tuned and a jersey is
        never a solo 0, so the model never learned it — which corrupts any score
        involving 0. That gap is what motivated the glyph reader above; prefer it.
"""
import os
import sys
import string
import collections

import cv2
import numpy as np


class GlyphScoreboardReader:
    """Goal detection by scorebug GLYPH CHANGE — no OCR, no model, no torch.

    A goal is a score change, and a score change is a *change of the rendered
    digit*. Comparing the digit's shape to its own recent history detects that
    change without ever deciding WHICH digit it is — which sidesteps the 0-vs-8
    confusion that made the PARSeq path unusable, and works on any broadcast
    font without retraining.

    Robustness comes from three stages, each removing a specific failure:
      1. PANEL LOCK  — the score panels are found once and their position fixed
         (median over samples). Per-frame contour search would drift onto a red
         advertising board or a blue kit whenever the scorebug is hidden, and
         read a garbage "digit" from it.
      2. MODAL SMOOTHING — each sample is replaced by the most common label in a
         +/-45 s window, deleting isolated misreads (replays, cut-aways,
         compression artefacts) that would otherwise look like score changes.
      3. PERSISTENCE — a change counts only if the new glyph then holds for
         ~2 minutes. A real score stands for the rest of the match; noise does not.

    Validated on the full Hamburg-Bayern broadcast (truth 2-2, goals at 33:57,
    41:36, 51:51, 58:36): 4 goals detected, 4 correct, 0 false positives.
    """

    # scorebug search window (x, y) — the overlay sits top-left on this layout
    SEARCH = (100, 500, 40, 200)          # x1, x2, y1, y2

    def __init__(self, sample_every_s=3.0, smooth_win_s=45.0, persist_s=120.0,
                 match_thr=0.80):
        self.sample_every_s = sample_every_s
        self.smooth_win_s = smooth_win_s
        self.persist_s = persist_s
        self.match_thr = match_thr
        self.home_box = None
        self.away_box = None

    # ---------- panel location ----------
    @classmethod
    def _find_panels(cls, frame):
        """Locate the coloured home/away panels in one frame."""
        x1, x2, y1, y2 = cls.SEARCH
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return {}
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        masks = {
            "home": cv2.inRange(hsv, (100, 120, 60), (130, 255, 255)),          # blue
            "away": (cv2.inRange(hsv, (0, 120, 60), (10, 255, 255))
                     | cv2.inRange(hsv, (170, 120, 60), (180, 255, 255))),      # red
        }
        out = {}
        for side, m in masks.items():
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 1500 or w < 40 or h < 50:
                continue
            out[side] = (x + x1, y + y1, x + x1 + w, y + y1 + h)
        return out

    def calibrate(self, cap, probes=40):
        """Fix the panel boxes from the median of several frames (stage 1)."""
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n <= 0:
            return False
        found = {"home": [], "away": []}
        for k in range(probes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / probes))
            ok, f = cap.read()
            if not ok:
                continue
            for side, box in self._find_panels(f).items():
                found[side].append(box)
        for side in ("home", "away"):
            if len(found[side]) < max(3, probes // 4):
                return False
            arr = np.array(found[side])
            setattr(self, f"{side}_box", tuple(int(v) for v in np.median(arr, axis=0)))
        return self.home_box is not None and self.away_box is not None

    # ---------- glyph extraction ----------
    @staticmethod
    def _glyph(frame, box):
        """Binary mask of the digit inside a locked panel, or None when the
        scorebug isn't showing (panel colour absent => not our overlay)."""
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        panel = frame[y1:y2, x1:x2]
        if panel.size == 0:
            return None
        hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
        # the panel must still be a saturated colour block; if the overlay is
        # hidden this crop is pitch/crowd and must not be read as a digit
        if (hsv[..., 1] > 110).mean() < 0.35:
            return None
        ph = panel.shape[0]
        dig = panel[int(ph * 0.42):ph, :]                  # digit sits below the team name
        m = cv2.inRange(cv2.cvtColor(dig, cv2.COLOR_BGR2HSV), (0, 0, 150), (180, 90, 255))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(c)
        if gw < 6 or gh < 15:
            return None
        return cv2.resize(m[gy:gy + gh, gx:gx + gw], (24, 32), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _iou(a, b):
        inter = np.logical_and(a > 0, b > 0).sum()
        union = np.logical_or(a > 0, b > 0).sum()
        return inter / max(1, union)

    # ---------- pipeline ----------
    def scan(self, video_path):
        """Sample the match -> (times, home_glyphs, away_glyphs)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        if not self.calibrate(cap):
            cap.release()
            print("[Scoreboard] no scorebug found — skipping scoreboard goals")
            return None
        print(f"[Scoreboard] panels locked home={self.home_box} away={self.away_box}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(fps * self.sample_every_s))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ts, H, A = [], [], []
        fi = 0
        while True:
            if not cap.grab():
                break
            if fi % step == 0:
                ok, f = cap.retrieve()
                if ok:
                    ts.append(fi / fps)
                    H.append(self._glyph(f, self.home_box))
                    A.append(self._glyph(f, self.away_box))
            fi += 1
        cap.release()
        return np.array(ts), H, A

    def _labels(self, glyphs):
        """Assign each glyph a prototype id; merge near-duplicate prototypes."""
        protos, lab = [], np.full(len(glyphs), -1)
        for n, g in enumerate(glyphs):
            if g is None or g.sum() < 80:
                continue
            best, bi = 0.0, -1
            for k, p in enumerate(protos):
                s = self._iou(g, p)
                if s > best:
                    best, bi = s, k
            if best >= self.match_thr:
                lab[n] = bi
            else:
                protos.append(g)
                lab[n] = len(protos) - 1
        merge = {}
        for i in range(len(protos)):
            for j in range(i):
                if j in merge:
                    continue
                if self._iou(protos[i], protos[j]) >= self.match_thr - 0.06:
                    merge[i] = merge.get(j, j)
                    break
        return np.array([merge.get(x, x) if x >= 0 else -1 for x in lab])

    def _smooth(self, ts, lab):
        out = np.full(len(lab), -1)
        for n in range(len(lab)):
            m = (ts >= ts[n] - self.smooth_win_s) & (ts <= ts[n] + self.smooth_win_s)
            vals = [v for v in lab[m] if v >= 0]
            if vals:
                out[n] = collections.Counter(vals).most_common(1)[0][0]
        return out

    def _transitions(self, ts, sm):
        ev, cur = [], None
        for n in range(len(sm)):
            v = sm[n]
            if v < 0:
                continue
            if cur is None:
                cur = v
                continue
            if v != cur:
                m = (ts >= ts[n]) & (ts <= ts[n] + self.persist_s)
                fut = [x for x in sm[m] if x >= 0]
                if fut and sum(1 for x in fut if x == v) / len(fut) >= 0.85:
                    ev.append(float(ts[n]))
                    cur = v
        return ev

    def detect_goals(self, video_path):
        """[(time_s, 'home'|'away', (home_score, away_score))] for the match.

        Scores are derived from the change sequence (the scorebug reads 0-0 at
        kickoff and every change is +1), so the values never depend on OCR.
        """
        scanned = self.scan(video_path)
        if scanned is None:
            return []
        ts, H, A = scanned
        raw = []
        for side, G in (("home", H), ("away", A)):
            for t in self._transitions(ts, self._smooth(ts, self._labels(G))):
                raw.append((t, side))
        raw.sort()
        goals, h, a = [], 0, 0
        for t, side in raw:
            if side == "home":
                h += 1
            else:
                a += 1
            goals.append((t, side, (h, a)))
        print(f"[Scoreboard] {len(goals)} goal(s) from score changes: "
              + ", ".join(f"{int(t//60)}:{int(t%60):02d} {s}" for t, s, _ in goals))
        return goals


class ScoreboardReader:
    def __init__(self, weights_path="models/parseq_local_v6.pt", device=None,
                 home_box=None, away_box=None, min_conf=0.30):
        # home_box / away_box: (x1, y1, x2, y2) pixel regions of each score digit
        import torch                       # lazy: the glyph reader needs no torch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.home_box = home_box
        self.away_box = away_box
        self.min_conf = min_conf

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parseq_dir = os.path.join(base, "parseq")
        if parseq_dir not in sys.path:
            sys.path.insert(0, parseq_dir)
        from strhub.models.utils import create_model
        from strhub.data.module import SceneTextDataModule

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model = create_model("parseq", pretrained=False,
                             charset_train=string.digits, charset_test=string.digits,
                             max_label_length=2)
        model.load_state_dict(ckpt, strict=False)
        self.model = model.eval().to(self.device)
        self.transform = SceneTextDataModule.get_transform((32, 128))
        print(f"[Scoreboard] PARSeq digit reader loaded from {weights_path}")

    @staticmethod
    def _crop(frame, box):
        if box is None or frame is None:
            return None
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return frame[y1:y2, x1:x2]

    def read_digit(self, crop_bgr):
        """OCR one score digit. Returns (value:int|None, confidence:float).
        Unlike the jersey reader this allows 0 and does no torso crop."""
        import torch
        from PIL import Image
        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0
        # Scoreboard digits are bold and high-contrast on a coloured box; binarise
        # to plain dark-on-light so the reader sees a clean glyph (rescues "0",
        # which the jersey-tuned model otherwise misreads on the coloured ground).
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if bw.mean() > 127:      # keep the digit dark on a light background
            bw = 255 - bw
        crop_bgr = cv2.cvtColor(255 - bw, cv2.COLOR_GRAY2BGR)
        # pad to a squarer aspect so a single digit isn't over-stretched by the 32x128 resize
        h, w = crop_bgr.shape[:2]
        if w < h:
            pad = (h - w) // 2
            crop_bgr = cv2.copyMakeBorder(crop_bgr, 0, 0, pad, pad, cv2.BORDER_REPLICATE)
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        t = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
            probs = logits.softmax(-1)
            text, _ = self.model.tokenizer.decode(probs)
            conf = probs.max(-1).values.min(-1).values.item()
        digits = [ch for ch in text[0] if ch.isdigit()]
        # A single-digit score box often decodes as a doubled char ("1"->"11")
        # because the model emits up to 2 positions; collapse an exact repeat.
        if len(digits) == 2 and digits[0] == digits[1]:
            digits = digits[:1]
        s = "".join(digits)
        if s and conf >= self.min_conf:
            return int(s), conf
        return None, conf

    def read_score(self, frame):
        """Return (home_score, away_score) for one frame (None where unreadable)."""
        h, _ = self.read_digit(self._crop(frame, self.home_box))
        a, _ = self.read_digit(self._crop(frame, self.away_box))
        return h, a

    def scan(self, video_path, sample_every_s=2.0, max_s=None):
        """Sample the score across the video -> [(time_s, home, away)]."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(fps * sample_every_s))
        out = []
        for fi in range(0, n, step):
            if max_s is not None and fi / fps > max_s:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, f = cap.read()
            if not ok:
                continue
            h, a = self.read_score(f)
            out.append((fi / fps, h, a))
        cap.release()
        return out

    @staticmethod
    def detect_goals(timeline, min_repeat=2):
        """A score change confirmed by `min_repeat` consecutive equal reads = a goal.
        Returns [(time_s, 'home'|'away', (home,away))]. Ignores decreases and
        multi-jumps (OCR noise)."""
        goals = []
        stable = None
        cand = None
        run = 0
        for (ts, h, a) in timeline:
            if h is None or a is None:
                continue
            cur = (h, a)
            if cur == cand:
                run += 1
            else:
                cand, run = cur, 1
            if run >= min_repeat and cur != stable:
                if stable is not None:
                    dh, da = cur[0] - stable[0], cur[1] - stable[1]
                    if dh == 1 and da == 0:
                        goals.append((ts, "home", cur))
                    elif da == 1 and dh == 0:
                        goals.append((ts, "away", cur))
                stable = cur
        return goals


if __name__ == "__main__":
    # Validation on the Bundesliga test match (known truth):
    #   0-0 -> 1-0 -> 1-1 -> 1-2 -> 2-2  (4 goals, final 2-2)
    import numpy as np
    VID = os.path.expanduser("~/work/football/data/videos/test_gdrive.mp4")
    KNOWN = [(1505, "0-0"), (2104, "1-0"), (2600, "1-1"), (3200, "1-2"), (3750, "2-2")]
    sb = ScoreboardReader(home_box=(205, 78, 258, 142), away_box=(278, 78, 342, 142))
    cap = cv2.VideoCapture(VID)
    print(f"{'src_s':>6} {'expected':>9} {'read':>6}")
    for sec, exp in KNOWN:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * 25))
        ok, f = cap.read()
        h, a = sb.read_score(f)
        print(f"{sec:>6} {exp:>9} {str(h)+'-'+str(a):>6}")
    cap.release()
