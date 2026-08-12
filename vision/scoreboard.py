"""Scoreboard-based goal detection.

Reads the on-screen score from broadcast/semi-pro footage and emits a goal event
on every score change — the most reliable goal signal when a scoreboard exists.
Uses the PARSeq digit recognizer already shipped for jersey numbers, so there is
NO new dependency (no tesseract/easyocr needed).

Scope: footage that HAS a scoreboard. Amateur footage without one (e.g. fixed
single-camera matches) has no score to read and must fall back to the vision
approach. Region calibration is per-broadcast-layout (see `home_box`/`away_box`).

Validation (Bundesliga test match, truth 0-0->1-0->1-1->1-2->2-2): reads every
NON-ZERO digit correctly (1-1, 1-2, 2-2 exact). KNOWN GAP: the jersey-tuned
PARSeq weights confuse "0" with "8" (jersey numbers are never a solo 0, so the
model never learned it), so scores involving 0 misread. Fix is a digit-specific
recognizer (tesseract/easyocr, or a small scoreboard-font template matcher, or a
PARSeq head fine-tuned to include 0) — swap it in behind `read_digit`; the rest
of this module (region crop, temporal tracking, score-change -> goal) is done.

Typical use:
    sb = ScoreboardReader(home_box=(205,78,258,142), away_box=(278,78,342,142))
    timeline = sb.scan("match.mp4", sample_every_s=2.0)
    goals = ScoreboardReader.detect_goals(timeline)   # [(time_s, team, (h,a)), ...]
"""
import os
import sys
import string

import cv2
import torch
from PIL import Image


class ScoreboardReader:
    def __init__(self, weights_path="models/parseq_local_v6.pt", device=None,
                 home_box=None, away_box=None, min_conf=0.30):
        # home_box / away_box: (x1, y1, x2, y2) pixel regions of each score digit
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
