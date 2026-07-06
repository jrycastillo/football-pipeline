"""
Pitch keypoint homography estimation.

Maps camera pixels -> pitch coordinates in METERS on the standard
105.0 x 68.0 m pitch used everywhere else in this repo (see
vision/camera.py): origin (0, 0) at the top-left corner, goals on
the lines x = 0 and x = 105, y in [0, 68].

Every frame, a YOLO keypoint model detects up to 32 named pitch
landmarks in the camera view. The real pitch position of each
landmark is known (PITCH_VERTICES_M below), so the confidently
detected ones (conf >= 0.50, minimum 4) feed cv2.findHomography
(RANSAC) to produce a 3x3 matrix H: camera px -> pitch meters.
That H is a drop-in for Camera(homography_matrix=H).

Partial-pitch handling: broadcast cameras never show the whole
pitch. Only the landmarks visible in the current frame are used;
when a frame has fewer than 4 confident landmarks the LAST VALID
homography is kept, so downstream projection never dies mid-video.

The 32-landmark schema and its index order come from the pitch
keypoint model (trained separately on a 32-named-landmark pitch
annotation set; same vertex layout as roboflow/sports
SoccerPitchConfiguration). The order of PITCH_VERTICES_M MUST
match the model's keypoint output indices exactly.
"""

import os

import cv2
import numpy as np

# Standard pitch dimensions (meters) — same convention as vision/camera.py
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

PENALTY_BOX_WIDTH = 41.0
PENALTY_BOX_LENGTH = 20.15
GOAL_BOX_WIDTH = 18.32
GOAL_BOX_LENGTH = 5.5
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0

# The 32 pitch keypoints in model index order.
# Index -> (x, y) in METERS.
# Left half (x = 0 side), then middle, then right half, then the two
# centre-circle side points — exactly the vertex layout of
# roboflow/sports SoccerPitchConfiguration.
PITCH_VERTICES_M = [

    # Left goal line (x = 0), top -> bottom
    (0.0, 0.0),                                                    # 0  left-top corner
    (0.0, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),                  # 1  left penalty box top
    (0.0, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),                     # 2  left goal box top
    (0.0, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),                     # 3  left goal box bottom
    (0.0, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),                  # 4  left penalty box bottom
    (0.0, PITCH_WIDTH),                                            # 5  left-bottom corner

    # Left goal box front edge
    (GOAL_BOX_LENGTH, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),         # 6  left goal box front-top
    (GOAL_BOX_LENGTH, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),         # 7  left goal box front-bottom

    # Left penalty spot
    (PENALTY_SPOT_DISTANCE, PITCH_WIDTH / 2),                      # 8  left penalty spot

    # Left penalty box front edge, top -> bottom
    (PENALTY_BOX_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),   # 9  left pen box front-top
    (PENALTY_BOX_LENGTH, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),      # 10 left pen box front (goal-box top line)
    (PENALTY_BOX_LENGTH, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),      # 11 left pen box front (goal-box bottom line)
    (PENALTY_BOX_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),   # 12 left pen box front-bottom

    # Halfway line, top -> bottom
    (PITCH_LENGTH / 2, 0.0),                                       # 13 halfway top
    (PITCH_LENGTH / 2, PITCH_WIDTH / 2 - CENTRE_CIRCLE_RADIUS),    # 14 centre circle top
    (PITCH_LENGTH / 2, PITCH_WIDTH / 2 + CENTRE_CIRCLE_RADIUS),    # 15 centre circle bottom
    (PITCH_LENGTH / 2, PITCH_WIDTH),                               # 16 halfway bottom

    # Right penalty box front edge, top -> bottom
    (PITCH_LENGTH - PENALTY_BOX_LENGTH,
     (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),                       # 17 right pen box front-top
    (PITCH_LENGTH - PENALTY_BOX_LENGTH,
     (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),                          # 18 right pen box front (goal-box top line)
    (PITCH_LENGTH - PENALTY_BOX_LENGTH,
     (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),                          # 19 right pen box front (goal-box bottom line)
    (PITCH_LENGTH - PENALTY_BOX_LENGTH,
     (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),                       # 20 right pen box front-bottom

    # Right penalty spot
    (PITCH_LENGTH - PENALTY_SPOT_DISTANCE, PITCH_WIDTH / 2),       # 21 right penalty spot

    # Right goal box front edge
    (PITCH_LENGTH - GOAL_BOX_LENGTH,
     (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),                          # 22 right goal box front-top
    (PITCH_LENGTH - GOAL_BOX_LENGTH,
     (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),                          # 23 right goal box front-bottom

    # Right goal line (x = PITCH_LENGTH), top -> bottom
    (PITCH_LENGTH, 0.0),                                           # 24 right-top corner
    (PITCH_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),         # 25 right penalty box top
    (PITCH_LENGTH, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2),            # 26 right goal box top
    (PITCH_LENGTH, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),            # 27 right goal box bottom
    (PITCH_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),         # 28 right penalty box bottom
    (PITCH_LENGTH, PITCH_WIDTH),                                   # 29 right-bottom corner

    # Centre circle left / right points
    (PITCH_LENGTH / 2 - CENTRE_CIRCLE_RADIUS, PITCH_WIDTH / 2),    # 30 centre circle left
    (PITCH_LENGTH / 2 + CENTRE_CIRCLE_RADIUS, PITCH_WIDTH / 2),    # 31 centre circle right
]

assert len(PITCH_VERTICES_M) == 32, (
    f"Expected 32 pitch vertices, got {len(PITCH_VERTICES_M)}"
)


class PitchHomographyEstimator:
    """
    Fits a px -> meters homography from detected pitch keypoints.
    predict(frame) mirrors the old PitchManager interface
    (vision/pitch.py) so it can be swapped in directly.
    """

    def __init__(self, model_path="models/pitch_keypoints.pt", device=None,
                 conf_threshold=0.50, min_keypoints=4, ransac_reproj_px=8.0):
        self.device = device
        self.conf_threshold = conf_threshold
        self.min_keypoints = min_keypoints
        self.ransac_reproj_px = ransac_reproj_px

        # Real pitch coordinates (meters) of all 32 landmarks
        self.pitch_points = np.array(PITCH_VERTICES_M, dtype=np.float32)

        # Last valid homography (camera px -> pitch meters)
        self.H = None

        # Counters for logging
        self.frames_seen = 0
        self.frames_fitted = 0
        self.frames_reused = 0
        self.last_num_points = 0

        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                # Lazy import so the module works without ultralytics
                # (fit_from_keypoints is pure math and needs no model)
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[pitch_homography] Loaded pitch keypoint model from {model_path}")
            except Exception as e:
                print(f"[pitch_homography] Failed to load model: {e}")
        elif model_path:
            print(f"[pitch_homography] No pitch keypoint model at {model_path} — homography disabled")

    def fit_from_keypoints(self, kps_xy, kps_conf):
        """
        Update the homography from one frame's detected keypoints.

        kps_xy   : (32, 2) keypoint (x, y) in camera pixels
        kps_conf : (32,) per-keypoint confidence

        Returns True if a fresh homography was fitted, False if the
        previous matrix was kept (partial pitch / degenerate fit).
        """
        self.frames_seen += 1

        if kps_xy is None or kps_conf is None:
            self.frames_reused += 1
            return False

        kps_xy = np.asarray(kps_xy, dtype=np.float32)
        kps_conf = np.asarray(kps_conf, dtype=np.float32)

        if kps_xy.ndim != 2 or kps_xy.shape[0] != len(self.pitch_points):
            self.frames_reused += 1
            return False

        # Confident AND inside the frame — the model emits (0, 0)
        # for landmarks not visible in the current view
        mask = (
            (kps_conf >= self.conf_threshold)
            & (kps_xy[:, 0] > 0)
            & (kps_xy[:, 1] > 0)
        )
        num_points = int(mask.sum())
        self.last_num_points = num_points

        if num_points < self.min_keypoints:
            # Not enough landmarks this frame — keep the last valid H
            self.frames_reused += 1
            return False

        camera_pts = kps_xy[mask]
        pitch_pts = self.pitch_points[mask]

        try:
            H, _inliers = cv2.findHomography(
                camera_pts, pitch_pts, cv2.RANSAC, self.ransac_reproj_px
            )
            if H is None:
                self.frames_reused += 1
                return False

            self.H = H
            self.frames_fitted += 1
            return True

        except Exception as e:
            print(f"[pitch_homography] Fit error: {e}")
            self.frames_reused += 1
            return False

    def predict(self, frame):
        """
        Run keypoint inference on a frame and refresh the homography.
        Returns (keypoints_xy_or_None, H) — same shape as the old
        PitchManager.predict. Never raises.
        """
        if self.model is None:
            return None, self.H

        try:
            # Detection conf 0.05: on broadcast footage the pitch object often
            # scores low (median det conf ~0.1 on HB), but every frame it IS
            # detected on yields >=4 good keypoints — measured fit-capable rate
            # 31% at 0.05 vs 20% at 0.30. RANSAC + the keypoint conf gate
            # protect against garbage fits.
            results = self.model.predict(
                frame, conf=0.05, imgsz=640, device=self.device, verbose=False
            )
            if not results:
                self.fit_from_keypoints(None, None)
                return None, self.H

            r = results[0]
            kp = getattr(r, "keypoints", None)
            if kp is None or kp.xy is None:
                self.fit_from_keypoints(None, None)
                return None, self.H

            xy = kp.xy.cpu().numpy()  # (n_det, 32, 2)
            if xy.shape[0] == 0:
                self.fit_from_keypoints(None, None)
                return None, self.H

            if kp.conf is not None:
                conf = kp.conf.cpu().numpy()  # (n_det, 32)
            else:
                conf = np.ones(xy.shape[:2], dtype=np.float32)

            # Keep only the highest-confidence detection of the pitch
            best = 0
            if xy.shape[0] > 1 and r.boxes is not None and len(r.boxes) == xy.shape[0]:
                best = int(r.boxes.conf.argmax())

            kps_xy = xy[best]
            self.fit_from_keypoints(kps_xy, conf[best])
            return kps_xy, self.H

        except Exception as e:
            print(f"[pitch_homography] Inference error: {e}")
            return None, self.H

    @property
    def is_ready(self):
        """True once at least one homography has been fitted."""
        return self.H is not None

    def fit_stats(self):
        """Counters for pipeline logging."""
        return {
            "frames_seen": self.frames_seen,
            "frames_fitted": self.frames_fitted,
            "frames_reused": self.frames_reused,
            "last_num_points": self.last_num_points,
            "is_ready": self.is_ready,
        }
