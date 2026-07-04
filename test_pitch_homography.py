#!/usr/bin/env python3
"""
Unit test for vision/pitch_homography.py (pure math, no model, no GPU).

Builds a known meters->pixels homography, projects the 32 pitch
landmarks into synthetic "detected" keypoints, and checks that
PitchHomographyEstimator.fit_from_keypoints recovers the inverse
(px->meters) mapping under full visibility, partial visibility,
too-few-points reuse, and RANSAC outlier rejection.

Run: python3 test_pitch_homography.py
"""

import os
import sys

# Headless environments: avoid Qt plugin issues before importing cv2
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import cv2  # noqa: F401
except Exception as e:
    print(f"SKIPPED-ENV: cv2 not importable here ({e}) — run on the worker")
    sys.exit(0)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.pitch_homography import PITCH_VERTICES_M, PitchHomographyEstimator


def project(H, pts):
    """Apply a 3x3 homography to (N, 2) points."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    homog = np.hstack([pts, ones]) @ H.T
    return homog[:, :2] / homog[:, 2:3]


def make_ground_truth():
    """
    A plausible broadcast meters->pixels matrix: scale + shift with
    slight perspective. Keeps all 32 landmarks at positive pixel
    coordinates so none get dropped by the in-frame filter.
    """
    H_true = np.array([
        [18.0, 2.0, 60.0],
        [1.0, 15.0, 40.0],
        [0.001, 0.0005, 1.0],
    ])
    pitch_m = np.array(PITCH_VERTICES_M, dtype=np.float64)
    kps_px = project(H_true, pitch_m).astype(np.float32)
    assert np.all(kps_px > 0), "ground-truth setup must keep keypoints in-frame"
    return H_true, kps_px


def max_roundtrip_error_m(estimator, H_true, rng, n=10):
    """
    Project n random pitch points meters->px through H_true, then
    px->meters through the fitted H; return the worst error in meters.
    """
    pts_m = np.column_stack([
        rng.uniform(5.0, 100.0, n),
        rng.uniform(5.0, 63.0, n),
    ])
    pts_px = project(H_true, pts_m)
    back_m = project(estimator.H, pts_px)
    return float(np.max(np.linalg.norm(back_m - pts_m, axis=1)))


def main():
    rng = np.random.RandomState(42)
    H_true, kps_px = make_ground_truth()

    # Case 1: all 32 keypoints confident
    est = PitchHomographyEstimator(model_path=None)
    conf = np.full(32, 0.9, dtype=np.float32)
    fitted = est.fit_from_keypoints(kps_px, conf)
    assert fitted, "full-visibility fit should succeed"
    assert est.is_ready
    err = max_roundtrip_error_m(est, H_true, rng)
    assert err < 0.2, f"full-visibility error {err:.4f} m exceeds 0.2 m"
    print(f"PASS case 1: full visibility (max error {err:.4f} m)")

    # Case 2: partial pitch — only 8 landmarks confident
    est2 = PitchHomographyEstimator(model_path=None)
    conf_partial = np.full(32, 0.1, dtype=np.float32)
    visible = [0, 5, 8, 13, 16, 21, 24, 29]
    conf_partial[visible] = 0.9
    fitted = est2.fit_from_keypoints(kps_px, conf_partial)
    assert fitted, "8-keypoint fit should succeed"
    err = max_roundtrip_error_m(est2, H_true, rng)
    assert err < 0.2, f"partial-visibility error {err:.4f} m exceeds 0.2 m"
    print(f"PASS case 2: partial visibility, 8 keypoints (max error {err:.4f} m)")

    # Case 3: too few keypoints — previous H must be retained
    probe_px = np.array([[900.0, 500.0]])
    before = project(est.H, probe_px)
    conf_few = np.full(32, 0.1, dtype=np.float32)
    conf_few[[0, 13, 29]] = 0.9
    reused_before = est.frames_reused
    fitted = est.fit_from_keypoints(kps_px, conf_few)
    assert not fitted, "3-keypoint fit must return False"
    assert est.frames_reused == reused_before + 1
    assert est.H is not None, "previous H must be kept"
    after = project(est.H, probe_px)
    assert np.allclose(before, after), "H changed despite failed fit"
    print("PASS case 3: too few keypoints reuses previous H")

    # Case 4: RANSAC rejects corrupted keypoints
    est4 = PitchHomographyEstimator(model_path=None)
    conf_out = np.full(32, 0.1, dtype=np.float32)
    good = [0, 1, 4, 5, 8, 9, 12, 13, 16, 30, 24, 29]
    bad = [21, 25]
    conf_out[good + bad] = 0.9
    kps_corrupt = kps_px.copy()
    kps_corrupt[bad] += 300.0
    fitted = est4.fit_from_keypoints(kps_corrupt, conf_out)
    assert fitted, "fit with outliers should still succeed"
    err = max_roundtrip_error_m(est4, H_true, rng)
    assert err < 0.3, f"outlier-robust error {err:.4f} m exceeds 0.3 m"
    print(f"PASS case 4: outlier robustness (max error {err:.4f} m)")

    print("ALL PASS")


if __name__ == "__main__":
    main()
