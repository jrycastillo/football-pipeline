import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# Standard Pitch Dimensions (UEFA/FIFA approx)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

class PitchManager:
    def __init__(self, model_path, device=None):
        self.model = None
        if model_path:
            try:
                self.model = YOLO(model_path)
                print(f"[pitch] Loaded Pitch Model from {model_path}")
            except Exception as e:
                print(f"[pitch] Failed to load model: {e}")
        
        self.device = device
        # Fallback Homography (Center Cam approx)
        # Maps 1920x1080 -> 105x68 (Naive)
        self.H_default = np.array([
            [0.055, 0.0, 0.0],
            [0.0, 0.063, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
    def predict(self, frame):
        """
        Run inference on frame.
        Returns: keypoints (Nx2), homography_matrix (3x3)
        """
        if self.model is None:
            return None, self.H_default
            
        # Inference
        results = self.model.predict(frame, verbose=False, device=self.device)
        if not results:
            return None, self.H_default
            
        r = results[0]
        if not hasattr(r, "keypoints") or r.keypoints is None:
            return None, self.H_default
            
        # Extract Keypoints
        # Shape: (1, 32, 2) or (1, 32, 3) with conf
        kps = r.keypoints.xy.cpu().numpy()[0] # (32, 2)
        
        # TODO: SolvenPnP or FindHomography if Schema is known.
        # Since Schema is unknown for these 32 points, we currently
        # cannot generate a High-Precision Homography.
        # We will return the keypoints for visualization and use Default H.
        
        return kps, self.H_default

    def get_homography(self):
        return self.H_default
