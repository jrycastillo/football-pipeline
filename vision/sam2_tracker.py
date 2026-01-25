import os
import sys
import torch
import numpy as np
import logging

# Ensure sam2 package is available
SAM2_PATH = "/home/ubuntu/football/sam2"
if SAM2_PATH not in sys.path:
    sys.path.append(SAM2_PATH)

from sam2.build_sam import build_sam2_video_predictor

class SAM2Tracker:
    """
    SAM2-powered Video Object Tracking.
    Handles memory-backed temporal propagation for robust player tracking.
    """
    def __init__(self, model_type="large", device="cuda"):
        self.device = torch.device(device)
        
        # Configuration mapping
        configs = {
            "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml"
        }
        checkpoints = {
            "large": "/home/ubuntu/football/sam2/checkpoints/sam2.1_hiera_large.pt",
            "base": "/home/ubuntu/football/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
            "small": "/home/ubuntu/football/sam2/checkpoints/sam2.1_hiera_small.pt",
            "tiny": "/home/ubuntu/football/sam2/checkpoints/sam2.1_hiera_tiny.pt"
        }
        
        if model_type not in configs:
            raise ValueError(f"Unsupported SAM2 model type: {model_type}")
            
        self.config = configs[model_type]
        self.checkpoint = checkpoints[model_type]
        
        logging.info(f"🚀 Initializing SAM2 ({model_type}) Predictor...")
        self.predictor = build_sam2_video_predictor(self.config, self.checkpoint, device=device)
        self.inference_state = None
        self.obj_ids = []

    def init_video(self, video_path):
        """
        Initialize tracking state for a new video.
        video_path: Path to MP4 or folder of frames.
        """
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
        
        self.inference_state = self.predictor.init_state(video_path=video_path)
        self.obj_ids = []
        logging.info(f"📽️ SAM2 state initialized for video: {video_path}")

    def add_player_prompts(self, frame_idx, detections):
        """
        Seed tracking with detection boxes.
        detections: list of {'bbox': [x1, y1, x2, y2], 'id': tid}
        """
        for det in detections:
            tid = det['id']
            # SAM2 expects [x1, y1, x2, y2]
            box = np.array(det['bbox'], dtype=np.float32)
            
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=tid,
                box=box
            )
            if tid not in self.obj_ids:
                self.obj_ids.append(tid)
        
        logging.info(f"📍 Added {len(detections)} prompts to SAM2 at Frame {frame_idx}")

    def propagate(self, start_frame_idx=0, reverse=False):
        """
        Generator yielding (frame_idx, object_ids, mask_logits)
        """
        return self.predictor.propagate_in_video(
            self.inference_state, 
            start_frame_idx=start_frame_idx,
            reverse=reverse
        )

    def get_bboxes_from_masks(self, out_obj_ids, out_mask_logits):
        """
        Convert mask logits to bounding boxes.
        Returns: {track_id: [x1, y1, x2, y2]}
        """
        # out_mask_logits: [N, 1, H, W]
        # Binarize masks
        masks = (out_mask_logits > 0.0)
        
        results = {}
        for i, tid in enumerate(out_obj_ids):
            mask = masks[i, 0]
            if not torch.any(mask):
                continue
            
            # Efficient bbox extraction on GPU
            y, x = torch.where(mask)
            x1, y1 = x.min().item(), y.min().item()
            x2, y2 = x.max().item(), y.max().item()
            
            results[tid] = [x1, y1, x2, y2]
            
        return results

    def reset(self):
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
            self.inference_state = None
            self.obj_ids = []
