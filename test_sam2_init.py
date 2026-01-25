import sys
import os

# Set paths
sys.path.append("/home/ubuntu/football/sam2")

import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor

def test_init():
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "/home/ubuntu/football/sam2/checkpoints/sam2.1_hiera_large.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Testing SAM2 init with config: {config}")
    try:
        predictor = build_sam2_video_predictor(config, checkpoint, device=device)
        print("✅ SAM2 Predictor built successfully!")
        return predictor
    except Exception as e:
        print(f"❌ Failed to build SAM2 Predictor: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_init()
