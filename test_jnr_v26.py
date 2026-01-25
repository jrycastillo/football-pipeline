import cv2
import numpy as np
import torch
from PIL import Image
import json
import os
import sys

# Path setup
sys.path.append(os.getcwd())

# Mock CONFIG
import yaml
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

import builtins
builtins.CONFIG = CONFIG
builtins.log = lambda x: print(f"LOG: {x}")

# Import from pipeline_consolidated
from pipeline_consolidated import JNRService, Image

def test_v26():
    print("🚀 Initializing JNRService v26...")
    service = JNRService()
    
    print("🖼️ Creating test image (Jersey #24)...")
    # Base image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, "24", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    print("🧪 Testing Preprocessing...")
    enhanced = service._preprocess_enhanced(img)
    if enhanced is not None:
        print(f"✅ Preprocessing OK. Shape: {enhanced.shape}")
        cv2.imwrite("test_enhanced.png", enhanced)
    
    print("🤖 Running Multi-Scale Inference...")
    # Test with no references
    results = service.predict_batch([img])
    print(f"Results (Single Image, No ICL): {results}")
    
    print("🧠 Testing ICL (In-Context Learning)...")
    # Mock references
    ref_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(ref_img, "24", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_pil = Image.fromarray(ref_rgb)
    
    references = {"24": [ref_pil]}
    results_icl = service.predict_batch([img], reference_crops=references)
    print(f"Results (ICL enabled): {results_icl}")

if __name__ == "__main__":
    test_v26()
