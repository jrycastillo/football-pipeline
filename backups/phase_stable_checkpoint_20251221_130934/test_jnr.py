import cv2
import sys
import os
import json

# Add path
sys.path.append(os.getcwd())

from vision.identity import get_jnr_service
import yaml
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)
    
# Mock CONFIG for identity module
import builtins
builtins.CONFIG = CONFIG

from vision.identity import get_jnr_service
def test_jnr():
    print("Loading Service...")
    service = get_jnr_service()
    if not service:
        print("Failed to load service.")
        return

    print("Service Loaded. Creating dummy image...")
    # Create a black image with text "10"
    try:
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(img, "10", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    except:
        print("CV2 Error")
        return
    
    print("Running Prediction...")
    results = service.predict_batch([img])
    print("Results:", results)

if __name__ == "__main__":
    test_jnr()
