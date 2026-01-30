import os
import sys
import importlib.util
import torch
from utils.device_utils import get_device_name

def check_import(module_name):
    if importlib.util.find_spec(module_name) is None:
        print(f"❌ Missing dependency: {module_name}")
        return False
    return True

def check_file(path):
    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        return False
    print(f"✅ Found file: {path}")
    return True

def main():
    print("=== Environment Check ===")
    
    # 1. Check Device
    print(f"Device: {get_device_name()}")
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) is available.")
    else:
        print("⚠️ MPS not available. Using CPU or CUDA (if linux).")

    # 2. Check Dependencies
    deps = ["torch", "torchvision", "ultralytics", "transformers", "cv2", "PIL", "yaml", "sklearn", "scipy"]
    all_deps = True
    for d in deps:
        if not check_import(d):
            all_deps = False
    
    if all_deps:
        print("✅ All Python dependencies installed.")

    # 3. Check Models
    print("\n=== Model Check ===")
    # Updated model list matching Google Drive
    models = [
        "models/yolo_player.pt",
        "models/yolo_pitch.pt",
        "models/resnet34_rgb_jnr.pt",
        "models/yolo_ball.pt"
    ]
    
    found_models = True
    for m in models:
        if not check_file(m):
            found_models = False
            
    if not found_models:
        print("\n⚠️  MISSING MODELS DETECTED")
        print("Please ensure the following models are in 'models/' directory:")
        print("1. yolo_player.pt")
        print("2. yolo_pitch.pt")
        print("3. resnet34_rgb_jnr.pt")
        print("4. yolo_ball.pt")
    else:
        print("✅ All model files found.")

if __name__ == "__main__":
    main()
