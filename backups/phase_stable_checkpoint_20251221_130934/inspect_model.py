from ultralytics import YOLO
import sys

# Path to the model
model_path = "/home/ubuntu/videoforprocessing_link/models/best_field_keypoint.pt"

try:
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("\n--- Model Names ---")
    print(model.names)
    
    print("\n--- Model Keypoints (if Pose) ---")
    if hasattr(model.model, "kpt_shape"):
        print(f"Keypoint Shape: {model.model.kpt_shape}")
    else:
        print("Not a Pose Model (or kpt_shape hidden).")

except Exception as e:
    print(f"Error loading model: {e}")
