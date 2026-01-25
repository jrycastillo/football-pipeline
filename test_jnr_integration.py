
import cv2
import torch
import sys
import os
from vision.resnet_recognition import ResNetRecognizerV2

# Setup paths
sys.path.append(os.getcwd())

def test_jnry_model():
    print("1. Initializing ResNetRecognizerV2...")
    # This should load the new ResNet34 Grayscale model by default
    recognizer = ResNetRecognizerV2()
    
    if recognizer.is_grayscale:
        print("✅ Model is correctly identified as Grayscale")
    else:
        print("❌ Model is NOT identified as Grayscale")
        
    print(f"Model type: {type(recognizer.model)}")
    print(f"Input size: {recognizer.size}")
    
    # 2. Load a test image
    test_img_path = "data/real_crops_labeled/images/crop_0000.jpg"
    if not os.path.exists(test_img_path):
        print(f"⚠️ Test image {test_img_path} not found. Creating dummy.")
        import numpy as np
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 20), (40, 80), (255, 255, 255), -1) # Draw a white block
    else:
        img = cv2.imread(test_img_path)
    
    # 3. Predict
    print("\n2. Running Prediction...")
    # The recognizer expects a list of images
    results = recognizer.predict_batch([img])
    
    print("\n3. Results:")
    for res in results:
        print(res)
        
    if results[0]['confidence'] > 0:
        print("\n✅ Inference successful!")
    else:
        print("\n⚠️ Inference returned 0 confidence (might be expected for unknown/bad crop)")

if __name__ == "__main__":
    test_jnry_model()
