"""
Debug script: Analyze color classification on frame_80850 (ALG white vs MAR red).
"""
import cv2
import numpy as np
from vision.color_classifier import TeamColorClassifier
from ultralytics import YOLO

img = cv2.imread('/Users/ronan/.gemini/antigravity/brain/e8d8b140-e7ef-4797-aff2-02a749fc8266/frame_80850.jpg')
if img is None:
    print("ERROR: Could not load frame_80850.jpg")
    exit(1)

try:
    model = YOLO('models/yolo_player.pt')
    use_custom = True
except:
    model = YOLO('yolov8x.pt')
    use_custom = False

results = model(img, verbose=False)
classifier = TeamColorClassifier()

colors = {}
for box in results[0].boxes:
    cls = int(box.cls[0])
    if use_custom and cls not in (1, 2):
        continue
    if not use_custom and cls != 0:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    color = classifier.predict(crop)
    colors[color] = colors.get(color, 0) + 1

print(f"Results: {colors}")
print(f"Expected: ~8 Red (Morocco), ~9 White (Algeria), 1 Blue, 1 Green")
