"""
Generate annotated frames for the color fix report.
Draws bounding boxes colored by the classifier's prediction on sample frames.
"""
import cv2
import numpy as np
import os
from vision.color_classifier import TeamColorClassifier
from ultralytics import YOLO

model = YOLO('models/yolo_player.pt')
classifier = TeamColorClassifier()

# BGR colors for drawing
COLOR_BGR = {
    "Red": (0, 0, 255),
    "Green": (0, 200, 0),
    "Blue": (255, 100, 0),
    "White": (255, 255, 255),
    "Black": (50, 50, 50),
    "Yellow": (0, 255, 255),
    "Orange": (0, 140, 255),
    "Purple": (200, 0, 200),
    "Unknown": (128, 128, 128),
}

out_dir = "output/color_report"
os.makedirs(out_dir, exist_ok=True)

videos = [
    ("test_videos/121364_0.mp4", 400, "121364_green_vs_red"),
    ("test_videos/69a33466fc234db.mp4", 40000, "69a33_white_vs_red_40k"),
    ("test_videos/69a33466fc234db.mp4", 80850, "69a33_white_vs_red_80k"),
    ("test_videos/clipped_ikorudo_tornadoes.mp4", 800, "ikorudo_white_vs_red"),
    ("test_videos/endpoint_14c0f4e8c4af40d.mp4", 50000, "14c0f4_green_vs_white"),
]

for vid_path, frame_num, name in videos:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"SKIP: {vid_path}")
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"SKIP frame {frame_num} from {vid_path}")
        continue

    results = model(frame, verbose=False)
    colors_count = {}
    annotated = frame.copy()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls not in (1, 2):
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        color = classifier.predict(crop)
        colors_count[color] = colors_count.get(color, 0) + 1

        bgr = COLOR_BGR.get(color, (128, 128, 128))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        # Label background
        label = color
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), bgr, -1)
        # Text color: black for light boxes, white for dark
        txt_color = (0, 0, 0) if color in ("White", "Yellow", "Green") else (255, 255, 255)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

    # Add summary text at bottom
    summary = " | ".join(f"{c}: {n}" for c, n in sorted(colors_count.items(), key=lambda x: -x[1]))
    cv2.putText(annotated, summary, (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = f"{out_dir}/{name}.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Saved {out_path} | {colors_count}")

print(f"\nAll frames saved to {out_dir}/")
