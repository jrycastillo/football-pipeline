"""
Test color classification + kit discovery across all local test videos.
Samples 15 frames per video spread across the match.
"""
import cv2
import numpy as np
from vision.color_classifier import TeamColorClassifier, KitCoordinator
from ultralytics import YOLO

model = YOLO('models/yolo_player.pt')

videos = [
    ("121364_0.mp4", "Expected: Green vs Red"),
    ("69a33466fc234db.mp4", "Expected: White vs Red (ALG vs MAR)"),
    ("clipped_ikorudo_tornadoes.mp4", "Expected: ? vs ?"),
    ("endpoint_14c0f4e8c4af40d.mp4", "Expected: ? vs ?"),
]

for vid_name, expected in videos:
    path = f"test_videos/{vid_name}"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"\n=== SKIP: {vid_name} (cannot open) ===")
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_min = total / fps / 60 if fps > 0 else 0

    classifier = TeamColorClassifier()
    kit_coord = KitCoordinator()
    all_colors = {}

    # Sample 15 frames evenly across the video (skip first/last 5%)
    start = int(total * 0.05)
    end = int(total * 0.95)
    step = max(1, (end - start) // 15)

    for target in range(start, end, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls not in (1, 2):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            color = classifier.predict(crop)
            all_colors[color] = all_colors.get(color, 0) + 1
            kit_coord.observe(cls, color)

    cap.release()

    kits = kit_coord.get_discovery_result()
    player_kits = kits.get("players", [])

    # Sort colors by count
    sorted_colors = sorted(all_colors.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*70}")
    print(f"VIDEO: {vid_name}")
    print(f"  Duration: {duration_min:.1f} min | Frames: {total} | FPS: {fps:.0f}")
    print(f"  {expected}")
    print(f"  Color counts: {dict(sorted_colors)}")
    print(f"  Kit discovery: players={player_kits}")

    # Check if discovery makes sense
    if len(player_kits) == 2:
        print(f"  RESULT: {player_kits[0]} vs {player_kits[1]}")
    else:
        print(f"  RESULT: Could not discover 2 teams (got {len(player_kits)})")
    print(f"{'='*70}")
