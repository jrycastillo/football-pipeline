import os
import sys
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project and sam2 are in path
PROJECT_ROOT = "/home/ubuntu/football"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from vision.sam2_tracker import SAM2Tracker

def main():
    video_path = "/home/ubuntu/videoforprocessing_link/clipped_ikorudo_tornadoes.mp4"
    player_model_path = "/home/ubuntu/videoforprocessing_link/football_analysis_v2/data/models/best_player_detect.pt"
    
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found.")
        return

    # 1. Load Player Detector
    print("🚀 Loading YOLO model...")
    player_model = YOLO(player_model_path)
    
    # 2. Initialize SAM2 Tracker
    print("🚀 Initializing SAM2 Tracker...")
    # Use small for testing speed
    tracker = SAM2Tracker(model_type="small") 
    tracker.init_video(video_path)
    
    # 3. Detect on Frame 0
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Failed to read first frame.")
        return
    
    print("🚀 Detecting players on Frame 0...")
    results = player_model(frame, classes=[1, 2], conf=0.3)[0] # Use players and goalkeepers
    
    detections = []
    for i, box in enumerate(results.boxes.xyxy):
        detections.append({
            'id': i,
            'bbox': box.tolist()
        })
    print(f"Found {len(detections)} players.")

    # 4. Prompt SAM2
    tracker.add_player_prompts(frame_idx=0, detections=detections)
    
    # 5. Propagate
    print("🚀 Propagating tracking (20 frames)...")
    limit = 20
    count = 0
    for frame_idx, obj_ids, mask_logits in tracker.propagate(start_frame_idx=0):
        if frame_idx > limit:
            break
            
        bboxes = tracker.get_bboxes_from_masks(obj_ids, mask_logits)
        print(f"Frame {frame_idx}: Tracked {len(bboxes)}/{len(detections)} objects.")
        count += 1
        
    print(f"✅ SAM2 Full Test Complete ({count} frames processed)!")

if __name__ == "__main__":
    main()
