import os
import cv2
import yaml
import torch
import numpy as np
import time
from ultralytics import YOLO

# Import User's Threaded Reader
try:
    from utils.threaded_reader import ThreadedVideoReader
except ImportError:
    import sys
    sys.path.append("/home/ubuntu/football")
    from utils.threaded_reader import ThreadedVideoReader

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CLASS = CONFIG["classes"]

def _torso_crop(img, xyxy):
    Himg, Wimg = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    H = y2 - y1; W = x2 - x1
    if H <= 0 or W <= 0: return None
    yA = y1 + int(0.20 * H); yB = y1 + int(0.70 * H)
    xA = x1 + int(0.20 * W); xB = x1 + int(0.80 * W)
    yA = max(0, yA); yB = min(Himg, yB)
    xA = max(0, xA); xB = min(Wimg, xB)
    if yB <= yA or xB <= xA: return None
    return img[yA:yB, xA:xB]

def track_video(video_path, weights_path, ball_model_path=None, max_frames=None, batch_size=32, resume_frame=0):
    conf = CONFIG["heuristics"]["DET_CONF"]
    iou = CONFIG["heuristics"]["DET_IOU"]
    vid_stride = CONFIG["heuristics"]["VID_STRIDE"] 
    imgsz = CONFIG["heuristics"]["DET_IMG_SIZE"]
    store_images = bool(CONFIG["heuristics"]["STORE_IMAGES"])
    store_images_up_to = CONFIG["heuristics"]["STORE_IMAGES_UP_TO"]
    
    if max_frames is None:
        max_frames = CONFIG["heuristics"]["MAX_TRACK_FRAMES"]
    
    device = 0 if torch.cuda.is_available() else "cpu"
    
    print(f"[track] Loading YOLO model from {weights_path} (Batch Size: {batch_size})...")
    model = YOLO(weights_path)
    
    print(f"[track] Initializing ThreadedVideoReader for {video_path} (Resume: {resume_frame})...")
    loader = ThreadedVideoReader(video_path, queue_size=128, start_frame=resume_frame)
    time.sleep(1.0) # Allow buffer to fill as requested
    
    ball_model = YOLO(ball_model_path) if ball_model_path else None
    
    n = resume_frame
    print(f"[track] Starting Threaded & Batched Tracking Loop...")
    
    ball_areas = []
    baseline_ball_area = None
    last_ball_pos = None
    
    while loader.more():
        # Build Batch Manually (Instant Read from Queue)
        batch = []
        for _ in range(batch_size):
            if not loader.more():
                break
            frame = loader.read()
            # If queue is technically empty but stream not stopped, handled by .read() waiting logic?
            # User code: queue.get() blocks. So loader.read() blocks.
            # checks more() first.
            if frame is None: # Just in case
                break
            batch.append(frame)
        
        if not batch:
            break
            
        # Run Inference on Batch (GPU Saturation)
        results = model.track(
            source=batch,
            persist=True,
            tracker="/home/ubuntu/football/football_bytetrack.yaml",
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
            imgsz=imgsz
        )
        
        # Ball Inference (Optional)
        ball_results = [None] * len(batch)
        if ball_model:
            ball_params = {"conf": 0.05, "verbose": False, "device": device, "imgsz": imgsz, "save": False}  # Phase 187: Lower threshold for more ball detections
            ball_results = ball_model.track(source=batch, persist=True, tracker="/home/ubuntu/football/football_bytetrack.yaml", **ball_params)

        # Process Results
        for i, res in enumerate(results):
            img = res.orig_img
            n += 1
            
            frame_data = {
                "orig_shape": img.shape[:2], 
                "path": None, 
                "boxes": [], 
                "crops": []
            }
            
            # 1. Player Detections
            if hasattr(res, "boxes") and len(res.boxes) > 0:
                for b in res.boxes:
                    box_data = {
                        "xyxy": b.xyxy[0].cpu().numpy().tolist(),
                        "id": int(b.id[0].item()) if b.id is not None else None,
                        "cls": int(b.cls[0].item()),
                        "conf": float(b.conf[0].item())
                    }
                    frame_data["boxes"].append(box_data)
                    
                    if box_data["cls"] in (CLASS["player"], CLASS["goalkeeper"]):
                         crop = _torso_crop(img, box_data["xyxy"])
                         if crop is not None and crop.size > 0:
                             frame_data["crops"].append({"box_idx": len(frame_data["boxes"])-1, "img": crop})

            # 2. Ball Detections (Fix for Task 54)
            if ball_results and i < len(ball_results):
                b_res = ball_results[i]
                if b_res and hasattr(b_res, "boxes") and len(b_res.boxes) > 0:
                    ball_dets = []
                    for b in b_res.boxes:
                        # Ball Class = 0 (or config)
                        # We use 0 internally for BallTracker to recognize it
                        ball_dets.append(b.xyxy[0].cpu().numpy().tolist() + [float(b.conf[0].item()), 0]) # [x1, y1, x2, y2, conf, cls]
                    
                    ball_dets = np.array(ball_dets)

                    # Baseline Area Logic (Pseudo-3D)
                    ball_area = 0
                    is_aerial = False
                    
                    if len(ball_dets) > 0:
                        # Simple association: Take highest confidence ball
                        # In a real system, you'd use a Kalman Filter here too
                        best_ball = ball_dets[np.argmax(ball_dets[:, 4])].tolist()
                        
                        # --- 3D Ball Heuristics ---
                        w = best_ball[2] - best_ball[0]
                        h = best_ball[3] - best_ball[1]
                        ball_area = w * h
                        ball_areas.append(ball_area)
                        
                        # Update Baseline (Median of first 100 frames)
                        if len(ball_areas) < 100:
                            baseline_ball_area = np.median(ball_areas)
                        elif baseline_ball_area is None:
                            # Lock baseline after 100 frames
                            baseline_ball_area = np.median(ball_areas)
                            
                        # Velocity Calculation
                        velocity = 0.0
                        bx, by = (best_ball[0]+best_ball[2])/2, (best_ball[1]+best_ball[3])/2
                        if last_ball_pos is not None:
                            velocity = np.sqrt((bx - last_ball_pos[0])**2 + (by - last_ball_pos[1])**2)
                        last_ball_pos = (bx, by)
                        
                        # Aerial Inference
                        # If area is significantly smaller than baseline AND moving fast -> Likely Aerial
                        if baseline_ball_area is not None and baseline_ball_area > 0:
                            if ball_area < 0.7 * baseline_ball_area and velocity > 5.0: # Thresholds heuristic
                                is_aerial = True
                        
                        # Append 3D flag to ball data: [x1, y1, x2, y2, conf, cls, is_aerial]
                        best_ball.append(is_aerial)
                        
                        # Add the processed ball data to frame_data["boxes"]
                        box_data = {
                            "xyxy": best_ball[0:4],
                            "id": None, # Ball usually has no ID or 1
                            "cls": best_ball[5], # Force Class 0 for BallTracker
                            "conf": best_ball[4],
                            "is_aerial": best_ball[6] # Add the new flag
                        }
                        frame_data["boxes"].append(box_data)
                    else:
                        last_ball_pos = None # Reset velocity on missing ball? Or Keep? Resetting is safer.
            
            yield n, frame_data, img
            
            if max_frames and n >= max_frames:
                loader.stop()
                return

            if n % 100 == 0:
                print(f"[track] Processed {n} frames...")
    
    loader.stop()




def legacy_track_video(video_path, weights_path, ball_model_path=None, max_frames=None):
    # Fallback to original logic just in case
    # Re-instantiate model to be safe
    model = YOLO(weights_path)
    kwargs = dict(source=video_path, stream=True, verbose=False, device=0)
    # ... Simplified fallback ...
    results = model.track(**kwargs)
    n = 0
    for res in results:
        n += 1
        yield n, {}, res.orig_img # Dummy fallback
