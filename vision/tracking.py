import os
import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CLASS = CONFIG["classes"]

def _torso_crop(img, xyxy):
    Himg, Wimg = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    H = y2 - y1; W = x2 - x1
    if H <= 0 or W <= 0:
        return None
    yA = y1 + int(0.20 * H); yB = y1 + int(0.70 * H)
    xA = x1 + int(0.20 * W); xB = x1 + int(0.80 * W)
    yA = max(0, yA); yB = min(Himg, yB)
    xA = max(0, xA); xB = min(Wimg, xB)
    if yB <= yA or xB <= xA:
        return None
    return img[yA:yB, xA:xB]



def track_video(video_path, weights_path, ball_model_path=None, max_frames=None):
    conf = CONFIG["heuristics"]["DET_CONF"]
    iou = CONFIG["heuristics"]["DET_IOU"]
    vid_stride = CONFIG["heuristics"]["VID_STRIDE"]
    imgsz = CONFIG["heuristics"]["DET_IMG_SIZE"]
    store_images = bool(CONFIG["heuristics"]["STORE_IMAGES"])
    store_images_up_to = CONFIG["heuristics"]["STORE_IMAGES_UP_TO"]
    
    # Use argument if provided, else config
    if max_frames is None:
        max_frames = CONFIG["heuristics"]["MAX_TRACK_FRAMES"]
    
    device = 0 if torch.cuda.is_available() else "cpu"
    
    print(f"[track] Loading YOLO model from {weights_path}...")
    model = YOLO(weights_path)
    ball_model = YOLO(ball_model_path) if ball_model_path else None
    
    use_half = (device != "cpu")
    
    kwargs = dict(
        source=video_path, 
        tracker="bytetrack.yaml", 
        conf=conf, 
        iou=iou, 
        stream=True, 
        verbose=False, 
        device=device, 
        persist=True, 
        vid_stride=vid_stride, 
        half=use_half
    )
    if imgsz not in (None, 0):
        kwargs["imgsz"] = imgsz
    kwargs["save"] = False 
        
    stream = model.track(**kwargs)
    
    ball_stream = None
    if ball_model:
        ball_kwargs = kwargs.copy()
        if conf > 0.15:
            ball_kwargs["conf"] = 0.15
        ball_stream = ball_model.track(**ball_kwargs)

    frames = []
    n = 0
    
    iterator = zip(stream, ball_stream) if ball_stream else stream
    
    print(f"[track] Starting tracking loop...")
    
    for item in iterator:
        if ball_stream:
            res, res_ball = item
        else:
            res = item
            
        n += 1
        if max_frames and n > max_frames:
            break
            
        img = res.orig_img
        frame = {
            "orig_shape": img.shape[:2], 
            "path": getattr(res, "path", None), 
            "boxes": [], 
            "crops": []
        }
        
        # Store full image if requested
        if store_images and (n <= store_images_up_to or n % 30 == 0):
            frame["orig_img"] = img
            
        # Process Main Model
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            b = res.boxes
            ids = (b.id.cpu().numpy().astype(int).tolist() if b.id is not None else [None] * len(b))
            cls = b.cls.cpu().numpy().astype(int).tolist()
            confs = b.conf.cpu().numpy().astype(float).tolist()
            xyxy = b.xyxy.cpu().numpy().astype(float).tolist()
            
            for i in range(len(cls)):
                if ball_model and cls[i] == CLASS["ball"]:
                    continue 
                    
                box = {"id": ids[i], "cls": cls[i], "conf": confs[i], "xyxy": xyxy[i]}
                frame["boxes"].append(box)
                
                # Store crops for JNR/Team Color
                if box["cls"] in (CLASS["player"], CLASS["goalkeeper"]):
                    if (n - 1) % 30 == 0:
                        crop = _torso_crop(img, box["xyxy"])
                        if crop is not None and crop.size > 0:
                            frame["crops"].append({"box_idx": len(frame["boxes"])-1, "img": crop})

        # Process Ball Model
        if ball_stream and hasattr(res_ball, "boxes") and res_ball.boxes is not None and len(res_ball.boxes) > 0:
            b = res_ball.boxes
            ids = (b.id.cpu().numpy().astype(int).tolist() if b.id is not None else [None] * len(b))
            cls = b.cls.cpu().numpy().astype(int).tolist()
            confs = b.conf.cpu().numpy().astype(float).tolist()
            xyxy = b.xyxy.cpu().numpy().astype(float).tolist()
            for i in range(len(cls)):
                if cls[i] == 32: 
                    ball_id = ids[i] + 10000 if ids[i] is not None else None
                    box = {"id": ball_id, "cls": CLASS["ball"], "conf": confs[i], "xyxy": xyxy[i]}
                    frame["boxes"].append(box)

        frames.append(frame)
        
        if n % 100 == 0:
            print(f"[track] Processed {n} frames...")
            
    return frames, model
