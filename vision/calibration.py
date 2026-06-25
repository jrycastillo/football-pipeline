import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
from ultralytics import YOLO
from vision.color_utils import get_closest_color_name

def calibrate_colors(video_path, weights_path, num_samples=50):
    print(f"[calibration] Scanning {video_path} for dominant team colors...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[calibration] Failed to open video.")
        return None, None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: total_frames = 1000 # Fallback
    
    # Sample random frames
    indices = np.random.choice(total_frames, min(num_samples, total_frames), replace=False)
    indices.sort()
    
    # Load Model locally
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"[calibration] YOLO Load Failed: {e}")
        return None, None
    
    samples = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # Detect
        results = model(frame, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls != 0: continue # Only players
                
                # Extract User Crop
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Center 50%
                h, w = crop.shape[:2]
                cy, cx = h // 2, w // 2
                dy, dx = int(h * 0.25), int(w * 0.25)
                center = crop[cy-dy:cy+dy, cx-dx:cx+dx]
                if center.size == 0: continue
                
                samples.append(cv2.mean(center)[:3]) # BGR
                
    cap.release()
    
    if len(samples) < 2:
        print("[calibration] Insufficient samples.")
        return None, None
        
    # Cluster
    data = np.array(samples, dtype=np.float32)
    try:
        centroids, labels = kmeans2(data, 2, minit='points')
        
        c0n = get_closest_color_name(centroids[0])
        c1n = get_closest_color_name(centroids[1])
        print(f"[calibration] Calibration Complete. Colors: {c0n} ({centroids[0]}), {c1n} ({centroids[1]})")
        
        return centroids[0], centroids[1], c0n, c1n
    except Exception as e:
        print(f"[calibration] Clustering failed: {e}")
        return None, None, None, None
