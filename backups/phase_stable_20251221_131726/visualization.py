import cv2
import os
import numpy as np

def get_jersey_color(crop):
    """
    Classify jersey color using center 50% crop and RGB means.
    """
    if crop is None or crop.size == 0:
        return "Unknown"
        
    # Check for Calibrated References first
    if REF_COLOR_0 is not None and REF_COLOR_1 is not None:
         return get_calibrated_color(crop)
    
    # Center crop 50%
    h, w = crop.shape[:2]
    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.25), int(w * 0.25)
    center_crop = crop[cy-dy:cy+dy, cx-dx:cx+dx]
    
    if center_crop.size == 0: return "Unknown"
    
    # Mean RGB
    mean_color = cv2.mean(center_crop)[:3] # struct returns (B, G, R, A)
    B, G, R = mean_color
    
    # Heuristic (User Requested)
    # R>200, G>200, B>200 -> White
    # R>150, G<100 -> Red
    
    if R > 200 and G > 200 and B > 200: return "White"
    if R < 50 and G < 50 and B < 50: return "Black"
    
    if R > 150 and G < 100 and B < 100: return "Red"
    if B > 150 and R < 100 and G < 100: return "Blue"
    if R > 200 and G > 200 and B < 100: return "Yellow"
    if G > 150 and R < 100 and B < 100: return "Green" # Simple green check
    
    # Fallback based on dominant channel if distinct
    # Avoid gray/muddy colors being classified poorly
    
    return "Unknown"

# Global References (set by orchestrator)
# Global References (set by orchestrator)
REF_COLOR_0 = None
REF_COLOR_1 = None
REF_NAME_0 = "Home"
REF_NAME_1 = "Away"

def set_reference_colors(c0, c1, n0="Home", n1="Away"):
    global REF_COLOR_0, REF_COLOR_1, REF_NAME_0, REF_NAME_1
    REF_COLOR_0 = c0
    REF_COLOR_1 = c1
    REF_NAME_0 = n0
    REF_NAME_1 = n1
    print(f"[viz] Reference Colors Set: {n0} / {n1}")

def get_calibrated_color(crop):
    if REF_COLOR_0 is None or REF_COLOR_1 is None:
        return get_jersey_color(crop) # Fallback
        
    if crop is None or crop.size == 0: return "Unknown"
    
    # Center Crop
    h, w = crop.shape[:2]
    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.25), int(w * 0.25)
    center = crop[cy-dy:cy+dy, cx-dx:cx+dx]
    if center.size == 0: return "Unknown"
    
    mean_bgr = cv2.mean(center)[:3]
    
    # Distance
    d0 = np.sum((np.array(mean_bgr) - np.array(REF_COLOR_0))**2)
    d1 = np.sum((np.array(mean_bgr) - np.array(REF_COLOR_1))**2)
    
    return REF_NAME_0 if d0 < d1 else REF_NAME_1

def draw_hud(frame, box, track_id, qwen_text, voting_result):
    x1, y1, x2, y2 = map(int, box)

    # 1. Bounding Box (Green)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 2. Info Tag Background
    info_y = max(0, y1 - 60)
    # Ensure background doesn't go out of bounds
    # cv2.rectangle(frame, (x1, info_y), (x1 + 180, y1), (0, 0, 0), -1)
    # cv2.rectangle(frame, (x1, info_y), (x1 + 180, y1), (0, 255, 0), 1)
    
    # Use a slightly more robust background drawing
    bg_h = 60
    bg_w = 180
    cv2.rectangle(frame, (x1, info_y), (x1 + bg_w, info_y + bg_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, info_y), (x1 + bg_w, info_y + bg_h), (0, 255, 0), 1)

    # 3. Text Stats
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"ID: {track_id}", (x1 + 5, info_y + 20), font, 0.5, (255, 255, 255), 1)

    # Show Voting Result (The "Best" number)
    # If 'Verify...', show in Yellow. If confirmed number, show in Cyan.
    color = (0, 255, 255) if "Verify" in str(voting_result) else (255, 255, 0)
    cv2.putText(frame, f"Jersey: {voting_result}", (x1 + 5, info_y + 45), font, 0.6, color, 2)
    
    # Optional: Show Qwen raw text if available
    if qwen_text:
         cv2.putText(frame, f"Q: {qwen_text}", (x1 + 5, info_y + 58), font, 0.4, (200, 200, 200), 1)

def draw_skeleton(frame, keypoints):
    # Standard skeleton connections (e.g., COCO format)
    skeleton_connections = [
        (5, 7), (7, 9), (6, 8), (8, 10), 
        (11, 13), (13, 15), (12, 14), (14, 16), 
        (5, 6), (11, 12)
    ]
    
    # keypoints is list of [x, y] or [x, y, conf]
    for i, j in skeleton_connections:
        if i < len(keypoints) and j < len(keypoints):
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            
            # Check confidence if available (usually index 2)
            conf1 = keypoints[i][2] if len(keypoints[i]) > 2 else 1.0
            conf2 = keypoints[j][2] if len(keypoints[j]) > 2 else 1.0
            
            if pt1[0] > 0 and pt2[0] > 0 and conf1 > 0.5 and conf2 > 0.5:
                cv2.line(frame, pt1, pt2, (255, 0, 255), 2) # Magenta Bones

def generate_debug_video(video_path, frames_data, jersey_map, output_path, ball_track=None):
    print(f"[viz] Generating debug video to {output_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[viz] Error opening video {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Ensure dimensions are even (requirement for some codecs)
    if width % 2 != 0: width -= 1
    if height % 2 != 0: height -= 1
        
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < len(frames_data):
            f_data = frames_data[frame_idx]
            
            # Draw Ball (Underlay)
            if ball_track and frame_idx < len(ball_track):
                ball_pos = ball_track[frame_idx]
                if ball_pos:
                    bx, by = map(int, ball_pos)
                    # Draw Orange Circle for Ball
                    cv2.circle(frame, (bx, by), 10, (0, 165, 255), -1)
                    cv2.circle(frame, (bx, by), 12, (0, 0, 0), 2)
                    cv2.putText(frame, "BALL", (bx - 20, by - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            # Draw Boxes
            for b in f_data["boxes"]:
                if b["id"] is not None:
                    # Get Jersey Info
                    j_info = jersey_map.get(b["id"], {})
                    j_num = j_info.get("number", "?")
                    q_text = j_info.get("qwen_text", "") # If we stored it
                    
                    draw_hud(frame, b["xyxy"], b["id"], q_text, j_num)
                    
                    # Draw Skeleton if available
                    if "keypoints" in b:
                        draw_skeleton(frame, b["keypoints"])
                        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"[viz] Processed {frame_idx} frames...")
            
    cap.release()
    out.release()
    print(f"[viz] Video saved to {output_path}")
