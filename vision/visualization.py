import cv2
import os
import numpy as np

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

def generate_debug_video(video_path, frames_data, jersey_map, output_path):
    print(f"[viz] Generating debug video to {output_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[viz] Error opening video {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < len(frames_data):
            f_data = frames_data[frame_idx]
            
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
