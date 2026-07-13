#!/usr/bin/env python3
"""
Extract calibration frames from a video for pitch landmark annotation.

Samples frames evenly across the match, prioritizing frames with visible pitch lines
and skipping near-duplicates using mean absolute difference (MAD).
"""

import os
import sys
import argparse
import numpy as np

# Guard cv2 import so it prints a clear message if libGL or cv2 is missing on local machines.
try:
    import cv2
except ImportError as e:
    if "cv2" not in sys.modules:
        print(f"Error: cv2 is not importable ({e}).")
        print("Please run this tool on the GPU worker where opencv-python is fully installed.")
        sys.exit(1)

def compute_pitch_score(frame):
    """
    Compute a score indicating the presence of pitch lines.
    
    frame: numpy array of shape (H, W, 3) in BGR format.
    Returns: float score >= 0.0.
    """
    if frame is None or frame.size == 0:
        return 0.0
        
    # Downsample frame for fast processing
    sampled = frame[::4, ::4]
    
    b = sampled[:, :, 0].astype(np.int16)
    g = sampled[:, :, 1].astype(np.int16)
    r = sampled[:, :, 2].astype(np.int16)
    
    # Grass detection: green should be dominant
    green_mask = (g > r + 15) & (g > b + 15) & (g > 40)
    green_ratio = np.mean(green_mask)
    
    # Pitch line (white) detection
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    white_mask = (min_c > 160) & ((max_c - min_c) < 25)
    
    # White ratio in the frame
    white_ratio = np.mean(white_mask)
    
    # Reject frames with insufficient grass coverage
    if green_ratio < 0.15:
        return 0.0
        
    return float(green_ratio * white_ratio)


def compute_mad(frame1, frame2):
    """
    Compute Mean Absolute Difference between two BGR frames.
    """
    if frame1 is None or frame2 is None:
        return 999.0
    s1 = frame1[::4, ::4].astype(np.float32)
    s2 = frame2[::4, ::4].astype(np.float32)
    return float(np.mean(np.abs(s1 - s2)))

def select_best_frame(candidates, last_selected_frame, duplicate_threshold=15.0):
    """
    Select the best frame from candidates based on pitch line score and diversity.
    
    candidates: list of (frame_index, frame_array) tuples.
    last_selected_frame: numpy array of the last selected frame, or None.
    duplicate_threshold: float, skip frames with MAD below this threshold.
    
    Returns: (best_frame_index, best_frame_array, score) or (None, None, 0.0)
    """
    best_idx = None
    best_frame = None
    best_score = -1.0
    
    for idx, frame in candidates:
        if last_selected_frame is not None:
            mad = compute_mad(frame, last_selected_frame)
            if mad < duplicate_threshold:
                continue
                
        score = compute_pitch_score(frame)
        if score > best_score:
            best_score = score
            best_idx = idx
            best_frame = frame
            
    if best_idx is not None and best_score >= 0.0:
        return best_idx, best_frame, best_score
    return None, None, 0.0

def main():
    parser = argparse.ArgumentParser(description="Extract calibration frames for pitch annotation.")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--every_s", type=int, default=20, help="Target interval in seconds between samples")
    parser.add_argument("--max_frames", type=int, default=150, help="Maximum number of frames to extract")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--duplicate_thresh", type=float, default=15.0, help="MAD threshold to skip near-duplicates")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Failed to open video file {args.video}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        print("Error: Could not retrieve video properties (FPS or frame count).")
        cap.release()
        sys.exit(1)
        
    duration_s = total_frames / fps
    print(f"Video loaded: {args.video}")
    print(f"Properties: {total_frames} frames, {fps:.2f} FPS, {duration_s:.1f}s duration")
    
    # Calculate sampling intervals
    target_interval_s = args.every_s
    num_intervals = int(duration_s / target_interval_s)
    
    if num_intervals > args.max_frames:
        target_interval_s = duration_s / args.max_frames
        num_intervals = args.max_frames
        print(f"Adjusting interval to {target_interval_s:.2f}s to respect max_frames={args.max_frames}")
        
    times_s = np.linspace(0, duration_s - target_interval_s, num_intervals)
    base_indices = [int(t * fps) for t in times_s]
    
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    last_selected_frame = None
    extracted_count = 0
    
    print(f"Extracting up to {len(base_indices)} frames...")
    
    for i, base_idx in enumerate(base_indices):
        # Look at a small window of 5 frames starting at base_idx to find the best one
        candidates = []
        for offset_sec in [0, 1, 2, 3, 4]:
            candidate_idx = base_idx + int(offset_sec * fps)
            if candidate_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                candidates.append((candidate_idx, frame))
                
        if not candidates:
            continue
            
        selected_idx, selected_frame, score = select_best_frame(
            candidates, last_selected_frame, args.duplicate_thresh
        )
        
        # If all candidates in the window were duplicates or rejected, relax the duplicate check
        if selected_idx is None:
            selected_idx, selected_frame, score = select_best_frame(
                candidates, None, args.duplicate_thresh
            )
            
        if selected_idx is not None and selected_frame is not None:
            out_path = os.path.join(args.output_dir, f"{video_stem}_f{selected_idx:06d}.jpg")
            cv2.imwrite(out_path, selected_frame)
            last_selected_frame = selected_frame
            extracted_count += 1
            print(f"[{extracted_count}/{len(base_indices)}] Extracted frame {selected_idx} (score: {score:.5f}) -> {out_path}")
            
    cap.release()
    print(f"Done. Extracted {extracted_count} frames to {args.output_dir}")

if __name__ == "__main__":
    main()
