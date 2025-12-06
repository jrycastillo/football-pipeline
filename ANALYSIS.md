# Football Pipeline Analysis and Fix Report

## Overview
The pipeline is a comprehensive solution for extracting player statistics and xG (Expected Goals) from football match footage using computer vision. It leverages YOLO for detection/tracking/pose estimation and a ResNet-based model for Jersey Number Recognition (JNR).

## Identified Issues & Fixes

### 1. Shot Detection Logic
**Issue:** The shot detection algorithm analyzed the ball velocity only within the window of player possession. Since peak ball velocity (the shot itself) typically occurs *after* the ball leaves the player's control (i.e., in the frame immediately following the end of possession), the pipeline was often missing shots, especially powerful ones where possession was lost instantly.
**Fix:** The search window for peak velocity (`t_star`) was extended to look ahead by 3 frames after ownership ends. This ensures the "kick" velocity is captured even if ownership is lost immediately.

### 2. Team Assignment Robustness
**Issue:** The original team assignment logic relied on a heuristic comparing player jersey Hue to fixed "Sky" and "Red" reference colors. This was fragile for teams with colors that didn't align with these axes (e.g., Green vs. Yellow) or for achromatic kits (White/Black) where Hue is unstable.
**Fix:** Replaced the heuristic with **K-Means Clustering (k=2)** using OpenCV.
- The feature vector for each player now includes `[cos(H), sin(H), Saturation, Value]`.
- This allows the system to unsupervisedly separate the two dominant team colors regardless of what they are, and handles white/black kits better by including Saturation and Value.
- Replaced `scipy` with `cv2.kmeans` to avoid extra dependencies.

### 3. Memory vs. Feature Availability
**Issue:** The "memory-aware" feature (`STORE_IMAGES_UP_TO=1500`) caused `orig_img` to be discarded after the first minute of video. This meant:
- No jersey number recognition after minute 1.
- No pitch polygon refinement after minute 1.
**Fix:** Implemented a **Hybrid Storage Strategy**.
- `orig_img` is now stored if `frame_idx <= STORE_IMAGES_UP_TO` (for initial dense sampling) **OR** if `frame_idx % STORE_INTERVAL == 0` (default 30).
- This ensures that Jersey Recognition and other image-dependent tasks have access to samples throughout the entire match without storing every single frame in memory.

### 4. Possession Logic
**Issue:** The possession logic defaulted to `max_owner_px=None`, which meant the "nearest" player always owned the ball, even if the ball was 50 meters away in the air.
**Fix:** Set a default `max_owner_px=150` in the main execution pipeline. This prevents assigning possession during long passes or clearances when no player is actually in control.

## Architecture Analysis

### Strengths
- **Modularity:** Clear separation of tracking, team assignment, JNR, and stats calculation phases.
- **Advanced Features:** Integration of Pose estimation for pitch polygon and custom JNR model adds significant value over simple tracking.
- **xG Model:** The heuristic xG model (distance + angle + pressure + header) is a good lightweight approximation.

### Remaining Limitations
- **Sequential Processing:** The pipeline processes the entire video to track, *then* calculates stats. For very long videos (full matches), the `frames` list (even without images) can be large. A streaming architecture (process & discard) would be more memory efficient but harder to implement with the current multi-pass logic (e.g., smoothing or global team clustering).
- **Environment Variables:** The code relies heavily on environment variables for configuration. Moving to a config file or CLI arguments would improve usability.
- **Error Handling:** The JNR and Pose models fail silently or print errors if weights are missing. This is good for robustness but might hide setup issues from the user.

## Conclusion
The pipeline is now significantly more robust. The fixes ensure that key events (shots) are detected correctly and that the system works for a wider variety of team colors and video lengths.
