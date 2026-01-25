
# Football Analysis Pipeline - Phase 85 Consolidated (Refined)
# ----------------------------------------------------
# This script combines the core modules of the Phase 79 pipeline 
# into a single executable flow for regression testing.
#
# UPDATES (Phase 85):
# 1. Purged "Cheating" Logic (No ID Mirroring)
# 2. Visual Signal Maximization (Torso Crop + Super-Res)
# 3. Real Qwen2.5-VL Integration with Structured JSON Prompt
# 4. Bayesian Temporal Aggregation

import os
import sys

# FORCE HF CACHE to Local Directory to avoid Permission Errors
os.environ["HF_HOME"] = "/home/ubuntu/football/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import yaml
import time
import json
import logging

# --- 0. ENV SETUP ---
# --- 0. ENV SETUP ---
# CACHE SET AT TOP OF FILE


# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("output/pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def log(msg):
    logging.info(msg)
    # print(msg) # StreamHandler handles this

import cv2
import numpy as np
import torch
import math
from collections import defaultdict, Counter, deque
from stats.metrics import StatsEngine # Import new engine
from vision.color_classifier import TeamColorClassifier, KitCoordinator  # Phase 139
from ultralytics import YOLO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# --- 1. CONFIGURATION ---
try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    log("Warning: config.yaml not found, using defaults.")
    CONFIG = {
        "env": {"DET_WEIGHTS": "models/detect.pt", "JNR_WEIGHTS": None, "BALL_MODEL_PATH": None},
        "heuristics": {"FPS": 25, "DET_CONF": 0.10, "DET_IOU": 0.50, "DET_IMG_SIZE": 832, "VID_STRIDE": 1, "MAX_TRACK_FRAMES": None, "STORE_IMAGES": 1, "STORE_IMAGES_UP_TO": 1500},
        "classes": {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}
    }

print("Configuration Loaded.")

# --- 2. THREADED VIDEO READER ---
import threading
import queue

class ThreadedVideoReader:
    def __init__(self, path, queue_size=256, start_frame=0):
        self.stream = cv2.VideoCapture(path)
        if start_frame > 0:
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.stopped = False
        self.queue = queue.Queue(maxsize=queue_size)
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.stopped: return
            if not self.queue.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    log(f"[reader] Stream end or error at frame {int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))}")
                    self.stopped = True
                    return
                self.queue.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.queue.get()

    def more(self):
        # Wait a bit for the queue to populate if not stopped and queue is empty
        wait_start = time.time()
        while self.queue.empty() and not self.stopped and (time.time() - wait_start < 5.0):
            time.sleep(0.1)
        return not self.queue.empty() or not self.stopped

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

# --- 3. CAMERA & PITCH ---
class PitchManager:
    def __init__(self, model_path="models/yolov8n-pose.pt", device=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                log(f"[pitch] Loaded Pitch Model from {model_path}")
            except Exception as e:
                log(f"[pitch] Failed to load model: {e}")
        else:
            log(f"[pitch] Warning: Model {model_path} not found. Using Fallback.")
        
        self.device = device
        self.H_default = np.array([
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
    def predict(self, frame):
        if self.model is None: return None, self.H_default
        results = self.model.predict(frame, verbose=False, device=self.device)
        if not results: return None, self.H_default
        return results[0].keypoints, self.H_default

class Camera:
    def __init__(self, homography_matrix=None):
        self.H = np.array(homography_matrix) if homography_matrix is not None else np.eye(3)
        if homography_matrix is None:
             self.H[0, 0] = 0.05; self.H[1, 1] = 0.05 # Fallback Scale

    def update(self, H):
        self.H = H

    def project_point(self, x, y):
        p = np.array([x, y, 1.0])
        mapped = np.dot(self.H, p)
        if mapped[2] != 0: return (mapped[0] / mapped[2], mapped[1] / mapped[2])
        return (0.0, 0.0)

    def calculate_distance(self, p1, p2): # p1, p2 are PIXELS
        x1_m, y1_m = self.project_point(p1[0], p1[1])
        x2_m, y2_m = self.project_point(p2[0], p2[1])
        return np.sqrt((x2_m - x1_m)**2 + (y2_m - y1_m)**2)

# --- 4. VISUALIZATION HELPERS ---
def get_jersey_color(crop):
    """Detect jersey color using K-means clustering to find dominant color.
    Filters out grass before analysis.
    """
    from sklearn.cluster import KMeans
    
    if crop is None or crop.size == 0: return "Unknown"
    
    h, w = crop.shape[:2]
    if h < 10 or w < 10: return "Unknown"
    
    # Sample center 40% to focus on jersey
    y1, y2 = int(h * 0.30), int(h * 0.70)
    x1, x2 = int(w * 0.30), int(w * 0.70)
    center_crop = crop[y1:y2, x1:x2]
    
    if center_crop.size == 0: 
        center_crop = crop
    
    # Convert to HSV
    hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    
    h_vals = pixels[:, 0]
    s_vals = pixels[:, 1]
    
    # Filter out grass (green hue 40-80)
    grass_mask = (h_vals >= 40) & (h_vals <= 80) & (s_vals > 40)
    filtered = pixels[~grass_mask]
    
    if len(filtered) < 20:
        filtered = pixels
    
    try:
        kmeans = KMeans(n_clusters=2, n_init=3, random_state=42)
        kmeans.fit(filtered)
        
        centers = kmeans.cluster_centers_
        counts = np.bincount(kmeans.labels_)
        
        # Pick cluster with highest saturation-weighted count
        best_idx = 0
        best_score = 0
        for i in range(len(centers)):
            sat = centers[i][1]
            score = counts[i] * (1 + sat / 100)
            if score > best_score:
                best_score = score
                best_idx = i
        
        h_val = centers[best_idx][0]
        s = centers[best_idx][1]
        v = centers[best_idx][2]
    except:
        h_val = np.median(filtered[:, 0])
        s = np.median(filtered[:, 1])
        v = np.median(filtered[:, 2])
    
    # Classify color
    if s < 40:
        if v > 150: return "White"
        return "Black"
    
    # Color ranges for Red vs Green match
    # Red range includes orange/yellow shades (0-50)
    if h_val < 50 or h_val > 165: return "Red"
    if 50 <= h_val < 85: return "Green"
    if 85 <= h_val < 130: return "Blue"
    if 130 <= h_val <= 165: return "Purple"
    
    return "Unknown"

# --- 5. IDENTITY MANAGER ---
class IdentityManager:
    def __init__(self):
        self.jersey_registry = {} 
        self.player_colors = {}
        self.track_colors = {}
        self.track_classes = {}  # Phase 132: {track_id: cls_id} (1=GK, 2=Player, 3=Referee)
        self.active_bindings = {}
        self.vote_buffer = {} 
        self.last_seen = {} 
        self.consecutive_counter = {} # {track_id: (jersey_num, count)}
        self.master_registry = {} # {jersey_num: (original_track_id, color)} - Phase 104
        self.color_registry = {} # {color_key: jersey_num} - Phase 113
        self.player_last_pos = {} 
        # Phase 128: Smart Role Detection
        self.color_counter = {}  # {color: count} - track color frequencies
        self.team_colors = []    # [Team A color, Team B color] - two most common
        self.goalkeeper_zone_threshold = 0.15  # Top/bottom 15% of pitch = GK zone 
        self.locking_mode = 1 # Default to Mode 1 (Confidence-based) for faster locking
        self.jersey_gallery = defaultdict(list) # Phase v26: {jersey_num: [pil_image, ...]}
        self.track_map = {} # Phase v26: {raw_tid: stable_tid}
        # Phase v27: Bayesian Dirichlet Consensus
        self.alpha = defaultdict(lambda: defaultdict(lambda: 1.0)) # {track_id: {jersey_num: score}}
        
    def set_locking_mode(self, mode):
        self.locking_mode = int(mode)
        log(f"⚙️ [IdentityManager] Locking Mode set to {self.locking_mode}")
        
    def get_player_color(self, jersey_num):
        return self.player_colors.get(str(jersey_num)) or self.player_colors.get(int(jersey_num))

    def is_jersey_number(self, val):
        return str(val) in self.jersey_registry or int(val) in self.jersey_registry if str(val).isdigit() else False

    def touch(self, track_id, frame_idx):
        self.last_seen[track_id] = frame_idx

    def process_detection(self, track_id, detected_number, confidence_str, confidence_val=0.0, detected_color=None):
        if track_id in self.active_bindings: return

        # Confidence Scoring
        score = 0.0
        if isinstance(confidence_val, float) and confidence_val > 0:
            score = confidence_val
        else:
            if str(confidence_str) == "high": score = 0.9
            elif str(confidence_str) == "medium": score = 0.6
            elif str(confidence_str) == "low": score = 0.3
            
            # Phase v26: Use VLM color to refine track color if confident
            if detected_color and score >= 0.7:
                self.set_track_color(track_id, detected_color)
        
        # --- MODE 1: Instant Lock on High Conf ---
        if self.locking_mode == 1:
            if score >= 0.65: # LOWERED to 65% per user request
                log(f"🔒 [IdentityManager] MODE 1 LOCK: Track {track_id} -> Jersey #{detected_number}")
                self._lock_identity(track_id, detected_number)
            return

        # --- MODE 2: Consecutive High Conf (Replacement for Bayesian) ---
        if self.locking_mode == 2:
            if score >= 0.65: # LOWERED to 65% per user request
                last_num, count = self.consecutive_counter.get(track_id, (None, 0))
                if detected_number == last_num:
                    count += 1
                else:
                    count = 1
                
                self.consecutive_counter[track_id] = (detected_number, count)
                
                if count >= 2:
                    log(f"🔒 [IdentityManager] MODE 2 LOCK: Track {track_id} -> Jersey #{detected_number}")
                    # Capture crop for gallery if not provided as argument, we'll needs to pass it in
                    self._lock_identity(track_id, detected_number)
            return

        # --- MODE 3: Bayesian Dirichlet Consensus (Phase v27) ---
        if self.locking_mode == 3:
            # Update Dirichlet parameters with weighted evidence
            # We treat confidence as a weight for the observation
            self.alpha[track_id][detected_number] += score
            
            # Calculate total evidence and expected probability
            track_alphas = self.alpha[track_id]
            total_evidence = sum(track_alphas.values())
            
            if total_evidence > 2.5: # Minimum evidence required (approx 3 strong detections)
                best_number = max(track_alphas, key=track_alphas.get)
                expected_prob = track_alphas[best_number] / total_evidence
                
                if expected_prob > 0.7: # Confidence threshold for winner
                    log(f"🔒 [IdentityManager] MODE 3 LOCK: Track {track_id} -> Jersey #{best_number} (Prob: {expected_prob:.2f}, Evidence: {total_evidence:.1f})")
                    self._lock_identity(track_id, best_number)
            return

    def _lock_identity(self, track_id, jersey_num, reference_crop=None):
        if jersey_num not in self.jersey_registry:
            self.jersey_registry[jersey_num] = f"Player_Jersey_{jersey_num}"
            log(f"🆕 [IdentityManager] NEW PLAYER CREATED: Jersey #{jersey_num} (Track {track_id})")
        else:
             log(f"🔄 [IdentityManager] WELCOME BACK: Track {track_id} re-linked to Jersey #{jersey_num}")
        self.active_bindings[track_id] = jersey_num

        # Phase v26: Safer gallery entry (only if locked OR extremely high conf)
        if reference_crop is not None:
            # Phase v27: Handle Temporal Sequences (Pick middle frame for gallery)
            if isinstance(reference_crop, list):
                if len(reference_crop) > 0:
                    mid_idx = len(reference_crop) // 2
                    reference_crop = reference_crop[mid_idx]
                else:
                    reference_crop = None
            
            if reference_crop is not None:
                # Only add to gallery if we have consensus or very high single-shot confidence
                is_locked = jersey_num in self.jersey_registry
                if is_locked or len(self.jersey_gallery[str(jersey_num)]) == 0:
                    if len(self.jersey_gallery[str(jersey_num)]) < 3:
                        # FINAL SAFETY CHECK (Phase v27.1)
                        if isinstance(reference_crop, np.ndarray) and reference_crop.ndim == 3:
                            # Store as PIL for Qwen direct usage
                            rgb = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(rgb)
                            self.jersey_gallery[str(jersey_num)].append(pil_img)
                            log(f"🖼️ [IdentityManager] Added reference crop to Gallery for #{jersey_num}")
                        else:
                            log(f"⚠️ [IdentityManager] Rejected reference crop for #{jersey_num} (Invalid Type/Shape: {type(reference_crop)})")

    def get_track_color(self, track_id):
        return self.track_colors.get(track_id, "Unknown")

    def resolve_identity(self, current_track_id, jersey_number, team_color):
        # Phase 107: Jersey-Only Matching (Ignore Color Inconsistencies)
        if jersey_number is None:
            return current_track_id 

        # Use JERSEY NUMBER as primary key (more stable than color)
        player_key = str(jersey_number)

        # CHECK: Do we already know this player?
        if player_key in self.master_registry:
            # Phase v27.2: Disable eager mapping in Bayesian mode (Collision Fix)
            # We want Dirichlet evidence to build independently for each track
            if self.locking_mode == 3:
                return current_track_id

            original_id, original_color = self.master_registry[player_key]

            # If the tracker assigned a NEW ID to a known player
            if current_track_id != original_id:
                # TEAM COLOR SANITY CHECK (Phase v26.2)
                # Relaxed: Only reject if we have multiple confirmed samples of the original color
                known_count = len(self.jersey_gallery.get(jersey_number, []))
                if known_count >= 2 and original_color != "Unknown" and team_color != "Unknown" and original_color != team_color:
                    log(f"⚠️ [IdentityManager] REJECTED {jersey_number} due to color mismatch: {team_color} vs known {original_color}")
                    return current_track_id
                
                log(f"🔄 [Re-ID] Persistent Mapping: Track {current_track_id} -> {original_id} (Jersey #{jersey_number})")
                self.track_map[current_track_id] = original_id
                # Inherit the ORIGINAL color (more reliable)
                self.set_track_color(current_track_id, original_color)
                return original_id # FORCE the old ID
        else:
            # First time seeing this jersey. Register it with color.
            self.master_registry[player_key] = (current_track_id, team_color)
            self.track_map[current_track_id] = current_track_id
            # FIX: Populate player_colors so StatsEngine knows the team
            self.player_colors[player_key] = team_color
            
            # Also register in color_registry for color-based Re-ID
            color_key = f"{team_color}_{current_track_id}"
            self.color_registry[color_key] = jersey_number
            log(f"📝 [IdentityManager] Registered Jersey #{jersey_number} to Track {current_track_id} ({team_color})")

        return current_track_id
    
    def resolve_by_color(self, current_track_id, team_color):
        """
        Phase 113: Color-Based Re-ID for Turn-Around Persistence
        When player turns around (no jersey visible), match by color.
        """
        if team_color == "Unknown":
            return current_track_id
        
        # Look for any locked jersey with this color
        for jersey_num, (orig_tid, orig_color) in self.master_registry.items():
            if orig_color == team_color and orig_tid != current_track_id:
                # Check if original track is "lost" (not seen recently)
                # For now, we'll trust color matching if colors match exactly
                if current_track_id not in self.active_bindings:
                    log(f"🎨 [Color Re-ID] Track {current_track_id} ({team_color}) -> Track {orig_tid} (Jersey #{jersey_num})")
                    return orig_tid
        
        return current_track_id

    def set_track_color(self, track_id, color):
        if track_id not in self.track_colors or self.track_colors[track_id] == "Unknown":
            self.track_colors[track_id] = color
            # Phase 128: Count color frequency
            if color != "Unknown":
                self.color_counter[color] = self.color_counter.get(color, 0) + 1
    
    def detect_team_colors(self):
        """Detect the two team colors as the most common colors (excluding Gray/Unknown)."""
        if not self.color_counter:
            return
        # Filter out non-team colors
        valid_colors = {c: cnt for c, cnt in self.color_counter.items() 
                       if c not in ["Unknown", "Gray"]}
        if len(valid_colors) < 2:
            return
        # Sort by frequency, take top 2
        sorted_colors = sorted(valid_colors.items(), key=lambda x: x[1], reverse=True)
        self.team_colors = [sorted_colors[0][0], sorted_colors[1][0]]
        log(f"🏟️ [Team Detection] Team A: {self.team_colors[0]}, Team B: {self.team_colors[1]}")

    def set_track_class(self, track_id, cls_id):
        """Store detection class for a track (only set once)."""
        if track_id not in self.track_classes:
            self.track_classes[track_id] = cls_id

    def get_role(self, track_id, y_pos=None, frame_height=None):
        """
        Role assignment based on model detection class (Phase 132).
        Model classes: 1=Goalkeeper, 2=Player, 3=Referee
        """
        # Use model detection class directly
        cls_id = self.track_classes.get(track_id)
        if cls_id == 1:
            return "Goalkeeper"
        elif cls_id == 3:
            return "Referee"
        
        # Default = Player (cls_id == 2 or unknown)
        return "Player"

    def get_jersey_num_color(self, jersey_num):
        # Find dominant color among tracks bound to this jersey
        colors = []
        for tid, j in self.active_bindings.items():
            if j == jersey_num and tid in self.track_colors:
                colors.append(self.track_colors[tid])
        if not colors: return "Unknown"
        return Counter(colors).most_common(1)[0][0]

    def finalize_bindings(self):
        """
        Phase v27.2: Bayesian Tracklet Consolidation (Refined)
        Retroactively link tracklets that didn't reach the lock threshold.
        """
        log("🔍 [IdentityManager] Starting Bayesian Tracklet Consolidation...")
        consolidated_count = 0
        
        # We iterate over all tracks that have some Dirichlet evidence
        for tid in list(self.alpha.keys()):
            # If this track is already bound, skip it
            if tid in self.active_bindings:
                continue
                
            # Find the number with the most evidence
            track_alphas = self.alpha[tid]
            if not track_alphas:
                continue
                
            best_number = max(track_alphas, key=track_alphas.get)
            evidence = track_alphas[best_number]
            
            # Threshold: 
            # 1. Evidence > 0.5 AND jersey was confirmed/locked by another track
            # 2. OR Evidence > 1.0 (Cold Start Fix - allow new players to appear)
            if (evidence > 0.5 and best_number in self.jersey_registry) or (evidence > 1.0):
                log(f"🔄 [Finalize] Consolidating Track {tid} -> Jersey #{best_number} (Evidence: {evidence:.1f})")
                self.active_bindings[tid] = best_number
                
                # Mapping Fix: Ensure track_map points to the Jersey Number for propagation
                self.track_map[tid] = best_number
                consolidated_count += 1
                
        log(f"✅ [Finalize] Consolidated {consolidated_count} fragmented tracklets.")

# --- 6. SUPER-RESOLUTION & SIGNAL MAXIMIZATION ---
def get_upsampler(model_path="models/EDSR_x4.pb"):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4)
    return sr

# Torso Crop: Top 50% (Legs removed) - Fallback when no pose
def _torso_crop(img, xyxy):
    Himg, Wimg = img.shape[:2]
    # Phase v26.9: Safe Body with 10% Padding (Prevents edge artifacts)
    Himg, Wimg = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    H = y2 - y1; W = x2 - x1
    if H <= 0 or W <= 0: return None
    
    pad_h = int(0.10 * H); pad_w = int(0.10 * W)
    yA = max(0, y1 - pad_h); yB = min(Himg, y2 + pad_h)
    xA = max(0, x1 - pad_w); xB = min(Wimg, x2 + pad_w)
    return img[int(yA):int(yB), int(xA):int(xB)]

def calculate_laplacian_variance(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

# --- PHASE 112: mkoshkina Framework Components ---

def is_legible(crop, min_height=30, min_width=20, blur_thresh=20):
    """
    Legibility Classifier (mkoshkina framework)
    Filters out crops that are too small, too blurry, or low contrast.
    """
    if crop is None or crop.size == 0:
        return False
    
    h, w = crop.shape[:2]
    
    # Size filter - too small to read
    if h < min_height or w < min_width:
        return False
    
    # Blur filter - Laplacian variance
    if calculate_laplacian_variance(crop) < blur_thresh:
        return False
    
    # Edge density filter (optional) - checks if there's enough structure
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < 0.02:  # Less than 2% edges = probably uniform/no text
        return False
    
    return True

def pose_torso_crop(img, keypoints, xyxy_fallback):
    """
    Pose-based Torso Crop (mkoshkina framework)
    Uses pose keypoints (shoulders, hips) to crop precise torso region.
    Falls back to fixed crop if keypoints not available.
    
    keypoints format: [[x, y, conf], ...] for 17 COCO keypoints
    - 5: left_shoulder, 6: right_shoulder
    - 11: left_hip, 12: right_hip
    """
    if keypoints is None or len(keypoints) < 13:
        return _torso_crop(img, xyxy_fallback)
    
    try:
        # Extract shoulder and hip keypoints
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Check confidence - if low, use fallback
        min_conf = 0.3
        kps = [left_shoulder, right_shoulder, left_hip, right_hip]
        if any(kp[2] < min_conf for kp in kps):
            return _torso_crop(img, xyxy_fallback)
        
        # Calculate torso bounding box
        x_coords = [kp[0] for kp in kps]
        y_coords = [kp[1] for kp in kps]
        
        x1 = int(min(x_coords)) - 10
        y1 = int(min(y_coords)) - 10
        x2 = int(max(x_coords)) + 10
        y2 = int(max(y_coords)) + 10
        
        # Clamp to image bounds
        Himg, Wimg = img.shape[:2]
        x1 = max(0, x1); x2 = min(Wimg, x2)
        y1 = max(0, y1); y2 = min(Himg, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return _torso_crop(img, xyxy_fallback)
        
        return img[y1:y2, x1:x2]
    except Exception:
        return _torso_crop(img, xyxy_fallback)

# --- 7. JNR SERVICE (Real Qwen) ---
class JNRService:
    def __init__(self):
        # Determine Model Path
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        if CONFIG["env"].get("JNR_WEIGHTS"):
            model_path = CONFIG["env"]["JNR_WEIGHTS"]
            log(f"Initializing Real JNRService from Custom Weights: {model_path}")
        else:
            log(f"Initializing Real JNRService (Base Model): {model_path}")

        self.processor = AutoProcessor.from_pretrained(model_path)
        # Fix padding warning for decoder-only architecture
        self.processor.tokenizer.padding_side = 'left'
        
        # Load Model
        max_mem = {0: "60GB"} 
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_mem,
            )
        except Exception as e:
            log(f"FP16 Load Failed: {e}. Fallback to 4-bit...")
            from transformers import BitsAndBytesConfig
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                quantization_config=qc,
                device_map="auto",
            )
            
        # Init SuperRes (EDSR x4) - DISABLED for speed (Phase 130)
        self.sr = None
        log("Super-Resolution DISABLED (Phase 130 - Speed).")
        
        # Init PaddleOCR as fast pre-verifier - Phase 115
        # DISABLED per user request (Phase 116) - causing multiple IDs
        self.ocr = None
        self.ocr_enabled = False
        log("PaddleOCR DISABLED (Phase 116).")

        
    def _preprocess_enhanced(self, img_bgr, target_h=448):
        """Phase v26.11: Signal Maximization (LANCZOS4 + CLAHE)."""
        if img_bgr is None or img_bgr.size == 0: return None
        
        # 1. Force High Resolution
        h, w = img_bgr.shape[:2]
        ratio = target_h / h
        upscaled = cv2.resize(img_bgr, (int(w*ratio), target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. CLAHE (Contrast Enhancement)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def predict_batch(self, images, reference_crops=None):
        """
        Predict jersey numbers with Temporal Context (Video) and Multi-Scale Consensus.
        
        Args:
            images: List of BGR crops (original size) OR List of List[BGR crops] for temporal sequences.
            reference_crops: Gallery from IdentityManager
        """
        results = []
        if not images: return []
        
        # Phase v27 Detection: Is this a list of sequences (video) or single frames?
        is_temporal = isinstance(images[0], list) or isinstance(images[0], np.ndarray) and images[0].ndim == 4
        
        # 1. Pre-process: Generate Multi-Scale Variants for EACH image/sequence
        target_h, target_w = 448, 224 # Standardized dimensions for batching
        target_seq_len = 5             # Standardized length for video batching
        
        all_variants_bgr = [] 
        source_counts = []
        is_temporal_variant = [] # Track per variant
        
        for input_item in images:
            # Per-item temporal detection
            item_is_temporal = isinstance(input_item, list) or (isinstance(input_item, np.ndarray) and input_item.ndim == 4)
            variants = []
            
            if item_is_temporal:
                # Sequence of frames
                seq = list(input_item)
                # Ensure all frames are numpy arrays and pad
                seq = [f for f in seq if isinstance(f, np.ndarray)]
                if not seq: continue
                
                while len(seq) < target_seq_len:
                    seq.append(seq[-1].copy())
                if len(seq) > target_seq_len:
                    seq = seq[-target_seq_len:]
                
                # Variant A: Standard Resized
                v1 = [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4) for f in seq]
                variants.append(v1)
            else:
                # Single frame
                img_bgr = input_item
                if not isinstance(img_bgr, np.ndarray): continue
                
                # Variant A: Standard
                v1 = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                variants.append(v1)

            source_counts.append(len(variants))
            for v in variants:
                all_variants_bgr.append(v)
                is_temporal_variant.append(item_is_temporal)

        if not all_variants_bgr: return [{"number": None, "confidence": 0.0}] * len(images)

        all_inputs = []
        for idx, v in enumerate(all_variants_bgr):
            curr_is_temporal = is_temporal_variant[idx]
            content = []
            if curr_is_temporal:
                # Video input
                video_pil = []
                for f in v:
                    if isinstance(f, np.ndarray) and f.ndim == 3:
                        video_pil.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                    else:
                        log(f"⚠️ [JNRService] Skipping invalid frame in sequence: {type(f)}")
                
                if not video_pil:
                     results.append({"number": None, "confidence": 0.0})
                     continue
                content.append({
                    "type": "video", 
                    "video": video_pil,
                    "max_pixels": target_h * target_w,
                    "min_pixels": target_h * target_w
                })
                prompt = """Task: Analyze the player over this video sequence to identify the jersey number and color.
Step 1: Observe the player in each frame. Folds in the jersey may obscure digits in some frames but reveal them in others.
Step 2: Identify the digits. '4' has a sharp corner/diagonal; '6' has a rounded loop; '5' has a horizontal top and curved bottom.
Step 3: Synthesize a final consensus.
Return JSON: {"number": int|null, "color": str, "confidence": float}"""
            else:
                # Image input
                if not isinstance(v, np.ndarray) or v.ndim != 3:
                     log(f"⚠️ [JNRService] Invalid frame type/shape for cvtColor: {type(v)}")
                     results.append({"number": None, "confidence": 0.0})
                     continue
                pil_img = Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                content.append({
                    "type": "image", 
                    "image": pil_img,
                    "max_pixels": target_h * target_w,
                    "min_pixels": target_h * target_w
                })
                prompt = """Task: Identify the jersey number and color.
Digit Identification Rules:
- '4' has a sharp corner on the left.
- '6' has a rounded loop at the bottom.
Observe the digits carefully. Do not confuse 4 with 6.
Return JSON: {"number": int|null, "color": str, "confidence": float}"""
            
            content.append({"type": "text", "text": prompt})
            all_inputs.append([{"role": "user", "content": content}])
        
        # 4. Batch Inference
        try:
            texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in all_inputs]
            image_inputs, video_inputs = process_vision_info(all_inputs)
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128) # Increased for CoT space
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        except Exception as e:
            log(f"❌ Qwen Batch Inference Failed: {e}")
            return [{"number": None, "confidence": 0.0, "visibility": "error"}] * len(images)

        # 5. Multi-Scale Consensus
        idx = 0
        for count in source_counts:
            variant_results = []
            for _ in range(count):
                raw_text = output_texts[idx]
                res = self._parse_json_robust(raw_text)
                res["raw_text"] = raw_text 
                variant_results.append(res)
                idx += 1
            
            # Take consensus (weighted by confidence)
            consensus = self._apply_consensus(variant_results)
            results.append(consensus)
            
        return results

    def _parse_json_robust(self, text):
        try:
            clean = text.strip()
            if "```json" in clean: clean = clean.split("```json")[-1].split("```")[0]
            elif "```" in clean: clean = clean.split("```")[-1].split("```")[0]
            data = json.loads(clean)
            return {
                "number": data.get("number"), 
                "color": data.get("color"),
                "confidence": data.get("confidence", 0.0), 
                "visibility": data.get("visibility")
            }
        except:
             # Fallback: extract any digits
             digits = "".join(c for c in text if c.isdigit())
             if digits and len(digits) <= 2:
                 return {"number": int(digits), "color": None, "confidence": 0.4, "visibility": "uncertain"}
             return {"number": None, "color": None, "confidence": 0.0, "visibility": "error"}

    def _apply_consensus(self, variants):
        """Phase v26: Select the best result from multi-scale variants."""
        valid = [v for v in variants if v["number"] is not None]
        if not valid: return variants[0]
        
        # If all agree, return highest conf
        numbers = [v["number"] for v in valid]
        if len(set(numbers)) == 1:
            return max(valid, key=lambda x: x["confidence"])
        
        # Weighted voting for Number
        num_votes = defaultdict(float)
        color_votes = defaultdict(float)
        for v in valid:
            num_votes[v["number"]] += v["confidence"]
            if v.get("color"):
                color_votes[v["color"]] += v["confidence"]
        
        best_num = max(num_votes, key=num_votes.get)
        best_color = max(color_votes, key=color_votes.get) if color_votes else None
        
        # Find best variant for this number
        best_variant = max([v for v in valid if v["number"] == best_num], key=lambda x: x["confidence"])
        if best_color: best_variant["color"] = best_color
        return best_variant
    
    def _extract_ocr_number(self, result):
        """Extract jersey number from PaddleOCR result with robust parsing."""
        try:
            if not result or len(result) == 0:
                return None, 0.0
            
            best_number = None
            best_conf = 0.0
            
            for page in result:
                if page is None:
                    continue
                for line in page:
                    try:
                        # Handle different output formats
                        if isinstance(line, dict):
                            text = str(line.get('text', ''))
                            conf = float(line.get('score', 0.0))
                        elif isinstance(line, (list, tuple)) and len(line) >= 2:
                            text_data = line[1] if len(line) > 1 else line[0]
                            if isinstance(text_data, (list, tuple)) and len(text_data) >= 2:
                                text = str(text_data[0])
                                conf = float(text_data[1])
                            elif isinstance(text_data, dict):
                                text = str(text_data.get('text', ''))
                                conf = float(text_data.get('score', 0.0))
                            else:
                                continue
                        else:
                            continue
                        
                        # Extract digits only
                        digits = ''.join(c for c in text if c.isdigit())
                        if digits and len(digits) <= 2:
                            number = int(digits)
                            if 1 <= number <= 99 and conf > best_conf:
                                best_number = number
                                best_conf = conf
                    except:
                        continue
            
            return best_number, best_conf
        except:
            return None, 0.0

# --- 7.1 SmolVLM2 Service (Lightweight Alternative to Qwen) - Phase 193 Original ---
class SmolVLM2Service:
    """
    Lightweight VLM-based Jersey Number Recognition using SmolVLM2-2.2B.
    Phase 193 (Original):
    - Single-scale inference (384px)
    - LANCZOS4 + CLAHE + Sharpening + Denoising
    - Simple voting/stability logic
    - No EasyOCR, No Multi-scale
    """
    def __init__(self):
        from transformers import AutoProcessor, SmolVLMForConditionalGeneration
        
        model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        log(f"Initializing SmolVLM2Service (Phase 193 Original): {model_path}")
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            model_path,
            _attn_implementation="eager",
            device_map="auto",
        ).to(torch.float16)
        
        # Phase 193+: Single scale (High Res for small crops)
        self.target_height = 768
        
        # Voting state
        self.vote_history = {}
        self.history_size = 5
        self.min_consensus = 1  # Aggressive: trust even a single sighting if consistent
        
        # Phase 193: Greedy decoding
        self.do_sample = False
        self.temperature = 1.0
        
        # Phase 194 "Better Prompt" (Restored for stability)
        self.prompt = """Look at this football player's jersey number.
Count each digit you can see on the back or front:
- If you see digits '1' and '0' together, the number is 10
- If you see digits '2' and '4' together, the number is 24
- If only a single digit is visible, give that digit
- Be careful: '6' has a curved tail, '9' is inverted '6', '4' has closed top
Reply with ONLY the jersey number (1-99), nothing else."""
        
        log("SmolVLM2Service Phase 193+ (Aggressive High-Res) initialized.")
    
    def _preprocess(self, img_bgr):
        """Preprocess image at single scale (768px)."""
        if img_bgr is None or img_bgr.size == 0:
            return None
            
        h, w = img_bgr.shape[:2]
        scale = self.target_height / h
        new_w = max(1, int(w * scale))
        
        # LANCZOS4 upscaling
        upscaled = cv2.resize(img_bgr, (new_w, self.target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # CLAHE
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Unsharp Masking (Restored - it helps define edges)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
        
        return denoised
    
    def _apply_voting(self, track_id, prediction):
        """Simple temporal voting for stability."""
        if track_id not in self.vote_history:
            self.vote_history[track_id] = []
        
        if prediction is not None and isinstance(prediction, int) and 1 <= prediction <= 99:
            self.vote_history[track_id].append(prediction)
            if len(self.vote_history[track_id]) > self.history_size:
                self.vote_history[track_id] = self.vote_history[track_id][-self.history_size:]
        
        history = self.vote_history[track_id]
        if not history:
            return None, 0.0
        
        from collections import Counter
        counts = Counter(history)
        most_common = counts.most_common(1)[0]
        number, count = most_common
        
        if count >= self.min_consensus:
            return number, count / len(history)
        
        return None, 0.0
    
    def predict_batch(self, images, track_ids=None, reference_crops=None):
        """
        Standard batch prediction (Phase 193).
        """
        results = []
        if not images:
            return []
        
        if track_ids is None:
            track_ids = list(range(len(images)))
        
        for idx, img_bgr in enumerate(images):
            track_id = track_ids[idx] if idx < len(track_ids) else idx
            
            if img_bgr is None or img_bgr.size == 0:
                results.append({"number": None, "confidence": 0.0, "visibility": "invalid_crop"})
                continue
            
            # Preprocess
            processed = self._preprocess(img_bgr)
            if processed is None:
                results.append({"number": None, "confidence": 0.0, "visibility": "invalid_crop"})
                continue
            
            # VLM Inference
            final_number = None
            confidence = 0.0
            
            try:
                img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": self.prompt}
                        ]
                    }
                ]
                
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False
                    )
                
                output_text = self.processor.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Parse
                num = self._parse_number(output_text)
                
                # Vote
                final_number, confidence = self._apply_voting(track_id, num)
                
            except Exception as e:
                pass
            
            results.append({
                "number": final_number,
                "confidence": confidence,
                "visibility": "detected" if final_number else "uncertain",
                "source": "vlm"
            })
        
        return results
    
    def _parse_number(self, text):
        """Extract digits only from VLM output."""
        digits = ''.join(c for c in text if c.isdigit())
        if digits and len(digits) <= 2:
            try:
                num = int(digits)
                if 1 <= num <= 99:
                    return num
            except:
                pass
        return None

# --- 7.5. VISUALIZER ---
class Visualizer:
    def __init__(self, fps=25.0):
        self.fps = fps
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_hud(self, img, frame_data, id_manager):
        # Class mapping for display
        cls_names = {1: "GK", 2: "P", 3: "REF", 32: "BALL"}
        
        # Draw Boxes & IDs
        for box in frame_data["boxes"]:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            tid = box["id"]
            cls = box.get("cls", 0)
            
            # Ball Detection (Class 32 only - not 1 which is now Goalkeeper)
            if cls == 32:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(10, (x2 - x1) // 2)
                cv2.circle(img, (cx, cy), radius, (0, 255, 255), 3) # Yellow Circle
                cv2.putText(img, "BALL", (cx - 20, cy - radius - 5), self.font, 0.6, (0, 255, 255), 2)
                continue
            
            if tid is None: continue
            
            # Phase 125/v26: Handle Persistent Remapping & Jersey Display
            stable_id = id_manager.track_map.get(tid, tid)
            # Use active_bindings to get the RAW number (Phase v26.10)
            jersey_num = id_manager.active_bindings.get(stable_id)

            # Color based on class
            if cls == 1:  # Goalkeeper
                color = (255, 165, 0)  # Orange
            elif cls == 3:  # Referee
                color = (0, 0, 0)  # Black
            else:  # Player
                color = (128, 255, 0)  # Lime Green
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label: Class | Stable ID | Jersey
            cls_name = cls_names.get(cls, f"C{cls}")
            label = f"{cls_name}|ID:{stable_id}"
            if jersey_num:
                label += f"|#{jersey_num}"
                # Background for jersey text
                cv2.rectangle(img, (x1, y1-20), (x2, y1), color, -1)
                
            cv2.putText(img, label, (x1, y1-5), self.font, 0.5, (255, 255, 255), 1)
            
        return img

# --- 8. EVENT DETECTOR (Possession & xG) ---
class EventDetector:
    def __init__(self, pitch_manager):
        self.pitch = pitch_manager
        
    def process(self, all_frames, stats_engine, id_manager):
        # 1. Identify Ball & Possession per Frame
        possession_chain = [] # [(frame_idx, player_id, team_color, x, y), ...]
        
        for f_idx, frame in enumerate(all_frames):
            # Find Ball (Class 32 only - NOT 1 which is Goalkeeper)
            ball = None
            for box in frame["boxes"]:
                if box["cls"] == 32:  # Ball only
                    ball = box
                    break
            
            if not ball: continue
            
            bx, by = (ball["xyxy"][0]+ball["xyxy"][2])/2, (ball["xyxy"][1]+ball["xyxy"][3])/2
            
            # Find Closest Player (classes 1=GK, 2=Player, 3=Referee)
            min_dist = float("inf")
            closest_pid = None
            closest_team = None
            
            for box in frame["boxes"]:
                # Include GK (1), Players (2), and even Referee (3) for possession
                if box["cls"] in [1, 2, 3] and box["id"] is not None:
                    px, py = (box["xyxy"][0]+box["xyxy"][2])/2, (box["xyxy"][1]+box["xyxy"][3])/2
                    dist = math.hypot(px-bx, py-by)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pid = box["id"]
                        
            # Threshold (e.g. 50 pixels or calibrated meters)
            # Assuming ~50px for now purely heuristic if no calibration
            if min_dist < 80 and closest_pid is not None:
                team = id_manager.track_colors.get(closest_pid, "Unknown")
                possession_chain.append((f_idx, closest_pid, team, bx, by))
                
        # 2. Analyze Chain for Events
        if not possession_chain: return

        # a. Dribbles & Touches
        for i in range(len(possession_chain)):
            frame, pid, team, bx, by = possession_chain[i]
            # Log Touch
            # stats_engine.update(pid, 'Touch', {}) 
            
        # b. Passes (Change of PID, Same Team)
        # Look for gaps or switches
        last_pid = possession_chain[0][1]
        last_team = possession_chain[0][2]
        
        for i in range(1, len(possession_chain)):
            frame, pid, team, bx, by = possession_chain[i]
            
            if pid != last_pid:
                # Switch happened
                if team == last_team:
                    # Pass!
                    stats_engine.update(last_pid, 'Pass', {'result': 'Complete', 'subtype': 'Simple'})
                else:
                    # Turnover / Tackle?
                    stats_engine.update(last_pid, 'Pass', {'result': 'Interception'}) 
                    stats_engine.update(pid, 'Interception', {})
                
                last_pid = pid
                last_team = team

        # c. Geometric xG (Simplistic: Is it a Shot?)
        # Only feasible if we detect "Goal" regions. skipping for now to be safe,
        # but the structure works.

# --- 9. STATS ENGINE ADAPTER ---
class StatsAdapter:
    def __init__(self, camera=None, pitch_manager=None):
        self.camera = camera
        # Initialize new engine (Patched to point to post_processor.py logic)
        from stats.metrics import StatsEngine 
        self.engine = StatsEngine() 
        # Note: EventDetector in StatsEngine creates its own Camera. 
        # We assume that is sufficient as it uses the same Homography logic.

    def process_events(self, all_frames, id_manager=None):
        # Delegate to new engine
        # returns (formatted_stats, events)
        formatted_stats, events = self.engine.process_events(all_frames, id_manager)
        
        # Return in order expected by pipeline: raw_tracks, player_stats
        return events, formatted_stats


        # Generate RAW TRACKS for Post-Processing Merge
        raw_tracks = []
        
        # Get frame height from image data (for goalkeeper zone detection)
        frame_h = 1080  # Default, will be updated from actual frame data
        if all_frames and "img_shape" in all_frames[0]:
            frame_h = all_frames[0]["img_shape"][0]
        
        # 1. Group frames by ID
        track_history = defaultdict(list) # {track_id: [(frame_idx, cx, cy), ...]}
        
        for f_idx, frame in enumerate(all_frames):
            for box in frame["boxes"]:
                tid = box["id"]
                if tid is None: continue
                cx = (box["xyxy"][0] + box["xyxy"][2]) / 2
                cy = (box["xyxy"][3]) 
                track_history[tid].append((f_idx, cx, cy))

        # 2. Compute Stats per Track (Using new Engine schema)
        for tid, points in track_history.items():
            # Get last known y position for goalkeeper zone detection
            last_y = points[-1][2] if points else None
            role = id_manager.get_role(tid, y_pos=last_y, frame_height=frame_h)
            
            # EXCLUDE REFEREES
            if role == "Referee":
                continue
                
            jersey_num = id_manager.active_bindings.get(tid)
            team_color = id_manager.track_colors.get(tid, "Unknown")
            
            # Initialize metrics for this track (using dummy ID for now)
            self.engine.initialize_player(tid)
            s = self.engine.stats[tid]
            
            # Distance Calculation
            # ... (Existing distance logic)
            total_dist = 0.0
            for i in range(1, len(points)):
                p1 = points[i-1][1:] # cx, cy
                p2 = points[i][1:]
                if self.camera:
                    dist = self.camera.calculate_distance(p1, p2)
                else:
                    dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) * 0.01
                total_dist += dist
            
            # Populate Schema
            s['total_distance'] = round(total_dist, 2)
            s['frame_count'] = len(points)
            s['role'] = role # NEW: Track Role in Output
            
            # Append RAW TRACK with full schema
            raw_tracks.append({
                "track_id": tid,
                "jersey_number": jersey_num,
                "team": team_color,
                "role": role, # NEW
                "frames": [p[0] for p in points],
                "stats": s # Full granular stats
            })
        
        # Build player_stats dict grouped by jersey number (or track_id for GK)
        player_stats = {}
        for track in raw_tracks:
            jnum = track["jersey_number"]
            role = track["stats"].get("role", "Player")
            
            # Use jersey_number if available, else use color for GK
            if jnum is not None:
                key = str(jnum)
                display_number = jnum
            elif role == "Goalkeeper":
                # GK without jersey number - merge by color (e.g., GK_Red)
                color = track["team"]
                key = f"GK_{color}"
                display_number = None
            else:
                # Skip players without jersey numbers
                continue
            
            if key not in player_stats:
                player_stats[key] = {
                    "jersey_number": display_number,
                    "team": track["team"],
                    "stats": dict(track["stats"])  # Copy stats
                }
            
        return raw_tracks, player_stats 

# --- 9. MAIN EXECUTION ---
if __name__ == "__main__":
    import argparse
    import tempfile
    import requests
    import json # Added for JSON output
    
    parser = argparse.ArgumentParser(description="Football Analysis Pipeline (Consolidated)")
    parser.add_argument("--video", type=str, help="Video path (local file or SPACES URL)")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--no_video_output", action="store_true", help="Skip video output generation")
    parser.add_argument("--max_frames", type=int, help="Limit number of frames to process")
    parser.add_argument("--locking_mode", type=int, choices=[1, 2, 3], default=3, help="Locking mode: 1=Instant, 2=Consecutive High Conf, 3=Bayesian Dirichlet")
    parser.add_argument("--jnr_stride", type=int, help="JNR processing stride (frames)")
    parser.add_argument("--vid_stride", type=int, default=1, help="Video frame stride (skip frames). Default=1 (process all). 2=half speed/2x faster.")
    args = parser.parse_args()
    
    # Determine video path
    video_path = args.video or os.environ.get("PIPELINE_VIDEO") or "/home/ubuntu/football/121364_0.mp4"
    output_dir = args.output_dir or os.environ.get("PIPELINE_OUTPUT") or "output"
    
    # Handle URL streaming/downloads
    temp_video_path = None
    if video_path.startswith("http"):
        # Attempt to stream first (Phase 154)
        log(f"Verifying stream: {video_path[:50]}...")
        cap = cv2.VideoCapture(video_path)
        stream_capable = cap.isOpened()
        # if cap.isOpened():
        #     ret, _ = cap.read() # Removing frame consumption
        #     if ret: stream_capable = True
        cap.release()

        if stream_capable:
            log(f"Streaming directly from URL.")
        else:
            log(f"Stream unstable. Downloading video to local temp...")
            temp_dir = tempfile.mkdtemp(prefix="football_")
            temp_video_path = os.path.join(temp_dir, "video.mp4")
            
            try:
                resp = requests.get(video_path, stream=True, timeout=600)
                resp.raise_for_status()
                with open(temp_video_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                video_path = temp_video_path
                log(f"Downloaded to {temp_video_path}")
            except Exception as e:
                log(f"Download failed: {e}")
                sys.exit(1)
    
    if not video_path.startswith("http") and not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found.")
        sys.exit(1)

    log(f"Starting Phase 85 Pipeline on {video_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase v27: Temporal Buffer
    track_history = defaultdict(lambda: deque(maxlen=5)) 
    # Init Components
    id_manager = IdentityManager()
    # Phase 196 Revert: Switch back to Qwen (JNRService) as requested
    jnr_service = JNRService()
    # jnr_service = SmolVLM2Service()  # Original Qwen-based
    # jnr_service = SmolVLM2Service()  # Lighter SmolVLM2-2.2B with preprocessing
    visualizer = Visualizer()
    color_classifier = TeamColorClassifier()  # Phase 139
    kit_coordinator = KitCoordinator()  # Phase 168
    pitch_manager = PitchManager(model_path="/home/ubuntu/videoforprocessing_link/football_analysis_v2/data/models/best_field_keypoint.pt", device=0)
    camera = Camera(pitch_manager.H_default)
    
    # Tracking
    player_model = YOLO("/home/ubuntu/videoforprocessing_link/football_analysis_v2/data/models/best_player_detect.pt")
    ball_model = YOLO("/home/ubuntu/videoforprocessing_link/football_analysis_v2/data/models/best_ball_latest.pt")
    loader = ThreadedVideoReader(video_path)
    time.sleep(1.0)
    
    # Video Writer (conditional)
    out_video_path = os.path.join(output_dir, "output_video.mp4")
    fps = loader.stream.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(loader.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(loader.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if args.no_video_output:
        writer = None
        log("Video output DISABLED (--no_video_output)")
    else:
        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # OPTIMIZATION v25: Default JNR stride to 3 seconds (was 1s) to reduce Qwen calls
    jnr_stride = args.jnr_stride if args.jnr_stride is not None else int(fps * 3)
    log(f"JNR Stride set to {jnr_stride} (approx once every {jnr_stride/fps:.1f} seconds)")
    
    # Configure ID Manager
    id_manager.set_locking_mode(args.locking_mode)
    
    all_frames = []
    best_crops = {} # {track_id: crop_image}
    MAX_FRAMES = args.max_frames if args.max_frames is not None else 1000000 
    
    n = 0
    start_time = time.time()
    
    try:
        while loader.more() and n < MAX_FRAMES:
            f = loader.read()
            if f is None: break
            n += 1
            
            # OPTIMIZATION v25: Skip frames based on vid_stride
            if args.vid_stride > 1 and n % args.vid_stride != 0:
                continue
            
            if n % 10 == 0: log(f"Processing Frame {n}...")
            
            try:
                # Track
                player_res = player_model.track(f, persist=True, tracker="botsort.yaml", verbose=False, device=0)[0]
                ball_res = ball_model.track(f, persist=True, tracker="botsort.yaml", verbose=False, device=0)[0]
                img = player_res.orig_img

                # Pitch Calib (Every 60 frames)
                if n % 60 == 0:
                     kps, H_new = pitch_manager.predict(f)
                     camera.update(H_new)
                
                frame_data = {"boxes": []}
                batch_crops = []
                batch_ids = []
            
                # Process Player Detections (includes goalkeeper, player, referee)
                # Model classes: 0=ball, 1=goalkeeper, 2=player, 3=referee
                if hasattr(player_res, "boxes"):
                    for b in player_res.boxes:
                        cls_id = int(b.cls[0].item())
                        # Skip ball detections from player model (if any)
                        if cls_id == 0:
                            continue
                        frame_data["boxes"].append({
                            "xyxy": b.xyxy[0].cpu().numpy().tolist(),
                            "id": int(b.id[0].item()) if b.id is not None else None,
                            "conf": float(b.conf[0].item()),
                            "cls": cls_id  # Use actual class: 1=GK, 2=Player, 3=Referee
                        })

                # Process Ball Detections
                if hasattr(ball_res, "boxes"):
                    for b in ball_res.boxes:
                        conf = float(b.conf[0].item())
                        # FILTER: Lower confidence for ball to catch distant/small balls
                        if conf < 0.3:
                            continue
                        frame_data["boxes"].append({
                            "xyxy": b.xyxy[0].cpu().numpy().tolist(),
                            "id": None, # Balls usually don't track well with ID
                            "conf": conf,
                            "cls": 32 # Force Class 32 (Standard Ball) for EventDetector compatibility
                        })
                
                # Post-Process for JNR
                for box_data in frame_data["boxes"]:
                        # Process GK (1), Player (2) - Skip Referee (3) and Ball (32)
                        if box_data["id"] is not None and box_data["cls"] in [1, 2]:
                            tid = box_data["id"]
                            cls_id = box_data["cls"]
                            
                            # Store detection class (Phase 132)
                            id_manager.set_track_class(tid, cls_id)
                            
                            # Phase 112: mkoshkina Framework - Torso Crop
                            crop = _torso_crop(img, box_data["xyxy"])
                            if crop is not None and crop.size > 0:
                                   # Store Color with voting (Phase 139)
                                   color = color_classifier.predict_with_voting(crop, tid)
                                   id_manager.set_track_color(tid, color)
                                   
                                   # Phase 168: Global Kit Discovery
                                   kit_coordinator.observe(cls_id, color)
                                   
                                   # Skip JNR for Goalkeepers (class 1) - only need color
                                   if cls_id == 1:
                                       continue
                                   
                                   # OPTIMIZATION: Skip JNR if ID is already locked!
                                   if tid in id_manager.active_bindings:
                                       continue
                                   
                                   # Capture Best Crop (Largest Area)
                                   current_area = crop.shape[0] * crop.shape[1]
                                   if tid not in best_crops or current_area > (best_crops[tid].shape[0] * best_crops[tid].shape[1]):
                                       best_crops[tid] = crop.copy()

                                   # Phase v27: Update Temporal History
                                   track_history[tid].append(crop.copy())

                                   # JNR Stride based on FPS (Phase 123)
                                   if n % jnr_stride == 0:
                                       if not is_legible(crop):
                                           log(f"DEBUG [JNR Skip]: Track {tid} not legible in frame {n}")
                                           continue
                                       
                                       # Phase v27: Send the sequence of frames (video context)
                                       seq = list(track_history[tid])
                                       log(f"DEBUG [JNR Trigger]: Track {tid} in frame {n} (Seq len: {len(seq)})")
                                       batch_crops.append(seq)
                                       batch_ids.append(box_data["id"])
            
                # JNR Inference (Phase v27: Temporal Context)
                if batch_crops:
                    log(f"🚀 [JNR Batch] Sending {len(batch_crops)} sequences to Qwen...")
                    predictions = jnr_service.predict_batch(batch_crops, reference_crops=id_manager.jersey_gallery)
                    for idx, pred in enumerate(predictions):
                        # NEW DEBUG LOG (Phase v26.2)
                        if pred.get("raw_text"):
                            log(f"VLM RAW [{batch_ids[idx]}]: {pred['raw_text']}")
                        if pred["number"] is not None:
                            # 1. Resolve Identity (Phase 104)
                            track_id = batch_ids[idx]
                            team_color = id_manager.get_track_color(track_id)
                            stable_id = id_manager.resolve_identity(track_id, pred["number"], team_color)
                            
                            log(f"DEBUG: Track {track_id} (Resolved: {stable_id}) => {pred['number']} (Color: {pred.get('color')}, Conf: {pred['confidence']})")
                            # Phase v26: Pass the high-conf crop to IdentityManager for gallery usage
                            current_crop = batch_crops[idx]
                            id_manager.process_detection(stable_id, pred["number"], "auto", pred["confidence"], detected_color=pred.get("color"))
                        
                        # Phase v27: Respect Mode 3 Consensus (Don't force lock in main loop)
                        if id_manager.locking_mode != 3 and pred["confidence"] >= 0.8:
                            id_manager._lock_identity(stable_id, pred["number"], reference_crop=current_crop)
                        
                        # 2. CRITICAL: Overwrite the Frame Data immediately
                        # If we found a stable ID, we must update the current frame so Stats see the right person.
                        if track_id != stable_id:
                             for box in frame_data["boxes"]:
                                 if box["id"] == track_id:
                                     box["id"] = stable_id
                                     # Also update color for the new ID just in case
                                     id_manager.set_track_color(stable_id, team_color)
                                     break

                # 3. Apply Remapping to Stats (HUD already handles visualization)
                for box in frame_data["boxes"]:
                    tid = box["id"]
                    if tid is not None:
                        # Update the box ID to the stable one for stats tracking downstream
                        box["id"] = id_manager.track_map.get(tid, tid)

            except Exception as e:
                import traceback
                log(f"💥 Frame-level Pipeline Error (Frame {n}): {e}")
                traceback.print_exc()
                # Continue processing next frame instead of crashing whole loop
                continue
    
            # Visualization
            annotated_img = visualizer.draw_hud(img.copy(), frame_data, id_manager)
            if writer:
                writer.write(annotated_img)
            all_frames.append(frame_data)

    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        loader.stop()
        if writer:
            writer.release()
            log(f"Video saved to {out_video_path}")
        log(f"Tracking finished in {time.time() - start_time:.2f}s.")

    # Phase v27.2: Bayesian Tracklet Consolidation
    # Perform this BEFORE stats and propagation
    id_manager.finalize_bindings()
    
    # Save Best Crops - ENABLED for evaluation (Phase 196)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    for tid, crop in best_crops.items():
        if crop is not None:
            cv2.imwrite(os.path.join(crops_dir, f"{tid}.jpg"), crop)
    log(f"Saved {len(best_crops)} track crops to {crops_dir}/")
    
    # Phase 170: Retroactive Identity Propagation (Fix for Zero Stats)
    # Apply final known identities to ALL historical frames to recover stats from before identification.
    log("Applying Retroactive Identity Propagation...")
    start_prop = time.time()
    updates_count = 0
    
    # Iterate ALL frames and update IDs based on final accumulated bindings
    for frame in all_frames:
        for box in frame["boxes"]:
            tid = box["id"]
            if tid is not None:
                # If this Track ID was eventually bound to a Jersey Number, use it!
                if tid in id_manager.active_bindings:
                     new_id = id_manager.active_bindings[tid]
                     if new_id != tid:
                         box["id"] = new_id
                         updates_count += 1
                         
    log(f"Propagated {updates_count} identity updates in {time.time() - start_prop:.2f}s")

    # --- 9. STATS GENERATION (Entity Resolution) ---
    stats_adapter = StatsAdapter(camera, pitch_manager) # Pass pitch_manager
    raw_tracks, player_stats = stats_adapter.process_events(all_frames, id_manager)
    
    # Save Raw Tracks
    with open(os.path.join(output_dir, "raw_tracks.json"), "w") as f:
        json.dump(raw_tracks, f, indent=2)
    log(f"Saved {output_dir}/raw_tracks.json")
    
    # Phase 186: Filter out Unknown players before saving
    # Drop if key starts with "Unknown" AND jersey_number is None
    original_count = len(player_stats)
    filtered_stats = {
        pid: pdata for pid, pdata in player_stats.items()
        if not (str(pid).startswith("Unknown") and pdata.get("jersey_number") is None)
    }
    player_stats = filtered_stats
    log(f"Filtered Unknown players: {original_count} -> {len(player_stats)}")
    
    # Save Player Stats
    with open(os.path.join(output_dir, "player_stats.json"), "w") as f:
        json.dump(player_stats, f, indent=2)
    log(f"Saved {output_dir}/player_stats.json")
    
    # Phase 168: Save Discovered Kits
    kits = kit_coordinator.get_discovery_result()
    with open(os.path.join(output_dir, "match_kits.json"), "w") as f:
        json.dump(kits, f, indent=2)
    log(f"Match Kits saved to match_kits.json: {kits}")
    
    # Phase 169: Color Reconciliation
    # Enforce only discovered colors in player_stats
    valid_colors = set(kits["goalkeepers"] + kits["players"])
    log(f"Reconciling colors against valid set: {valid_colors}")
    
    reconciled_count = 0
    for pid, pdata in player_stats.items():
        original_color = pdata.get("team", "Unknown")
        if original_color not in valid_colors:
            pdata["team"] = "Unknown"
            reconciled_count += 1
            
    if reconciled_count > 0:
        with open(os.path.join(output_dir, "player_stats.json"), "w") as f:
            json.dump(player_stats, f, indent=2)
        log(f"Reconciled {reconciled_count} players to 'Unknown' color.")
    
    # Run Entity Resolution Script
    # DISABLED per user request (Phase 116) - not needed anymore
    log("Entity Resolution SKIPPED (Phase 116).")
    
    # Identify Players (Debug)
    log(f"Identified Players: {list(player_stats.keys())}")
    
    print("\n--- Validation Results ---")
    print(f"Stats Saved to {output_dir}/player_stats.json.")
