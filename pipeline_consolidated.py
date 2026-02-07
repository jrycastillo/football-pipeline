
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
from vision.resnet_recognition import ResNetRecognizerV2 as JNRService

# FORCE HF CACHE to Local Directory to avoid Permission Errors
# os.environ["HF_HOME"] = "/home/ubuntu/football/hf_cache" # Removed hardcoded path
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from utils.device_utils import get_device, empty_cache

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

def safe_crop(img, box):
    """Safely crop image with bounds checking."""
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return None
    return img[y1:y2, x1:x2]


import inspect
import inspect
# from ultralytics.trackers.byte_tracker import BYTETracker # Original
from vision.custom_bytetrack import BYTETracker # Custom with Rescue
from collections import namedtuple

# Wrapper for clean ByteTrack
class ByteTrackTracker(BYTETracker):
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        
    def update(self, results, img=None):
        # Adapter to match Ultralytics tracker update signature
        # results: can be direct det boxes or ultralytics result object
        # but BYTETracker.update expects: results (preds), img
        return super().update(results, img)

# START MOCK CLASSES (For explicit control)
class MockBox:
    def __init__(self, xyxy, tid, conf, cls_id):
        d = get_device()
        self.xyxy = [torch.tensor(xyxy, dtype=torch.float32).to(d)]
        self.id = [torch.tensor([tid], dtype=torch.float32).to(d)]
        self.conf = [torch.tensor([conf], dtype=torch.float32).to(d)]
        self.cls = [torch.tensor([cls_id], dtype=torch.float32).to(d)]

class MockResults:
    def __init__(self, boxes, img):
        self.boxes = boxes
        self.orig_img = img
# END MOCK CLASSES

# Try to import custom_botsort to check its path
try:
    import vision.custom_botsort as cb
    from vision.custom_botsort import JerseyBoTSORT
    logging.warning(f"[DEBUG] custom_botsort.py path = {cb.__file__}")
except ImportError:
    logging.warning("[DEBUG] vision.custom_botsort not found - BoT-SORT will be unavailable")
    JerseyBoTSORT = None

logging.warning(f"[DEBUG] pipeline_consolidated.py path = {os.path.abspath(__file__)}")

def build_tracker(tracker_name, fps, enable_reid, cfg):
    name = tracker_name.lower().strip()
    if name == "bytetrack":
        logging.warning("🛡️ [Tracker] Using ByteTrack (NO ReID)")
        return ByteTrackTracker(cfg, frame_rate=fps)
    elif name == "botsort":
        if JerseyBoTSORT is None:
            raise ImportError("JerseyBoTSORT not found (check vision/custom_botsort.py)")
        logging.warning("🛡️ [Tracker] Using BoT-SORT")
        # Passing enable_reid via config or arg if supported
        # JerseyBoTSORT usually takes args object. We'll ensure cfg has what it needs.
        return JerseyBoTSORT(cfg, frame_rate=fps)
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")

import cv2
import numpy as np
import torch
import math
from collections import defaultdict, Counter, deque
from stats.metrics import StatsEngine # Import new engine
from vision.color_classifier import TeamColorClassifier, KitCoordinator  # Phase 139
from ultralytics import YOLO
from vision.resnet_recognition import ResNetRecognizerV2 as JNRService # ResNet32 only
from PIL import Image
from vision.track_utils import calculate_iou

def is_near_feet(ball_box, player_boxes):
    """
    Check if ball is near player feet using Distance Heuristic.
    ball_box: [x1, y1, x2, y2]
    player_boxes: list of [x1, y1, x2, y2]
    """
    bx1, by1, bx2, by2 = ball_box
    b_cx = (bx1 + bx2) / 2
    b_cy = (by1 + by2) / 2
    
    for pbox in player_boxes:
        px1, py1, px2, py2 = pbox
        p_w = px2 - px1
        p_h = py2 - py1
        
        # Player Anchor: Bottom Center
        p_anchor_x = (px1 + px2) / 2
        p_anchor_y = py2
        
        # Distance from Ball Center to Player Anchor
        dist = math.sqrt((b_cx - p_anchor_x)**2 + (b_cy - p_anchor_y)**2)
        
        # Dynamic Threshold: 
        # Radius = Max(90% of Width, 25% of Height)
        # This creates a semi-circle zone around the feet
        thresh = max(p_w * 0.9, p_h * 0.25)
        
        if dist < thresh:
             return True
            
    return False

def filter_detections_strict(boxes_obj, width, height, frame_idx):
    """
    Remove detections that are:
    1. Too close to image edges (Partial bodies)
    2. Malformed Aspect Ratio (Too thin/wide)
    3. Invalid coordinates
    """
    if boxes_obj is None or len(boxes_obj) == 0:
        return boxes_obj
        
    valid_indices = []
    # Tuned V2: Stricter Margin, Wider AR
    MARGIN = 30 # px
    MIN_AR = 0.2
    MAX_AR = 4.0 
    
    removed_count = 0
    
    for i, box in enumerate(boxes_obj.xyxy):
        x1, y1, x2, y2 = box.cpu().numpy()
        
        # 0. Sanity Check
        if x2 <= x1 or y2 <= y1:
            continue
            
        # 1. Edge Filter
        if x1 < MARGIN or y1 < MARGIN or x2 > (width - MARGIN) or y2 > (height - MARGIN):
            removed_count += 1
            continue
            
        # 2. Aspect Ratio Filter
        w = x2 - x1
        h = y2 - y1
        ar = w / h
        if ar < MIN_AR or ar > MAX_AR:
            removed_count += 1
            continue
            
        valid_indices.append(i)
        
    if removed_count > 0 and frame_idx % 30 == 0:
        print(f"✂️ [Filter] Removed {removed_count} detections at Frame {frame_idx}")
        
    if len(valid_indices) == len(boxes_obj):
        return boxes_obj
        
    return boxes_obj[valid_indices]

def prevent_ghost_spawns(online_targets, all_tracks, frame_idx):
    """
    Reject NEW tracks that significantly overlap with EXISTING stable tracks.
    """
    valid_targets = []
    
    # 1. Separate New vs Stable
    new_tracks = []
    stable_tracks = []
    
    for t in online_targets:
        if hasattr(t, 'track_id'):
            tid = int(t.track_id)
            tlbr = t.tlbr
            # STrack uses start_frame
            start_frame = getattr(t, 'start_frame', frame_idx)
        else:
            # Numpy fallback
            t_list = t.tolist() if hasattr(t, 'tolist') else t
            tid = int(t_list[4])
            tlbr = t_list[:4]
            start_frame = frame_idx # Assume new if numpy?
            
        age = frame_idx - start_frame
        
        # Tuned V2: Treat anything < 5 frames as "New" / Unstable
        if age <= 5:
            new_tracks.append((t, tlbr, tid))
        else:
            stable_tracks.append((t, tlbr, tid))
            valid_targets.append(t) # Keep stable
            
    # 2. Check Overlaps
    # If a NEW track overlaps a STABLE track > IoU 0.15, KILL IT.
    killed_count = 0
    for new_t, new_box, new_tid in new_tracks:
        is_ghost = False
        for stable_t, stable_box, stable_tid in stable_tracks:
            iou = calculate_iou(new_box, stable_box)
            if iou > 0.15: # Tuned V2: Stricter (0.2 -> 0.15)
                is_ghost = True
                if frame_idx % 30 == 0:
                    print(f"👻 [Ghost] Killed NEW Track {new_tid} (Overlap {iou:.2f} with Stable {stable_tid})")
                break
        
        if not is_ghost:
            valid_targets.append(new_t)
        else:
            killed_count += 1
            
    return valid_targets


def bytetrack_update_with_stale_guard(tracker, dets, img,
                                      stale_frames: int,
                                      hard_frames: int):
    """
    Prevent very old LOST tracks from matching again (ID hijack),
    while still allowing long occlusion recovery up to hard_frames.
    Works by temporarily removing stale lost tracks from association.
    """

    # Split lost tracks into "recent" vs "stale"
    lost_all = getattr(tracker, "lost_stracks", [])
    recent_lost = []
    stale_lost = []

    for t in lost_all:
        age = getattr(t, "time_since_update", None)
        
        if age is None:
             if hasattr(tracker, 'frame_id') and hasattr(t, 'frame_id'):
                 age = tracker.frame_id - t.frame_id
        
        if age is None:
            recent_lost.append(t)
            continue

        if age <= stale_frames:
            recent_lost.append(t)
        else:
            stale_lost.append(t)

    tracker.lost_stracks = recent_lost
    outputs = tracker.update(dets, img=img)

    if stale_lost:
        tracker.lost_stracks.extend(stale_lost)

    tracker.lost_stracks = [
        t for t in tracker.lost_stracks
        if (getattr(t, "time_since_update", 0) <= hard_frames)
    ]

    return outputs

# --- 1. CONFIGURATION ---
try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    log("Warning: config.yaml not found, using defaults.")
    CONFIG = {
        "env": {"DET_WEIGHTS": "models/yolo_player.pt", "JNR_WEIGHTS": None, "BALL_MODEL_PATH": None},
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
    except Exception as e:
        # Fallback to median if K-means fails
        log(f"Warning: K-means color clustering failed: {e}")
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
        self.color_counter = {}  # {color: count} - track PLAYER color frequencies only
        self.team_colors = []    # [Team A color, Team B color] - two most common player colors
        self.goalkeeper_colors = {}  # {track_id: color} - GK colors stored separately
        self.goalkeeper_zone_threshold = 0.15  # Top/bottom 15% of pitch = GK zone 
        self.locking_mode = 2 # Default to Mode 2 (Consecutive) for Precision
        self.jersey_gallery = defaultdict(list) # Phase v26: {jersey_num: [pil_image, ...]}
        self.track_map = {} # Phase v26: {raw_tid: stable_tid}
        self.track_boxes = {} # Phase v28: {track_id: [ymin, xmin, ymax, xmax]} (relative)
        # Phase v27: Bayesian Dirichlet Consensus
        self.alpha = defaultdict(lambda: defaultdict(lambda: 1.0)) # {track_id: {jersey_num: score}}
        self.visual_history = defaultdict(list)
        # Run 20 Attributes
        self.vote_counts = defaultdict(lambda: defaultdict(float)) # {track_id: {jersey: score}}
        self.locks = {} # {track_id: {jersey: X, locked: True}}
        self.last_jnr_update = {} # {track_id: frame_idx}
        self.locked_map = {} # Key: (team, number), Value: track_id -- Strict Uniqueness
        
        # Team colors will be auto-detected during processing (Phase 128)

    def set_track_class(self, track_id, cls_id):
        self.track_classes[track_id] = cls_id

    def set_track_color(self, track_id, color):
        self.track_colors[track_id] = color

    def set_locking_mode(self, mode):
        self.locking_mode = int(mode)
        log(f"⚙️ [IdentityManager] Locking Mode set to {self.locking_mode}")
        
    def get_player_color(self, jersey_num):
        return self.player_colors.get(str(jersey_num)) or self.player_colors.get(int(jersey_num))

    def is_jersey_number(self, val):
        return str(val) in self.jersey_registry or int(val) in self.jersey_registry if str(val).isdigit() else False

    def touch(self, track_id, frame_idx):
        self.last_seen[track_id] = frame_idx

    def try_lock(self, tid, team, num, score, track_scores):
        """
        Attempt to acquire a unique lock on (team, number).
        Returns True if successful, False if denied.
        """
        key = (team, num)
        
        # 1. Check if already owned by this track
        owner = self.locked_map.get(key)
        if owner == tid:
            return True
        
        # 2. If free, take it
        if owner is None:
            self.locked_map[key] = tid
            return True
            
        # 3. Conflict Resolution (Steal if significantly stronger)
        # Check owner's strength for this number
        owner_score = self.vote_counts[owner].get(num, 0.0)
        
        # Phase 216: DISABLED STEAL - Multiple tracks can claim same jersey
        # This ensures consistent visualization (e.g., both goalkeepers wear #1)
        # The STEAL mechanism was causing jersey numbers to disappear from video
        # when a new track claimed the same number
        if score > (owner_score + 3.0):
            log(f"⚔️ [IdentityManager] STEAL: Track {tid} (Score {score:.1f}) takes Jersey #{num} from Track {owner} (Score {owner_score:.1f})")
            
            # Phase 216: DO NOT unlock the loser - keep their binding for visualization
            # self.locks.pop(owner, None)
            # if self.active_bindings.get(owner) == num:
            #      del self.active_bindings[owner]
            
            # Transfer lock ownership (for future conflict resolution)
            self.locked_map[key] = tid
            return True
            
        # Denied
        # log(f"🔒 [IdentityManager] DENIED: Track {tid} wanted #{num} (Score {score:.1f}) but held by {owner} (Score {owner_score:.1f})")
        return False

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
            if score >= 0.15: # Aggressively LOWERED to 15% per user request (was 0.65)
                log(f"🔒 [IdentityManager] MODE 1 LOCK: Track {track_id} -> Jersey #{detected_number}")
                self._lock_identity(track_id, detected_number)
            return

        # --- MODE 2: Strict Voting & Locking (Run 20) ---
        if self.locking_mode == 2:
            # 1. Update Vote Counts (if confident)
            # User Request: Lower threshold to 0.50, but require 5 votes
            if score >= 0.50: 
                self.vote_counts[track_id][detected_number] += score
                # Track vote count (tally)
                if not hasattr(self, "vote_tallies"): self.vote_tallies = defaultdict(lambda: defaultdict(int))
                self.vote_tallies[track_id][detected_number] += 1
                
            # 2. Check for Lock Condition
            votes = self.vote_counts[track_id]
            if not votes: return

            # Find top 2 candidates
            sorted_candidates = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            best_num, best_score = sorted_candidates[0]
            second_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0.0
            
            # Get tally
            if not hasattr(self, "vote_tallies"): self.vote_tallies = defaultdict(lambda: defaultdict(int))
            best_tally = self.vote_tallies[track_id][best_num]
            
            # --- Phase 221: AGGRESSIVE SOFT REGISTRATION ---
            # Register jersey in registry after minimal observations
            # This allows StatsEngine to recognize it as a valid player faster
            if best_score >= 1.5 and best_num not in self.jersey_registry:
                team = self.track_colors.get(track_id, "Unknown")
                self.jersey_registry[best_num] = {"track_id": track_id, "team": team, "soft": True}
                log(f"📝 [IdentityManager] Soft-registered Jersey #{best_num} for Track {track_id} (Score: {best_score:.1f})")
            
            # Lock Rule (User Request V6):
            # Score >= 0.50 (Filter)
            # 3 Votes required (Faster locking)
            # Keep margin check for safety (>= 1.0)
            if best_tally >= 3 and (best_score - second_score) >= 1.0:
                 if track_id not in self.locks:
                     # Check Global Uniqueness Logic
                     team = self.track_colors.get(track_id, "Unknown")
                     if self.try_lock(track_id, team, best_num, best_score, self.vote_counts):
                         log(f"🔒 [IdentityManager] LOCKED Track {track_id} -> Jersey #{best_num} (Votes: {best_tally}, Score: {best_score:.1f}, Margin: {best_score-second_score:.1f})")
                         self._lock_identity(track_id, best_num)
                         self.locks[track_id] = {"jersey": best_num, "locked": True}
            
            # 3. Unlocking / Hysteresis (Only if locked)
            # User instruction: "never change unless you have overwhelming evidence"
            # Logic: If locked to A, but B is winning by HUGE margin (e.g. +8.0), maybe switch?
            # For now, stick to "Once Locked, Stay Locked" as per QA instruction ("don't unlock at all during QA").
            return

        # --- MODE 3: Bayesian Dirichlet Consensus (Legacy) ---
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
            # We know this jersey number exists.
            original_id, original_color = self.master_registry[player_key]
            
            # Use User Rule: "Do NOT use jersey number as the ID key for stitching unless LOCKED"
            # Actually, User said: "player_id = track_id always"
            # So we NEVER merge. We just acknowledge the jersey attribute.
            
            if current_track_id != original_id:
                # Just log, don't merge.
                pass
                
        else:
            # First time seeing this jersey. Register it.
            self.master_registry[player_key] = (current_track_id, team_color)
            # Make sure track_map points to self
            self.track_map[current_track_id] = current_track_id
            
            self.player_colors[player_key] = team_color
            
            # Also register in color_registry
            color_key = f"{team_color}_{current_track_id}"
            self.color_registry[color_key] = jersey_number
            log(f"📝 [IdentityManager] Registered Jersey #{jersey_number} to Track {current_track_id} ({self.get_team_label(current_track_id)})")

        # STRICT: Always return the tracker's ID
        return current_track_id
    
    
    def get_resolved_id(self, track_id):
        """
        Recursively resolve the track ID to its persistent root ID.
        NOTE: This returns TRACK IDs only, not jersey numbers.
        Jersey numbers are looked up separately in visualization and stats.
        """
        current = track_id
        visited = set()
        while current in self.track_map:
            if current in visited: break
            visited.add(current)
            mapped = self.track_map[current]
            if mapped == current: break
            current = mapped
        
        # Phase 216: REMOVED jersey number return - that was causing duplicate IDs
        # Jersey numbers should only be used in visualization, not in frame data
        return current


    def validate_visual_consistency(self, track_id, jersey_num, crop, classifier):
        """
        Check if the current crop matches the visual history of the track.
        Returns: True (consistent), False (inconsistent/rejected)
        """
        # 1. Retrieve history
        if track_id not in self.visual_history:
             return True, 1.0 # No history yet, assume consistent

        history = self.visual_history[track_id]
        if not history:
             return True, 1.0

    # --- Run 20 Helpers ---
    def suppress_conflicts(self, active_tracks, frame_idx):
        """
        Run 20.1: Enforce 1-to-1 mapping between Jersey Number and Active Track.
        """
        if not active_tracks: return
        
        jersey_map = defaultdict(list)
        for t in active_tracks:
            tid = t.track_id
            # Resolve to root ID to catch "Ghost -> Real" duplicates
            root_id = self.get_resolved_id(tid)
            jnum = self.active_bindings.get(root_id)
            
            if jnum is not None:
                jersey_map[jnum].append(t)
                
        for jnum, tracks in jersey_map.items():
            if len(tracks) > 1:
                self.resolve_collision(tracks, jnum)

    def resolve_collision(self, tracks, jnum):
        """
        Resolve Identity Collision: Locked > Active > Confidence
        """
        # Sort tracks by strength: [Locked, TrackID(Older=Smaller)]
        # Use root_id to check lock status
        ranked = sorted(tracks, key=lambda t: (
            self.is_locked(self.get_resolved_id(t.track_id)),  # 1. Locked wins (Root)
            -(t.track_id) # 2. Smaller ID (Older) wins 
        ), reverse=True)
        
        winner = ranked[0]
        losers = ranked[1:]
        
        for l in losers:
            l_id = l.track_id
            log(f"⚔️ [Conflict] Jersey #{jnum}: Track {winner.track_id} beats Track {l_id}. Suppressing {l_id}.")
            
            # Action: Break the link!
            # 1. If direct binding (unlikely if collision via root), remove it.
            if l_id in self.active_bindings:
                del self.active_bindings[l_id]
                self.locks.pop(l_id, None)
            
            # 2. If mapped (Ghost -> Real), remove the mapping
            if l_id in self.track_map and self.track_map[l_id] != l_id:
                old_root = self.track_map[l_id]
                # Reset mapping to self
                self.track_map[l_id] = l_id 
                log(f"   -> Unmapped Track {l_id} from {old_root}")

    def is_locked(self, track_id):
        return track_id in self.locks

    def should_update_jnr(self, track_id, frame_idx, cadence=5):
        """Throttle JNR updates per track."""
        last = self.last_jnr_update.get(track_id, -999)
        if frame_idx - last >= cadence:
            return True
        return False
        
    def record_jnr_update(self, track_id, frame_idx):
        self.last_jnr_update[track_id] = frame_idx
        
        # 2. Compare against reference crop (first one)
        # For now, we trust the locking mechanism more than visual re-check to avoid over-rejection
        # This is a placeholder for a more advanced visual consistency check (e.g. cosine similarity)
        return True, 1.0



    def resolve_by_visual(self, current_track_id, crop, classifier, team_color="Unknown"):
        """
        Hybrid ReID Tier 3: Visual Appearance Match (SigLIP).
        Called when new track appears (no jersey number yet).
        """
        if classifier is None or crop is None or crop.size == 0:
            return current_track_id
            
        # 1. Query the Visual Gallery
        match_tid, score = classifier.query_identity(crop, threshold=0.85)
        
        if match_tid is not None and match_tid != current_track_id:
            # 2. Check if matched track is "Lost" (not currently active in this frame)
            # (Simplification: If match_tid is in active_bindings, ensure it's not THIS track)
            
            # 3. Retrieve Jersey Info for the matched track
            jersey_num = self.active_bindings.get(match_tid)
            if jersey_num:
                # 4. Color Sanity Check (if provided)
                if team_color != "Unknown":
                    orig_color = self.track_colors.get(match_tid, "Unknown")
                    if orig_color != "Unknown" and orig_color != team_color:
                        log(f"⚠️ [Visual Re-ID] REJECTED {match_tid} (Score {score:.2f}) due to color: {team_color} vs {orig_color}")
                        return current_track_id
                        
                log(f"👁️ [Visual Re-ID] Track {current_track_id} -> {match_tid} (Jersey #{jersey_num}, Score {score:.2f})")
                
                # MERGE LOGIC
                self.track_map[current_track_id] = match_tid
                return match_tid
                
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

    def set_track_color(self, track_id, color, cls_id=None):
        """Set track color with role-based logic.
        - Referee (cls_id=3): Skip color assignment entirely
        - Goalkeeper (cls_id=1): Store in goalkeeper_colors, don't count for team detection
        - Player (cls_id=2): Store and count for team detection
        """
        # Skip color for Referee
        if cls_id == 3:
            return
        
        if track_id not in self.track_colors or self.track_colors[track_id] == "Unknown":
            self.track_colors[track_id] = color
            
            # Goalkeeper: Store separately, don't count for team colors
            if cls_id == 1:
                if color != "Unknown":
                    self.goalkeeper_colors[track_id] = color
                return
            
            # Player: Count color frequency for team detection
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

    def get_team_label(self, track_id):
        """Get team label for a track (Team A, Team B, Goalkeeper, or Unknown).
        - Players with Team A color -> 'Team A'
        - Players with Team B color -> 'Team B'
        - Goalkeepers -> 'GK' + their actual color
        - Others -> raw color
        """
        cls_id = self.track_classes.get(track_id)
        color = self.track_colors.get(track_id, "Unknown")
        
        # Goalkeeper: Return GK + color
        if cls_id == 1:
            gk_color = self.goalkeeper_colors.get(track_id, color)
            return f"GK ({gk_color})"
        
        # Referee: Should not have color, but fallback
        if cls_id == 3:
            return "Referee"
        
        # Player: Map to Team A or Team B
        if len(self.team_colors) >= 2:
            if color == self.team_colors[0]:
                return "Team A"
            elif color == self.team_colors[1]:
                return "Team B"
        
        # Fallback: return raw color
        return color

    def set_track_class(self, track_id, cls_id):
        """Store detection class for a track (only set once)."""
        if track_id not in self.track_classes:
            self.track_classes[track_id] = cls_id

    def get_role(self, track_id, y_pos=None, frame_height=None, x_pos=None, frame_width=None):
        """
        Role assignment based on model detection class (Phase 132).
        Model classes: 1=Goalkeeper, 2=Player, 3=Referee
        
        Refined Logic (V2):
        - If explicit Model Class 1 -> Goalkeeper
        - If explicit Model Class 3 -> Referee
        - If Class 2 (Player) but Color Outlier:
            - If near edges (Goal Area) -> Goalkeeper
            - Else -> Referee
        """
        # Use model detection class directly if explicit
        cls_id = self.track_classes.get(track_id)
        if cls_id == 1:
            return "Goalkeeper"
        elif cls_id == 3:
            return "Referee"
        
        # Default = Player (cls_id == 2 or unknown)
        
        # Phase v33: Implicit Referee Detection via Color Outlier
        # If a "Player" has a stable color that matches NEITHER team, call them Referee OR Goalkeeper.
        if len(self.team_colors) >= 2:
            color = self.track_colors.get(track_id, "Unknown")
            if color != "Unknown" and color not in self.team_colors:
                 # It's an outlier (GK or Ref). Use Position Heuristic.
                 if x_pos is not None and frame_width is not None:
                     # Goal Zone Heuristic: Left 15% or Right 15% -> Likely GK
                     is_near_edge = (x_pos < frame_width * 0.15) or (x_pos > frame_width * 0.85)
                     if is_near_edge:
                         return "Goalkeeper"
                     else:
                         # Midfield Outliers: Revert to "Player" (Unknown) instead of Referee
                         # to avoid weird shadow/lighting false positives
                         return "Player"
                 
                 # Fallback if no position info
                 return "Player"

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
        self.prompt = "Identify the two-digit or single-digit printed jersey number on the player's kit. Reply ONLY with the integer value (0-99). Do not write sentences."
        
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
            
            # FIX: Handle Temporal Sequence (List of crops)
            if isinstance(img_bgr, list):
                if not img_bgr: continue
                img_bgr = img_bgr[-1] # Take the most recent frame
            
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
                    empty_cache()
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
            except ValueError:
                pass  # Not a valid integer
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
            
            # --- Visualizer Filter (Run 18) ---
            # Only draw CONFIRMED tracks (not candidate/lost ghosts)
            # frame_data usually contains all 'online_targets' which should be Tracked
            # But let's be safe: if 'state' is in box and not 1 (Tracked), skip?
            # Pipeline usually only exports tracked, but let's assume all here.
            # ----------------------------------
            
            # Phase 125/v26: Handle Persistent Remapping & Jersey Display
            # STRICT MODE: Track ID is Source of Truth
            stable_id = tid 
            
            # Phase 216: Check both locks AND active_bindings for jersey number
            locked_info = id_manager.locks.get(tid)
            locked_jersey = locked_info["jersey"] if (locked_info and locked_info.get("locked")) else None
            
            # Also check active_bindings (includes soft-registered jerseys)
            if locked_jersey is None:
                locked_jersey = id_manager.active_bindings.get(tid)
            
            # Get Team Label
            team_color = id_manager.get_track_color(tid)
            team_label = team_color if team_color != "Unknown" else ""

            # Resolve Role (handling Implicit Referees)
            # Pass y2 as approximation of foot position for GK zone check (updated to use CX for X-Zone)
            cx = (x1 + x2) // 2
            role = id_manager.get_role(tid, y_pos=y2, frame_height=img.shape[0], x_pos=cx, frame_width=img.shape[1])
            
            # Color based on Role
            if role == "Goalkeeper":
                color = (255, 165, 0)  # Orange
            elif role == "Referee":
                color = (0, 0, 0)  # Black
                team_label = "" # Suppress Team Label for Referee
            else:  # Player
                # Visual Team Coding
                if team_color == "Red": color = (0, 0, 255) # Red
                elif team_color == "White": color = (200, 200, 200) # White/Grey
                else: color = (128, 255, 0)  # Lime Green (Unknown)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label Construction: T{tid} {team} #{jersey/cs}
            # User Request: "T{track_id} always, #{jersey} only if locked"
            
            if locked_jersey is not None:
                # Locked or bound
                label = f"T{tid} {team_label} #{locked_jersey}"
                # Background for locked
                cv2.rectangle(img, (x1, y1-20), (x2, y1), color, -1)
            else:
                # Unlocked
                label = f"T{tid} {team_label} #?"
                
            cv2.putText(img, label, (x1, y1-5), self.font, 0.6, (255, 255, 255), 2)
            
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
    parser.add_argument("--locking_mode", type=int, choices=[1, 2, 3], default=2, help="Locking mode: 1=Instant, 2=Consecutive High Conf, 3=Bayesian Dirichlet")
    parser.add_argument("--jnr_stride", type=int, default=30, help="Stride for JNR (frames)")
    parser.add_argument("--vid_stride", type=int, default=1, help="Video frame stride (skip frames). Default=1 (process all). 2=half speed/2x faster.")
    parser.add_argument("--tracking_mode", type=str, default="bytetrack", choices=["bytetrack", "botsort"], help="Tracking backend (Legacy Arg)") 
    parser.add_argument("--tracker", type=str, default="bytetrack", choices=["bytetrack", "botsort"], help="Strict Tracker Selector")
    parser.add_argument("--enable_reid", type=int, default=0, help="Enable ReID (0/1)")
    parser.add_argument("--audit_rejections", type=int, default=0, help="Enable Rejection Audit (0/1)")
    parser.add_argument("--resize_h", type=int, default=0, help="Resize height (0 to disable)")
    args = parser.parse_args()
    
    # Determine video path
    video_path = args.video or os.environ.get("PIPELINE_VIDEO") or CONFIG.get('env', {}).get('SRC_VIDEO') or "/home/ubuntu/football/121364_0.mp4"
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
    # Phase 216: Temporal Crop Buffer for JNR (stores best crops per track)
    jnr_crop_buffer = defaultdict(lambda: deque(maxlen=10))  # Store up to 10 recent torso crops per track 
    # Init Components
    id_manager = IdentityManager()
    # Phase 196 Revert: Switch back to Qwen (JNRService) as requested
    # Update: Switching to SmolVLM2Service to resolve    # Phase 200: Hybrid JNR (ResNet + Qwen Verification)

    # Phase 20: JNR Integration (Hybrid Queue)
    logging.info("🔄 [JNR] Initializing JNR Service...")
    # Clean hardcoded paths
    jnr_weights = CONFIG['env']['JNR_WEIGHTS'] if CONFIG['env']['JNR_WEIGHTS'] else "models/resnet34_rgb_jnr.pt"
    jnr_service = JNRService(weights_path=jnr_weights)




    # ReID Component (SigLIP)
    siglip_classifier = None
    if args.enable_reid:
        log("🚀 [SigLIP] initializing for ReID...")
        from vision.color_classifier import SigLIPTeamClassifier
        siglip_classifier = SigLIPTeamClassifier()
    else:
        log("💤 [SigLIP] ReID DISABLED (Lazy Load)")
    
    visualizer = Visualizer()
    color_classifier = TeamColorClassifier()  # Phase 139
    kit_coordinator = KitCoordinator()  # Phase 168
    pitch_manager = PitchManager(model_path=CONFIG['env']['POSE_WEIGHTS'], device=get_device().type)
    camera = Camera(pitch_manager.H_default)
    
    player_model = YOLO(CONFIG['env']['DET_WEIGHTS'])
    ball_model = YOLO(CONFIG['env']['BALL_MODEL_PATH'])
    loader = ThreadedVideoReader(video_path)
    time.sleep(1.0)
    
    # Video Writer (conditional)
    out_video_path = os.path.join(output_dir, "output_video.mp4")
    fps = loader.stream.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(loader.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(loader.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tracking Setup (Strict) - MOVED HERE
    # Ensure args has required tracker config
    if not hasattr(args, 'track_high_thresh'): args.track_high_thresh = 0.50 # User Request V4: 0.50
    if not hasattr(args, 'track_low_thresh'): args.track_low_thresh = 0.15 # User Request V4: 0.15
    if not hasattr(args, 'new_track_thresh'): args.new_track_thresh = 0.90 # User Request V4: 0.90 (Ultra-Strict)
    if not hasattr(args, 'match_thresh'): args.match_thresh = 0.60 # User Request V4: 0.60 (Loose)
    if not hasattr(args, 'track_buffer'): args.track_buffer = 240 # User Request V4: 240 (8s)
    if not hasattr(args, 'proximity_thresh'): args.proximity_thresh = 0.5
    if not hasattr(args, 'appearance_thresh'): args.appearance_thresh = 0.25
    # FIX: Force with_reid=False to prevent Ultralytics BoTSORT from loading OSNet
    # We rely on enable_reid for our Custom SigLIP interaction
    args.with_reid = False 

    # FIX: Add missing Ultralytics tracker args
    if not hasattr(args, 'fuse_score'): args.fuse_score = True # User Request: True
    if not hasattr(args, 'gmc_method'): args.gmc_method = "sparseOptFlow" # or None
    if not hasattr(args, 'mot20'): args.mot20 = False
    if not hasattr(args, 'model'): args.model = "osnet_x0_25_msmt17.pt" # Local restored model

    # Configure Tracker
    tracker = build_tracker(args.tracker, fps, args.enable_reid, args)
    log(f"🛡️ [Tracker] FINAL = {tracker.__class__.__name__}")
    if args.tracker == "bytetrack":
        assert "Byte" in tracker.__class__.__name__ or "BYTE" in tracker.__class__.__name__
    
    # Step 5: Resize Control
    target_width, target_height = width, height
    if args.resize_h and args.resize_h > 0:
        ratio = args.resize_h / height
        target_height = args.resize_h
        target_width = int(width * ratio)
        log(f"📏 [Coords] Resize Enabled: {width}x{height} -> {target_width}x{target_height}")
    else:
        log(f"📏 [Coords] Native Resolution: {width}x{height} (No Resize)")
    
    # Update global proc dims for consistency if used elsewhere
    PROC_W, PROC_H = target_width, target_height # Although we don't necessarily enforce this on the frame yet unless we resize!

    if args.no_video_output:
        writer = None
        log("Video output DISABLED (--no_video_output)")
    else:
        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))
    
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
            
            # Step 5: Apply Resize if requested
            if args.resize_h and args.resize_h > 0:
                f = cv2.resize(f, (target_width, target_height))
            
            if args.vid_stride > 1 and n % args.vid_stride != 0:
                continue
            
            if n % 10 == 0: log(f"Processing Frame {n}...")
            
            try:
                # 1. Detect Players (Original Params)
                det_res = player_model(f, classes=[1, 2, 3], conf=CONFIG["heuristics"]["DET_CONF"], verbose=False)[0]
                    
                if det_res.boxes is not None and len(det_res.boxes) > 0:
                    # Ensure CPU for tracker
                    det_res = det_res.cpu()
                        
                    # --- Phase v31: Part-Box Filter (Strict Spawn Suppression) ---
                    # Filter bad detections BEFORE they hit the tracker
                    # DETRIMENTAL: Edge filter causes fragmentation. Disabled for Run 4.
                    # det_res.boxes = filter_detections_strict(det_res.boxes, width, height, n)
                    
                # 2. Update Tracker (Standard)
                # 2. Update Tracker (Standard)
                if args.tracker == "bytetrack":
                    # Tune match_thresh to user spec (Run 15 - High Tolearnce)
                        
                    # --- Run 15 Param Setup ---
                    effective_fps = 30 # Default assumption if not calc
                    if args.vid_stride > 0:
                         effective_fps = int(round(30 / max(1, args.vid_stride))) 
                             
                    stale_seconds = 12
                    hard_seconds = 60
                        
                    stale_frames = int(effective_fps * stale_seconds)
                    hard_frames_limit = int(effective_fps * hard_seconds)
                        
                    if hasattr(tracker, 'args'):
                        # Run 20 Config (Strict IoU + Rescue + Locking)
                        tracker.args.match_thresh = 0.65       # User: 0.65
                        tracker.args.track_high_thresh = 0.45  # User: 0.45
                        tracker.args.track_low_thresh = 0.08   # User: 0.08
                        tracker.args.new_track_thresh = 0.85   # User: 0.85
                        tracker.args.track_buffer = 240
                        
                    # ByteTrack generally handles detection objects or numpy arrays
                    pass


                # Define for compatibility with downstream update call
                filtered_boxes_obj = det_res.boxes

                if filtered_boxes_obj is not None and len(filtered_boxes_obj) > 0:
                    # Use Stale-Guard Wrapper
                    if args.tracker == "bytetrack":
                         online_targets = bytetrack_update_with_stale_guard(
                            tracker, filtered_boxes_obj, f,
                            stale_frames=stale_frames,
                            hard_frames=hard_frames_limit
                         )
                    else:
                         online_targets = tracker.update(filtered_boxes_obj, img=f)
                else:
                    online_targets = []
                        
                # --- Phase v31: Near-Miss Suppression ---
                # Filter ghost spawns from the output
                online_targets = prevent_ghost_spawns(online_targets, None, n)
                    
                # 3. Package Results (MockBox)
                mock_boxes = []
                # online_targets is list of STrack usually
                for t in online_targets:
                    # STrack vs BOTrack vs Numpy differences
                    if hasattr(t, 'tlbr'):
                        # STrack Object
                        tlbr = t.tlbr
                        tid = t.track_id
                        conf = t.score
                        cls_id = int(t.cls) if hasattr(t, 'cls') else 2 # Default to player
                    else:
                        # Numpy/List Format: [x1, y1, x2, y2, id, conf, cls, ...]
                        # JerseyBoTSORT returns [x1, y1, x2, y2, id, conf, cls, ind]
                        t = t.tolist() if hasattr(t, 'tolist') else t
                        tlbr = t[:4]
                        tid = int(t[4])
                        conf = t[5]
                        cls_id = int(t[6]) if len(t) > 6 else 2

                    # Guard for ghost boxes (Step 6)
                    x1, y1, x2, y2 = tlbr
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        if n % 60 == 0: log(f"⚠️ [Tracker] Invalid Box ignored: {tlbr} in frame {n}")
                        continue

                    # MockBox expects xyxy, tid, conf, cls_id
                    mock_boxes.append(MockBox(tlbr, tid, conf, cls_id))
                    
                player_res = MockResults(mock_boxes, f)
                    
                # Ball Tracking: Keep independent for now (clean ByteTrack doesn't touch ball logic)
                ball_res = ball_model.track(f, persist=True, tracker="botsort.yaml", verbose=False, device=get_device().type)[0]
                
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
                        
                # FILTER: Remove balls near feet (false positives)
                # 1. Collect all player boxes from this frame
                current_player_boxes = [b["xyxy"] for b in frame_data["boxes"] if b["cls"] in [1, 2]]
                
                # 2. Filter balls
                filtered_boxes = []
                for box_data in frame_data["boxes"]:
                     if box_data["cls"] == 32: # Ball
                         if is_near_feet(box_data["xyxy"], current_player_boxes):
                             # log(f"Dropped ball near feet: {box_data['xyxy']}")
                             continue
                     filtered_boxes.append(box_data)
                
                frame_data["boxes"] = filtered_boxes
                
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
                                   id_manager.set_track_color(tid, color, cls_id=cls_id)
                                   
                                   # Phase v28: SigLIP Observation
                                   if siglip_classifier:
                                       siglip_classifier.add_observation(tid, crop)
                                   
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

                                   # --- 6. Jersey Number Recognition (Run 20: Throttled & Gated) ---
                                   if n > 1:
                                       # 1. Cadence Check (Don't spam JNR)
                                       should_check = id_manager.should_update_jnr(tid, n, cadence=4) # Faster cadence (Phase 216)
                                       
                                       if should_check:
                                            # 2. Quality Gate (Phase 216: RELAXED for Super-Res)
                                            h_crop, w_crop = crop.shape[:2]
                                            # Allow tiny crops since we now have 4x Super-Res
                                            if h_crop >= 20 and w_crop >= 10: 
                                               # Blur Check (Variance of Laplacian)
                                               gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                               var = cv2.Laplacian(gray, cv2.CV_64F).var()
                                               if var >= 20: # Very relaxed sharpness (SR will fix blur)
                                                   
                                                   # 3. Torso Crop (Focus on number)
                                                   # Reduce Y range to upper body (ignore head/legs)
                                                   # y1 += 0.15 * h, y2 = y1 + 0.75 * h
                                                   box_h = box_data["xyxy"][3] - box_data["xyxy"][1]
                                                   x1_raw, y1_raw, x2_raw, y2_raw = box_data["xyxy"]
                                                   
                                                   pad_x = int(0.10 * (x2_raw - x1_raw))
                                                   torso_y1 = int(y1_raw + 0.15 * box_h)
                                                   torso_y2 = int(y1_raw + 0.75 * box_h)
                                                   # Safety clamp
                                                   torso_y1 = max(0, torso_y1)
                                                   torso_y2 = min(height, torso_y2)
                                                   torso_x1 = max(0, int(x1_raw) - pad_x)
                                                   torso_x2 = min(width, int(x2_raw) + pad_x)
                                                   
                                                   # Extract Torso
                                                   torso_crop = safe_crop(f, (torso_x1, torso_y1, torso_x2, torso_y2))
                                                   
                                                   if torso_crop is not None:
                                                       # Phase 216: Temporal Frame Stacking
                                                       # Store crop with sharpness score
                                                       gray_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2GRAY)
                                                       sharpness = cv2.Laplacian(gray_torso, cv2.CV_64F).var()
                                                       jnr_crop_buffer[tid].append((torso_crop, sharpness, n))
                                                       
                                                       # After 3+ crops buffered, select the sharpest and queue (Phase 216)
                                                       if len(jnr_crop_buffer[tid]) >= 3:
                                                           best_crop, best_sharp, best_frame = max(jnr_crop_buffer[tid], key=lambda x: x[1])
                                                           jnr_service.queue_request(tid, best_crop, best_frame)
                                                           id_manager.record_jnr_update(tid, n)
                                                           jnr_crop_buffer[tid].clear()  # Reset buffer after sending
                    
                                   # JNR Stride based on FPS (Phase 123) - DISABLED in favor of Run 20 Logic
                                   # if n % jnr_stride == 0:
                                   #    if not is_legible(crop): ...
            
                # --- Poll JNR Results (Run 20) ---
                predictions = jnr_service.get_results() 
                if predictions:
                    for pred in predictions:
                         track_id = pred["track_id"]
                         stable_id = track_id
                         
                         if pred.get("raw_text"):
                            log(f"VLM RAW [{track_id}]: {pred['raw_text']}")
                            
                         if pred["number"] is not None:
                            # 1. Resolve Identity
                            team_color = id_manager.get_track_color(track_id)
                            stable_id = id_manager.resolve_identity(track_id, pred["number"], team_color)
                            
                            log(f"DEBUG: Track {track_id} (Resolved: {stable_id}) => {pred['number']} (Color: {pred.get('color')}, Conf: {pred['confidence']})")
                            id_manager.process_detection(stable_id, pred["number"], "auto", pred["confidence"], detected_color=pred.get("color"))

                        
                        # NEW DEBUG LOG (Phase v26.2)


                # --- 3. Phase v32.2: GLOBAL ID RESOLUTION LOOP (Every Frame) ---
                # CRITICAL: This must run ON EVERY FRAME, regardless of JNR stride.
                # It ensures that even if the tracker spawns a temp ID (e.g. 241), 
                # we immediately remap it to the known player (e.g. 210) before visualization.
                for box in frame_data["boxes"]:
                    raw_id = box["id"]
                    if raw_id is not None:
                         resolved_id = id_manager.get_resolved_id(raw_id)
                         if resolved_id != raw_id:
                             # log(f"DEBUG: Remapping Frame {n}: {raw_id} -> {resolved_id}")
                             box["id"] = resolved_id
                             # Also ensure color is consistent
                             box["color"] = id_manager.get_track_color(resolved_id)
                
                # 4. Dump Frame Data
                all_frames.append(frame_data)


                # 3. Apply Remapping to Stats (HUD already handles visualization)
                for box in frame_data["boxes"]:
                    tid = box["id"]
                    if tid is not None:
                        # Update the box ID to the stable one for stats tracking downstream
                        box["id"] = id_manager.get_resolved_id(tid)

            except Exception as e:
                import traceback
                log(f"💥 Frame-level Pipeline Error (Frame {n}): {e}")
                traceback.print_exc()
                # Continue processing next frame instead of crashing whole loop
                continue
    
            # Run 20.1: Duplicate Suppression (Before Drawing)
            id_manager.suppress_conflicts(online_targets, n)

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
        
        # DEBUG: Dump raw frame data for metric analysis
        with open(os.path.join(output_dir, "debug_all_frames.json"), "w") as f:
            json.dump(all_frames, f)
        log(f"Dumping debug_all_frames.json ({len(all_frames)} frames)")

    # Phase v27.2: Bayesian Tracklet Consolidation
    # Perform this BEFORE stats and propagation
    id_manager.finalize_bindings()
    
    # Phase v28: SigLIP Team Clustering Report
    if siglip_classifier:
        log("📊 [SigLIP] Generating Team Clustering Report...")
        try:
            team_map = siglip_classifier.cluster_teams(n_teams=2)
            for tid, team_label in team_map.items():
                jersey = id_manager.active_bindings.get(tid, "??")
                log(f"  Track {tid} (Jersey #{jersey}) -> Team Cluster {team_label}")
        except Exception as e:
            log(f"⚠️ [SigLIP] Clustering report failed: {e}")
    else:
        log("📊 [SigLIP] Team Clustering Skipped (ReID Disabled)")
    
    # Save Best Crops - ENABLED for evaluation (Phase 196)
    # crops_dir = os.path.join(output_dir, "crops")
    # os.makedirs(crops_dir, exist_ok=True)
    # for tid, crop in best_crops.items():
    #     if crop is not None:
    #         # cv2.imwrite(os.path.join(crops_dir, f"{tid}.jpg"), crop)
    #         pass
    # log(f"Saved {len(best_crops)} track crops to {crops_dir}/")
    
    # Phase 170: Retroactive Identity Propagation (DISABLED for Run 21 Strict Mode)
    # User Rule: "player_id = track_id always". Do not propagate bindings as IDs.
    # log("Applying Retroactive Identity Propagation (Locked Tracks Only)...")
    # start_prop = time.time()
    # updates_count = 0
    # for frame in all_frames:
    #     pass 
    # log(f"Retroactive Propagation SKIPPED (Strict Identity Mode)")


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
    # DISABLED: Team clustering now handles color assignment correctly (Fix V4)
    # The reconciliation was too strict and overwrote the Unknown assignment logic
    # Keeping match_kits.json for reference but not enforcing it
    log(f"Color Reconciliation DISABLED - Team clustering handles assignment correctly")
    
    # Run Entity Resolution Script
    # DISABLED per user request (Phase 116) - not needed anymore
    log("Entity Resolution SKIPPED (Phase 116).")
    
    # Identify Players (Debug)
    log(f"Identified Players: {list(player_stats.keys())}")
    
    print("\n--- Validation Results ---")
    print(f"Stats Saved to {output_dir}/player_stats.json.")
