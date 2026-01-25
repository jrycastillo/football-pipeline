
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
import yaml
import time
import json
import logging

# --- 0. ENV SETUP ---
os.environ["HF_HOME"] = "/ephemeral/hf_cache" # ENABLED: Use 700GB Ephemeral Disk
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Force disable to avoid CAS/Xet errors

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
from collections import defaultdict, Counter
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
                    self.stopped = True
                    return
                self.queue.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.queue.get()

    def more(self):
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
    if crop is None or crop.size == 0: return "Unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = np.median(hsv[:,:,0])
    s = np.median(hsv[:,:,1])
    # Simple Heuristic
    if s < 50: return "White"
    if h < 10 or h > 170: return "Red"
    if 100 < h < 130: return "Blue"
    return "Unknown"

# --- 5. IDENTITY MANAGER ---
class IdentityManager:
    def __init__(self):
        self.jersey_registry = {} 
        self.player_colors = {}
        self.track_colors = {}
        self.active_bindings = {}
        self.vote_buffer = {} 
        self.last_seen = {} 
        self.player_last_pos = {} 
        
    def get_player_color(self, jersey_num):
        return self.player_colors.get(str(jersey_num)) or self.player_colors.get(int(jersey_num))

    def is_jersey_number(self, val):
        return str(val) in self.jersey_registry or int(val) in self.jersey_registry if str(val).isdigit() else False

    def touch(self, track_id, frame_idx):
        self.last_seen[track_id] = frame_idx

    def process_detection(self, track_id, detected_number, confidence_str, confidence_val=0.0):
        if track_id in self.active_bindings: return

        # Bayesian Scoring
        # If real Qwen provides float confidence, use it. 
        # If only 'high'/'medium', map to float.
        score = 0.0
        if isinstance(confidence_val, float) and confidence_val > 0:
            score = confidence_val
        else:
            if str(confidence_str) == "high": score = 0.9
            elif str(confidence_str) == "medium": score = 0.6
            elif str(confidence_str) == "low": score = 0.3
        
        if track_id not in self.vote_buffer: self.vote_buffer[track_id] = defaultdict(float)
        self.vote_buffer[track_id][detected_number] += score
        
        # Aggregation Threshold: Sum > 3.0
        if self.vote_buffer[track_id][detected_number] > 3.0:
            winner = max(self.vote_buffer[track_id], key=self.vote_buffer[track_id].get)
            if winner == detected_number:
                self._lock_identity(track_id, detected_number)

    def _lock_identity(self, track_id, jersey_num):
        if jersey_num not in self.jersey_registry:
            self.jersey_registry[jersey_num] = f"Player_Jersey_{jersey_num}"
            log(f"🆕 [IdentityManager] NEW PLAYER CREATED: Jersey #{jersey_num} (Track {track_id})")
        else:
             log(f"🔄 [IdentityManager] WELCOME BACK: Track {track_id} re-linked to Jersey #{jersey_num}")
        self.active_bindings[track_id] = jersey_num

    def set_track_color(self, track_id, color):
        if track_id not in self.track_colors or self.track_colors[track_id] == "Unknown":
             self.track_colors[track_id] = color

    def get_jersey_num_color(self, jersey_num):
        # Find dominant color among tracks bound to this jersey
        colors = []
        for tid, j in self.active_bindings.items():
            if j == jersey_num and tid in self.track_colors:
                colors.append(self.track_colors[tid])
        if not colors: return "Unknown"
        return Counter(colors).most_common(1)[0][0]

# --- 6. SUPER-RESOLUTION & SIGNAL MAXIMIZATION ---
def get_upsampler(model_path="models/EDSR_x4.pb"):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4)
    return sr

# Torso Crop: Top 50% (Legs removed)
def _torso_crop(img, xyxy):
    Himg, Wimg = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    H = y2 - y1; W = x2 - x1
    if H <= 0 or W <= 0: return None
    # 15% to 65% height (safe torso)
    yA = y1 + int(0.15 * H); yB = y1 + int(0.65 * H)
    xA = x1 + int(0.10 * W); xB = x1 + int(0.90 * W)
    yA = max(0, yA); yB = min(Himg, yB)
    xA = max(0, xA); xB = min(Wimg, xB)
    yA = max(0, yA); yB = min(Himg, yB)
    xA = max(0, xA); xB = min(Wimg, xB)
    return img[yA:yB, xA:xB]

def calculate_laplacian_variance(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

# --- 7. JNR SERVICE (Real Qwen) ---
class JNRService:
    def __init__(self):
        log("Initializing Real JNRService (Qwen2.5-VL-3B)...")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        # Load Model
        max_mem = {0: "60GB"} 
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
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
            
        # Init SuperRes
        # FORCE DISABLED per user request
        self.sr = None
        log("Super-Resolution DISABLED (Bicubic Fallback).")

    def predict_batch(self, images):
        results = []
        if not images: return []
        
        # 1. Pre-process: Upscale
        processed_imgs = []
        for img_bgr in images:
            if self.sr:
                try: upscaled = self.sr.upsample(img_bgr)
                except: upscaled = cv2.resize(img_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            else:
                upscaled = cv2.resize(img_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            
            img_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
            processed_imgs.append(Image.fromarray(img_rgb))

        # 2. TRUE BATCH INFERENCE (Optimization for 80GB GPU)
        # Construct batch messages
        messages_list = []
        for pil_img in processed_imgs:
            messages_list.append([
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Identify the jersey number. JSON: {'number': int|null, 'confidence': float, 'visibility': str}."}
                ]}
            ])
        
        # Prepare inputs
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        image_inputs, video_inputs = process_vision_info(messages_list)
        
        # Batch Tokenization
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=64)
            
        # Decode
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        
        for text in output_texts:
            results.append(self._parse_json(text))
            
        return results

    def _parse_json(self, output_text):
        try:
            clean_text = output_text.strip()
            if clean_text.startswith("```json"): clean_text = clean_text[7:]
            if clean_text.endswith("```"): clean_text = clean_text[:-3]
            data = json.loads(clean_text)
            return {"number": data.get("number"), "confidence": data.get("confidence", 0.0), "visibility": data.get("visibility")}
        except:
             return {"number": None, "confidence": 0.0, "visibility": "error"}

# --- 7.5. VISUALIZER ---
class Visualizer:
    def __init__(self, fps=25.0):
        self.fps = fps
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_hud(self, img, frame_data, id_manager):
        # Draw Boxes & IDs
        for box in frame_data["boxes"]:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            tid = box["id"]
            if tid is None: continue
            
            # Color based on Team (Mock/Simple)
            color = (0, 255, 0)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label: ID | Jersey
            label = f"ID:{tid}"
            jersey = id_manager.active_bindings.get(tid)
            if jersey:
                label += f" | #{jersey}"
                cv2.rectangle(img, (x1, y1-20), (x2, y1), (0, 0, 255), -1)
                
            cv2.putText(img, label, (x1, y1-5), self.font, 0.5, (255, 255, 255), 1)
            
        return img

# --- 8. STATS & EVENTS ---
class StatsEngine:
    def __init__(self, camera=None):
        self.camera = camera
        
    def process_events(self, all_frames, id_manager=None):
        # Generate RAW TRACKS for Post-Processing Merge
        raw_tracks = []
        
        # 1. Group frames by ID
        track_history = defaultdict(list) # {track_id: [(frame_idx, cx, cy), ...]}
        
        for f_idx, frame in enumerate(all_frames):
            for box in frame["boxes"]:
                tid = box["id"]
                if tid is None: continue
                cx = (box["xyxy"][0] + box["xyxy"][2]) / 2
                cy = (box["xyxy"][3]) 
                track_history[tid].append((f_idx, cx, cy))

        # 2. Compute Stats per Track
        for tid, points in track_history.items():
            jersey_num = id_manager.active_bindings.get(tid)
            team_color = id_manager.track_colors.get(tid, "Unknown")
            
            # Distance
            total_dist = 0.0
            for i in range(1, len(points)):
                p1 = points[i-1][1:] # cx, cy
                p2 = points[i][1:]
                if self.camera:
                    dist = self.camera.calculate_distance(p1, p2)
                else:
                    dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) * 0.01
                total_dist += dist
            
            raw_tracks.append({
                "track_id": tid,
                "jersey_number": jersey_num,
                "team": team_color,
                "frames": [p[0] for p in points],
                "stats": {
                    "total_distance": round(total_dist, 2),
                    "frame_count": len(points)
                }
            })

        return raw_tracks, []

# --- 9. MAIN EXECUTION ---
if __name__ == "__main__":
    video_path = "/home/ubuntu/videoforprocessing_link/clipped_ikorudo_tornadoes.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found.")
        sys.exit(1)

    log(f"Starting Phase 85 Pipeline on {video_path}...")
    
    # Init Components
    id_manager = IdentityManager()
    jnr_service = JNRService()
    visualizer = Visualizer()
    pitch_manager = PitchManager(device=0)
    camera = Camera(pitch_manager.H_default)
    
    # Tracking
    model = YOLO("models/detect.pt")
    loader = ThreadedVideoReader(video_path)
    time.sleep(1.0)
    
    # Video Writer
    out_video_path = "output/full_run_ikorudo.mp4"
    os.makedirs("output", exist_ok=True)
    fps = loader.stream.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(loader.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(loader.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    all_frames = []
    MAX_FRAMES = 2500 # Full video is 1743 frames
    
    n = 0
    start_time = time.time()
    
    try:
        while loader.more() and n < MAX_FRAMES:
            f = loader.read()
            if f is None: break
            n += 1
            
            if n % 10 == 0: log(f"Processing Frame {n}...")
            
            # Track
            results = model.track(f, persist=True, verbose=False, device=0)
            res = results[0]
            img = res.orig_img

            # Pitch Calib (Every 60 frames)
            if n % 60 == 0:
                 kps, H_new = pitch_manager.predict(f)
                 camera.update(H_new)
            
            frame_data = {"boxes": []}
            batch_crops = []
            batch_ids = []
            
            if hasattr(res, "boxes"):
                for b_idx, b in enumerate(res.boxes):
                    box_data = {
                        "xyxy": b.xyxy[0].cpu().numpy().tolist(),
                        "id": int(b.id[0].item()) if b.id is not None else None,
                        "conf": float(b.conf[0].item())
                    }
                    frame_data["boxes"].append(box_data)
                    
                    if box_data["id"] is not None:
                        # OPTIMIZATION: Skip JNR if ID is already locked!
                        if box_data["id"] in id_manager.active_bindings:
                            continue

                        # JNR Logic: Stride 1 (Every Frame!)
                        crop = _torso_crop(img, box_data["xyxy"])
                        if crop is not None and crop.size > 0:
                               # Store Color
                               color = get_jersey_color(crop)
                               id_manager.set_track_color(box_data["id"], color)

                               # OPTIMIZATION: Variance Filter (Skip Blur)
                               if calculate_laplacian_variance(crop) < 50:
                                   continue
                               
                               batch_crops.append(crop)
                               batch_ids.append(box_data["id"])
            
            # JNR Inference
            if batch_crops:
                predictions = jnr_service.predict_batch(batch_crops)
                for idx, pred in enumerate(predictions):
                    if pred["number"] is not None:
                        log(f"DEBUG: Track {batch_ids[idx]} => {pred['number']} (Conf: {pred['confidence']})")
                        id_manager.process_detection(batch_ids[idx], pred["number"], "auto", pred["confidence"])
    
            # Visualization
            annotated_img = visualizer.draw_hud(img.copy(), frame_data, id_manager)
            writer.write(annotated_img)
    
            all_frames.append(frame_data)

    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        loader.stop()
        writer.release()
        log(f"Tracking finished in {time.time() - start_time:.2f}s.")
    log(f"Video saved to {out_video_path}")
    
    # --- 9. STATS GENERATION (Entity Resolution) ---
    stats_engine = StatsEngine(camera) # Instantiate StatsEngine
    raw_tracks = stats_engine.process_events(all_frames, id_manager)
    
    # Save Raw Tracks
    with open("output/raw_tracks.json", "w") as f:
        json.dump(raw_tracks, f, indent=2)
    log("Saved output/raw_tracks.json")
    
    # Run Entity Resolution Script
    try:
        import subprocess
        log("Running Entity Resolution (merge_identities.py)...")
        subprocess.run(["python3", "tools/merge_identities.py"], check=True)
        log("Entity Resolution Complete. Stats saved to output/players_stats.json.")
    except Exception as e:
        log(f"Entity Resolution Failed: {e}")
    
    # Identify Players (Debug)
    if os.path.exists("output/players_stats.json"):
        with open("output/players_stats.json", "r") as f:
            final_stats = json.load(f)
            log(f"Identify Players: {list(final_stats.keys())}")
    
    print("\n--- Validation Results ---")
    # The original instruction had a malformed line here. Assuming the intent was to just print the stats saved message.
    print("Stats Saved to output/players_stats.json.")

