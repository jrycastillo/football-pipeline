import time
from collections import defaultdict, Counter
from .visualization import get_jersey_color

class IdentityManager:
    def __init__(self):
        # Maps Jersey Number -> Global UUID (The "Permanent" ID)
        self.jersey_registry = {} 
        
        # Maps Jersey Number -> Detected Color (e.g. "Red")
        self.player_colors = {}
        
        # Maps Track ID -> Detected Color (Fallback for Unknowns)
        self.track_colors = {}
        
        # Color history per track for majority voting at lock time
        self.track_color_history = {}  # {track_id: [color1, color2, ...]}
        
        # Maps Current YOLO Track ID -> Jersey Number
        self.active_bindings = {}

        # Buffer to prevent "Flickering"
        self.vote_buffer = {} 
        
        # Memory Management
        self.last_seen = {} # track_id -> frame_idx
        self.player_last_pos = {} # jersey_num -> (x_center, y_center, frame_idx)
        self.color_samples = {} # track_id -> list of [r,g,b]
        self.jersey_color_samples = {} # jersey_num -> list of [r,g,b]

    def get_global_id(self, track_id):
        """
        Returns the Permanent UUID for a given YOLO Track ID.
        """
        jersey_num = self.active_bindings.get(track_id)
        if jersey_num is not None:
            return self.jersey_registry.get(jersey_num)
        return None
        
    def get_player_color(self, jersey_num):
        return self.player_colors.get(str(jersey_num)) or self.player_colors.get(int(jersey_num))

    def is_jersey_number(self, val):
        return str(val) in self.jersey_registry or int(val) in self.jersey_registry if str(val).isdigit() else False

    def touch(self, track_id, frame_idx):
        self.last_seen[track_id] = frame_idx

    def update_position(self, track_id, box, frame_idx):
        # If this track is bound to a jersey, update the Jersey's last known location
        jersey = self.active_bindings.get(track_id)
        if jersey:
            x1, y1, x2, y2 = box["xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self.player_last_pos[jersey] = (cx, cy, frame_idx)

    def try_merge(self, track_id, box, frame_idx):
        # Task 1: Spatial Proximity Merging
        # Check if this NEW track appears where a KNOWN player disappeared
        x1, y1, x2, y2 = box["xyxy"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        candidates = []
        for jersey, (lx, ly, lframe) in self.player_last_pos.items():
            # Only consider players lost recently (e.g. < 30 frames ago)
            if frame_idx - lframe > 30: continue
            
            # Simple Euclidean distance
            dist = ((cx - lx)**2 + (cy - ly)**2)**0.5
            
            # Threshold: 50 pixels (tuned for 1080p sports)
            if dist < 50:
                candidates.append((dist, jersey))
        
        if candidates:
            # Pick closest
            candidates.sort(key=lambda x: x[0])
            best_dist, best_jersey = candidates[0]
            
            # Force Link
            print(f"👻 [IdentityManager] GHOST MERGE: Track {track_id} -> Jersey #{best_jersey} (Dist: {best_dist:.1f}, Frames: {frame_idx - self.player_last_pos[best_jersey][2]})")
            self.active_bindings[track_id] = best_jersey
            return True
            
        return False

    def cleanup(self, current_frame):
        # Delete tracks not seen in last 30 seconds (900 frames)
        expired = [tid for tid, last in self.last_seen.items() if current_frame - last > 900]
        cleaned = 0
        for tid in expired:
            if tid in self.active_bindings:
                del self.active_bindings[tid]
            if tid in self.vote_buffer:
                del self.vote_buffer[tid]
            if tid in self.last_seen:
                del self.last_seen[tid]
            cleaned += 1
        
        if cleaned > 0:
            import gc
            gc.collect()
            print(f"[IdentityManager] Cleanup: Removed {cleaned} expired tracks.", flush=True)

    def process_detection(self, track_id, detected_number, confidence, crop=None):
        """
        Main Logic: Decides if we should lock this track to a jersey.
        """
        # --- Task 56: Always store track color for fallbacks ---
        # MOVED TO TOP (Phase 74 Fix): Collect samples even if locked!
        if crop is not None and crop.size > 0:
             detected_color = get_jersey_color(crop) # "Red", "White", "Unknown"
             if detected_color != "Unknown":
                 self.track_colors[track_id] = detected_color
                 # Accumulate color history for majority voting at lock time
                 if track_id not in self.track_color_history:
                     self.track_color_history[track_id] = []
                 if len(self.track_color_history[track_id]) < 50:
                     self.track_color_history[track_id].append(detected_color)
             
             # Capture Sample for K-Means (Center 50%)
             try:
                 h, w = crop.shape[:2]
                 cy, cx = h // 2, w // 2
                 dy, dx = int(h * 0.25), int(w * 0.25)
                 center = crop[cy-dy:cy+dy, cx-dx:cx+dx]
                 if center.size > 0:
                     mean_rgb = cv2.mean(center)[:3] # (B, G, R)
                     
                     # Check if bound
                     if track_id in self.active_bindings:
                         jersey = self.active_bindings[track_id]
                         if jersey not in self.jersey_color_samples: self.jersey_color_samples[jersey] = []
                         if len(self.jersey_color_samples[jersey]) < 100:
                             self.jersey_color_samples[jersey].append(mean_rgb)
                     else:
                         if track_id not in self.color_samples: self.color_samples[track_id] = []
                         if len(self.color_samples[track_id]) < 50: 
                             self.color_samples[track_id].append(mean_rgb)
             except Exception:
                 pass

        # 1. If already locked, ignore new detections (Strict Mode)
        if track_id in self.active_bindings:
            return

        # 2. CONFIDENCE PARSING & SCORING (Bayesian-ish)
        # Map Qwen confidence ("high", "medium", "low") to scores
        score_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
        
        try:
             # Try float first (Legacy/YOLO)
             conf_val = float(confidence)
             # Map float to score? 0.9->3, 0.5->2
             if conf_val >= 0.8: score = 3.0
             elif conf_val >= 0.5: score = 2.0
             else: score = 1.0
        except:
             # String handling
             s_conf = str(confidence).lower().strip()
             score = score_map.get(s_conf, 1.0)
        
        # Rule A: High Confidence (Score 3) -> Lock Instantly?
        # REMOVED (Phase 85): Fast lock causes hallucinations to stuck.
        # We now require aggregation in all cases (unless extremely persistent).
        # if score >= 3.0:
        #      self._lock_identity(track_id, detected_number, crop=crop)
        #      return

        # 3. Bayesian Accumulation
        if track_id not in self.vote_buffer:
            self.vote_buffer[track_id] = defaultdict(float)
        
        votes = self.vote_buffer[track_id]
        votes[detected_number] += score
        
        # 4. Check Threshold
        # Threshold: 6.0 (Equivalent to 2 Highs, or 3 Mediums)
        # Increased to prevent single-frame hallucinations
        if votes[detected_number] >= 6.0:
            # Check if it's the dominant winner
            winner = max(votes, key=votes.get)
            if winner == detected_number:
                self._lock_identity(track_id, detected_number, crop=crop)

    def _get_majority_color(self, track_id, crop=None):
        """
        Get the most reliable color for a track using majority vote.
        Uses accumulated color observations from track_color_history.
        Falls back to single-frame detection if no history available.
        """
        # Try majority vote from accumulated history
        if track_id in self.track_color_history and len(self.track_color_history[track_id]) >= 2:
            counter = Counter(self.track_color_history[track_id])
            majority_color = counter.most_common(1)[0][0]
            total = len(self.track_color_history[track_id])
            top_count = counter.most_common(1)[0][1]
            print(f"   [IdentityManager] Color majority vote: {majority_color} ({top_count}/{total} votes) | All: {dict(counter)}")
            return majority_color
        
        # Fallback: single frame detection
        if crop is not None and crop.size > 0:
            return get_jersey_color(crop)
        
        # Fallback: last known track color
        if track_id in self.track_colors:
            return self.track_colors[track_id]
        
        return "Unknown"

    def _lock_identity(self, track_id, jersey_num, crop=None):
        """
        The 'Bind' Event. Links temporary Track ID to permanent Jersey ID.
        """
        if jersey_num not in self.jersey_registry:
            self.jersey_registry[jersey_num] = f"Player_Jersey_{jersey_num}"
            print(f"🆕 [IdentityManager] NEW PLAYER CREATED: Jersey #{jersey_num}")
            
            # Detect Color using majority vote from accumulated observations
            color = self._get_majority_color(track_id, crop)
            self.player_colors[jersey_num] = color
            print(f"   [IdentityManager] Assigned Color: {color} (via {'majority vote' if track_id in self.track_color_history and len(self.track_color_history[track_id]) > 1 else 'single frame'})")
        else:
             print(f"🔄 [IdentityManager] WELCOME BACK: Track {track_id} re-linked to Jersey #{jersey_num}")
             # Update color if unknown using majority vote
             if self.player_colors.get(jersey_num) == "Unknown":
                 color = self._get_majority_color(track_id, crop)
                 if color != "Unknown":
                     self.player_colors[jersey_num] = color

        # Step B: Bind the current track to this Jersey
        self.active_bindings[track_id] = jersey_num
        
        # Step C: Migrate Color Samples
        if track_id in self.color_samples:
            if jersey_num not in self.jersey_color_samples: self.jersey_color_samples[jersey_num] = []
            # Append track samples to jersey samples
            self.jersey_color_samples[jersey_num].extend(self.color_samples[track_id])
            # Free memory
            del self.color_samples[track_id]
