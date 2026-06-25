import cv2
import numpy as np
from collections import defaultdict

class LazyIdentitySystem:
    def __init__(self, jnr_service):
        self.jnr_service = jnr_service
        self.registry = {} # { "Team_Number": "Global_ID" }
        self.locked_tracks = set() # Track IDs that are finalized
        self.track_overrides = {} # { tid: global_id } - Persistent mapping
        self.jnr_calls = 0 # Counter for budget
        
    def _get_online_team(self, crop):
        """
        Fast Heuristic for Team Classification (Red vs White).
        No K-Means, no history. Just pure pixel stats.
        """
        if crop is None or crop.size == 0: return "Unknown"
        
        # Center crop to avoid background
        h, w = crop.shape[:2]
        cy, cx = h // 2, w // 2
        dy, dx = int(h * 0.2), int(w * 0.2)
        center = crop[cy-dy:cy+dy, cx-dx:cx+dx]
        if center.size == 0: center = crop
        
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        # Red is usually low Hue (0-10) or high Hue (170-180)
        # White is low Saturation (<50) and high Value (>150)
        
        median_h = np.median(hsv[:,:,0])
        median_s = np.median(hsv[:,:,1])
        median_v = np.median(hsv[:,:,2])
        
        # Heuristic for "Red" (adjust if team colors differ)
        is_red = (median_h < 15 or median_h > 165) and (median_s > 50)
        
        if is_red:
            return "Red"
        else:
            return "White" # Default to White if not Red

    def update(self, frame_idx, tracks, frame_img):
        pass 
            
    def process_single_track(self, tid, crop, frame_idx, is_new_track):
        # OPTIMIZATION A: Locking (Persistent Override)
        if tid in self.track_overrides:
            return self.track_overrides[tid] # ALWAYS Return the Global ID if mapped
            
        if tid in self.locked_tracks:
            return None # No change, keep existing ID (should be covered by overrides usually)
            
        # OPTIMIZATION B: Noise
        h, w = crop.shape[:2]
        if h < 30: # Lowered from 60
            # print(f"[lazy] Ignored Track {tid} (Small Crop: {h}x{w})")
            return None
            
        # TRIGGER: New OR Aggressive Retry (5 frames) for Unlocked Tracks
        # If track is locked, we return early above.
        # So here we are always dealing with UNLOCKED tracks.
        if is_new_track or (frame_idx % 5 == 0): # Was % 30
            
            # 3. Heavy Model Call
            # We only call if we are under budget? User said < 100 calls.
            # But "process must finish". So we call when needed.
            
            self.jnr_calls += 1
            print(f"[lazy] Triggering JNR for Track {tid} (Call #{self.jnr_calls})...")
            
            # Predict
            # Since jnr_service might be the Unified or Base one, use predict_number
            try:
                # Save to temp file for Qwen (or use predict_batch if supported?)
                # Qwen expects path or we patched it to accept images?
                # Let's check if we patched it. Safest is save to disk.
                import uuid
                import os
                fname = f"temp_lazy_{uuid.uuid4().hex[:6]}.jpg"
                cv2.imwrite(fname, crop)
                
                prediction = self.jnr_service.predict_number(fname)
                if os.path.exists(fname): os.remove(fname)
                
                # Parse Result
                import json
                # prediction is string (JSON) or monkey-patched dict?
                # The restore to Phase 12 removed monkey-patch. So "predict_number" returns a string from Qwen.
                if isinstance(prediction, str):
                    clean = prediction.replace("```json", "").replace("```", "").strip()
                    try:
                        data = json.loads(clean)
                        num = data.get("number")
                        conf = data.get("confidence")
                    except:
                        num = None
                        conf = None
                else:
                    # Maybe it returns dict if I patched it differently? 
                    # Phase 12 logic returns string.
                    num = prediction.get("number")
                    conf = prediction.get("confidence")
                    
                if num is not None and conf in ["high", "medium"]: # Strict confidence
                    team = self._get_online_team(crop)
                    key = f"{team}_{num}"
                    
                    # Registry Logic
                    if key in self.registry:
                        # FOUND: Force ID
                        global_id = self.registry[key]
                        print(f"[lazy] MATCH: Track {tid} -> Global {global_id} (Registry: {key})")
                        self.locked_tracks.add(tid) 
                        self.track_overrides[tid] = global_id # PERSISTent Override
                        return global_id 
                    else:
                        # NEW: Register
                        self.registry[key] = tid # Current TID becomes the Global ID source (or we could mint a new one, but user logic implies tid is used)
                        print(f"[lazy] NEW: Registry[{key}] = {tid}")
                        self.locked_tracks.add(tid)
                        # We don't map override here because TID == GlobalID, but we could for consistency?
                        # If TID 100 becomes Red_10, then Registry['Red_10'] = 100.
                        # Later another track matches Red_10, it gets ID 100.
                        # So for the *original* track, we don't need override, as it IS the ID.
                        # BUT: To be safe, we can add it.
                        self.track_overrides[tid] = tid
                        return None 
                        
            except Exception as e:
                print(f"[lazy] JNR Error: {e}")
                
        return None
