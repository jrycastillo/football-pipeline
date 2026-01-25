"""
TeamColorClassifier - K-Means + HSV-Based Color Classification
Phase 139: Robust jersey color detection using HSV ranges for wider tolerance.
"""

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import Counter
from transformers import SiglipImageProcessor, SiglipVisionModel
from PIL import Image

# HSV-based color ranges with wide tolerance for lighting variations
# Format: (H_min, H_max, S_min, V_min) - HSV ranges for each color
# Note: OpenCV uses H: 0-180, S: 0-255, V: 0-255
HSV_COLOR_RANGES = {
    # (H_min, H_max, S_min, S_max, V_min, V_max)
    "Maroon": [(0, 10, 50, 255, 20, 100), (170, 180, 50, 255, 20, 100)],
    "Red": [(0, 10, 60, 255, 40, 255), (170, 180, 60, 255, 40, 255)],
    "Orange": [(10, 20, 80, 255, 60, 255)],
    "Gold": [(20, 30, 40, 255, 50, 255)],
    "Yellow": [(25, 35, 60, 255, 40, 255)], # Narrowed to make room for Gold/Lime
    "Lime": [(35, 55, 40, 255, 40, 255)], # The requested "Light Green"
    "Green": [(55, 85, 40, 255, 20, 255)], # Shifted up
    "Teal": [(80, 95, 40, 255, 30, 150)],
    "Cyan": [(85, 105, 50, 255, 40, 255)],
    "Blue": [(100, 140, 50, 255, 40, 255)],
    "Navy": [(105, 145, 40, 255, 20, 80)],
    "Purple": [(140, 160, 40, 255, 40, 255)],
    "Pink": [(150, 170, 30, 255, 80, 255)],
    "White": [(0, 180, 0, 50, 180, 255)],
    "Silver": [(0, 180, 0, 30, 120, 180)],
    "Black": [(0, 180, 0, 255, 0, 30)],
    "Gray": [(0, 180, 0, 40, 40, 150)],
}


class TeamColorClassifier:
    """HSV-based color classifier with K-means clustering and wide tolerance."""
    
    def __init__(self):
        # Color voting buffer per track ID
        self.color_buffer = {}  # {track_id: [color1, color2, ...]}
        self.buffer_size = 30
    
    def _mask_grass(self, hsv_image):
        """Create mask to exclude grass (green pitch) pixels."""
        # More robust grass mask (Phase 141)
        # H 35-90 covers most turf/grass. S > 30 ensures it's actually green.
        h = hsv_image[:, :, 0]
        s = hsv_image[:, :, 1]
        v = hsv_image[:, :, 2]
        
        grass_mask = (h >= 35) & (h <= 90) & (s > 25) & (v > 20)
        return ~grass_mask
    
    def _get_torso_roi(self, crop):
        """Extract center-upper body region (torso) from crop."""
        h, w = crop.shape[:2]
        
        # If crop is already very narrow/short relative to a full person, it might be a torso
        # Regular person aspect is ~2-3. Torso aspect is ~1.
        aspect = h / w if w > 0 else 0
        
        if aspect < 1.5:
            # Likely ALREADY a torso or close to it. Don't double crop too much.
            y1 = int(h * 0.1)
            y2 = int(h * 0.9)
            x1 = int(w * 0.1)
            x2 = int(w * 0.9)
        else:
            # Torso: center region to avoid shorts/socks
            y1 = int(h * 0.15)
            y2 = int(h * 0.65)
            x1 = int(w * 0.20)
            x2 = int(w * 0.80)
        
        return crop[y1:y2, x1:x2]
    
    def _find_dominant_hsv(self, hsv_pixels):
        """
        Find dominant color using Histogram (Mode) rather than Median.
        Focuses on Hue for chromatic colors, handles Grayscale separately.
        """
        if len(hsv_pixels) < 10:
            return None
            
        # 1. Separate into Chromatic (Color) and Achromatic (Gray/Black/White)
        # S < 40 considered achromatic (white/black/grey)
        is_chromatic = hsv_pixels[:, 1] > 40
        
        chromatic_pixels = hsv_pixels[is_chromatic]
        achromatic_pixels = hsv_pixels[~is_chromatic]
        
        # 2. Determine if the jersey is mostly Color or Grayscale
        # Heuristic: If > 30% pixels are chromatic, treat as Colored. 
        # (Jerseys usually have strong color unless they are White/Black)
        if len(chromatic_pixels) > len(hsv_pixels) * 0.3:
            # --- CHROMATIC PATH (Find Dominant Hue) ---
            
            # Histogram for Hue (0-180), bin size 10 -> 18 bins
            hist, bins = np.histogram(chromatic_pixels[:, 0], bins=18, range=(0, 180))
            
            # Find peak bin
            peak_bin_idx = np.argmax(hist)
            start_h = bins[peak_bin_idx]
            end_h = bins[peak_bin_idx + 1]
            
            # Select pixels within this Hue range
            mask = (chromatic_pixels[:, 0] >= start_h) & (chromatic_pixels[:, 0] < end_h)
            dominant_group = chromatic_pixels[mask]
            
            if len(dominant_group) == 0:
                # Fallback to simple median if binning failed (rare)
                return np.median(chromatic_pixels, axis=0)
            
            # Return median of the DOMINANT CLUSTER (Robust)
            return np.median(dominant_group, axis=0)
            
        else:
            # --- ACHROMATIC PATH (Black/White/Grey) ---
            if len(achromatic_pixels) == 0:
                return np.median(hsv_pixels, axis=0)
                
            # For greyscale, we care about Value (Brightness)
            return np.median(achromatic_pixels, axis=0)
    
    def _classify_hsv(self, h, s, v):
        """Classify HSV values to color name with wide tolerance."""
        # 1. Check Specific Color Ranges first
        for color_name, ranges in HSV_COLOR_RANGES.items():
            for r in ranges:
                # 6-param tuple: (h_min, h_max, s_min, s_max, v_min, v_max)
                if (r[0] <= h <= r[1]) and (r[2] <= s <= r[3]) and (r[4] <= v <= r[5]):
                    return color_name
        
        # 2. Heuristic Fallbacks for very desaturated 
        if s < 30:
            if v > 180: return "White"
            if v < 60:  return "Black"
            return "Gray"
            
        # 3. Last Resort: Closest Hue
        if h < 10 or h > 170: return "Red"
        if h < 20: return "Orange"
        if h < 30: return "Gold"
        if h < 40: return "Yellow"
        if h < 55: return "Lime"
        if h < 85: return "Green"
        if h < 95: return "Teal"
        if h < 105: return "Cyan"
        if h < 145: return "Blue"
        if h < 165: return "Purple"
        return "Pink"
    
    def predict(self, crop, track_id=None):
        """Predict jersey color from image crop using HSV analysis."""
        if crop is None or crop.size == 0:
            return "Unknown"
        
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return "Unknown"
        
        # Step 1: Get torso region
        torso = self._get_torso_roi(crop)
        if torso.size == 0:
            torso = crop
        
        # Step 2: Convert to HSV
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        
        # Step 3: Mask out grass pixels
        keep_mask = self._mask_grass(hsv)
        
        # Flatten and apply mask
        hsv_flat = hsv.reshape(-1, 3)
        mask_flat = keep_mask.flatten()
        kept_pixels = hsv_flat[mask_flat]
        
        if len(kept_pixels) < 20:
            kept_pixels = hsv_flat
        
        if len(kept_pixels) < 10:
            return "Unknown"
        
        # Step 4: Find dominant HSV via K-Means
        dominant_hsv = self._find_dominant_hsv(kept_pixels)
        
        if dominant_hsv is None:
            return "Unknown"
        
        # Step 5: Classify using HSV ranges
        h_val, s_val, v_val = dominant_hsv
        color_name = self._classify_hsv(h_val, s_val, v_val)
        
        return color_name
    
    def predict_with_voting(self, crop, track_id):
        """Predict color with temporal voting for stability."""
        color = self.predict(crop, track_id=track_id)
        
        if color != "Unknown":
            if track_id not in self.color_buffer:
                self.color_buffer[track_id] = []
            
            self.color_buffer[track_id].append(color)
            
            # Keep only last N colors
            if len(self.color_buffer[track_id]) > self.buffer_size:
                self.color_buffer[track_id] = self.color_buffer[track_id][-self.buffer_size:]
        
        return self.get_stable_color(track_id)
    
    def get_stable_color(self, track_id):
        """Get most frequent color from voting buffer."""
        if track_id not in self.color_buffer or not self.color_buffer[track_id]:
            return "Unknown"
        
        counter = Counter(self.color_buffer[track_id])
        return counter.most_common(1)[0][0]
    
    def clear_buffer(self, track_id=None):
        """Clear voting buffer."""
        if track_id:
            self.color_buffer.pop(track_id, None)
        else:
            self.color_buffer.clear()


class KitCoordinator:
    """Aggregates detection colors to find base team kits (Phase 168)."""
    def __init__(self):
        # 1=GK, 2=Player
        self.counts = {1: Counter(), 2: Counter()}

    def observe(self, cls_id, color):
        """Register a color observation for a class."""
        if color == "Unknown":
            return
        if cls_id in self.counts:
            self.counts[cls_id][color] += 1
            
    def get_discovery_result(self):
        """Return top 2 colors for GKs and Players."""
        res = {
            "goalkeepers": [],
            "players": []
        }
        
        # Top 2 GK colors
        for color, _ in self.counts[1].most_common(2):
            res["goalkeepers"].append(color)
            
        # Top 2 Player colors
        for color, _ in self.counts[2].most_common(2):
            res["players"].append(color)
            
        return res


class SigLIPTeamClassifier:
    """
    SigLIP-based semantic team classifier (Phase v28).
    Extracts high-dimensional embeddings for player crops to cluster teams robustly.
    """
    def __init__(self, model_id="google/siglip-base-patch16-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 [SigLIP] Initializing component on {self.device}...")
        self.processor = SiglipImageProcessor.from_pretrained(model_id)
        self.model = SiglipVisionModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # Buffer for embeddings per track: {track_id: [embedding1, embedding2, ...]}
        self.embeddings = {}
    
    def extract_embedding(self, crop):
        """Extract a semantic embedding for a player crop (BGR numpy or PIL)."""
        if crop is None: return None
        if isinstance(crop, np.ndarray) and crop.size == 0: return None
            
        try:
            # Handle Input Type
            if isinstance(crop, np.ndarray):
                # Convert BGR to RGB and PIL
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            elif isinstance(crop, Image.Image):
                pil_img = crop
            else:
                return None
            
            with torch.no_grad():
                inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                # pooler_output: [1, 768]
                embed = outputs.pooler_output[0].cpu().numpy()
            return embed
        except Exception as e:
            print(f"⚠️ [SigLIP] Embedding failed: {e}")
            return None

    def verify_consistency(self, crop, reference_crops, threshold=0.6):
        """
        Verify if a crop is visually consistent with a set of reference crops (PIL images).
        Returns: (is_consistent, max_score)
        """
        if not reference_crops:
            return True, 1.0 # No prior data to contradict
            
        target_embed = self.extract_embedding(crop)
        if target_embed is None:
            return False, 0.0
            
        target_tensor = torch.tensor(target_embed).to(self.device).unsqueeze(0)
        
        max_score = -1.0
        
        # Check against gallery
        for ref_pil in reference_crops:
            ref_embed = self.extract_embedding(ref_pil)
            if ref_embed is None: continue
            
            ref_tensor = torch.tensor(ref_embed).to(self.device).unsqueeze(0)
            sim = torch.nn.functional.cosine_similarity(target_tensor, ref_tensor).item()
            
            if sim > max_score:
                max_score = sim
                
        # If we have matches, use the best score
        # 0.6 is a conservative threshold for "same person" in SigLIP space
        if max_score >= threshold:
            return True, max_score
            
        return False, max_score

    def add_observation(self, track_id, crop):
        """Add an embedding observation for a track."""
        embed = self.extract_embedding(crop)
        if embed is not None:
            if track_id not in self.embeddings:
                self.embeddings[track_id] = []
            self.embeddings[track_id].append(embed)

    def get_track_embedding(self, track_id):
        """Get the mean embedding for a track."""
        if track_id not in self.embeddings or not self.embeddings[track_id]:
            return None
        return np.mean(self.embeddings[track_id], axis=0)

    def query_identity(self, crop, threshold=0.85, top_k=1):
        """
        Query the gallery for a matching track ID using Cosine Similarity.
        Refined Logic:
        - Helper for Hybrid ReID (Tier 3).
        - Returns (best_match_tid, score) or (None, 0.0) if below threshold.
        """
        query_embed = self.extract_embedding(crop)
        if query_embed is None:
            return None, 0.0

        best_tid = None
        best_score = -1.0
        
        # Convert query to tensor once
        q_tensor = torch.tensor(query_embed).to(self.device).unsqueeze(0) # [1, D]

        # Compare against all known tracks
        # Optimization: Could stack all embeddings, but loop is fine for <50 tracks
        for tid, embedding_list in self.embeddings.items():
            # Use the mean embedding of the track for stability
            target_mean = np.mean(embedding_list, axis=0)
            t_tensor = torch.tensor(target_mean).to(self.device).unsqueeze(0) # [1, D]
            
            # Cosine Similarity
            sim = torch.nn.functional.cosine_similarity(q_tensor, t_tensor).item()
            
            if sim > best_score:
                best_score = sim
                best_tid = tid
        
        if best_score >= threshold:
            return best_tid, best_score
        
        return None, best_score

    def cluster_teams(self, n_teams=2):
        """
        Perform clustering across all observed tracks to identify team assignments.
        Returns: {track_id: team_label}
        """
        track_ids = []
        track_embeds = []
        
        for tid in self.embeddings.keys():
            mean_embed = self.get_track_embedding(tid)
            if mean_embed is not None:
                track_ids.append(tid)
                track_embeds.append(mean_embed)
        
        if len(track_embeds) < n_teams:
            return {tid: 0 for tid in track_ids}
            
        kmeans = KMeans(n_clusters=n_teams, n_init=10)
        labels = kmeans.fit_predict(np.array(track_embeds))
        
        return {tid: label for tid, label in zip(track_ids, labels)}


# Legacy function
def get_jersey_color(crop):
    """Legacy function - use TeamColorClassifier for new code."""
    classifier = TeamColorClassifier()
    return classifier.predict(crop)
