
import json
import os
import sys
import argparse
import numpy as np
import cv2
import torch
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MERGER] %(message)s')
def log(msg): logging.info(msg)

# --- CONFIG ---
CROP_DIR = "output/crops"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

class DeepVerifier:
    def __init__(self):
        self.model = None
        self.processor = None
        self.sr = None
        self._init_models()

    def _init_models(self):
        log("Loading Verifier Models...")
        # 1. Super-Resolution
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            if os.path.exists("models/EDSR_x4.pb"):
                self.sr.readModel("models/EDSR_x4.pb")
                self.sr.setModel("edsr", 4)
                log("SR (EDSR_x4) Loaded.")
            else:
                log("Warning: SR Model missing. Using Bicubic.")
        except Exception as e:
            log(f"SR Init Failed: {e}")

        # 2. Qwen (Using 4-bit to save memory alongside main pipeline?)
        # Ideally main pipeline has finished, so we have room.
        try:
            from transformers import BitsAndBytesConfig
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, quantization_config=qc, device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            log("Qwen2.5-VL-3B (4-bit) Loaded.")
        except Exception as e:
            log(f"Qwen Load Failed: {e}")
            sys.exit(1)

    def enhance(self, crop_path):
        if not os.path.exists(crop_path): return None
        img = cv2.imread(crop_path)
        if img is None: return None
        
        if self.sr:
            return self.sr.upsample(img)
        return cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    def verify_identity(self, img):
        # Prepare Qwen Input
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Assuming we can use processing logic similar to JNR
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": rgb},
                {"type": "text", "text": "Identify the jersey number. Return JSON: {\"number\": int, \"confidence\": float}."}
            ]
        }]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_input],
            images=[rgb],
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=64)
            gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
            out_text = self.processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
            
        return self._parse(out_text)

    def _parse(self, text):
        try:
            clean = text.strip()
            if clean.startswith("```json"): clean = clean[7:]
            if clean.endswith("```"): clean = clean[:-3]
            data = json.loads(clean)
            return data.get("number"), data.get("confidence", 0.0)
        except:
            return None, 0.0

def merge_identities(input_path="output/raw_tracks.json", output_path="output/final_merged_stats.json"):
    log(f"Starting Entity Resolution on {input_path}")
    if not os.path.exists(input_path):
        log("Error: Input file missing.")
        return

    with open(input_path, "r") as f:
        raw_tracks = json.load(f)

    verifier = DeepVerifier()

    # Step 2: Group by Key (Team_Jersey)
    groups = defaultdict(list)
    for track in raw_tracks:
        if not track.get("jersey_number"): continue
        key = f"{track.get('team','Unknown')}_{track['jersey_number']}"
        groups[key].append(track)

    final_players = {}

    # Step 3: Conflict Resolution
    for key, tracks in groups.items():
        tracks.sort(key=lambda x: x["frames"][0] if x["frames"] else 0)
        
        timeline = []
        valid_tracks = []
        
        for tr in tracks:
            tid = tr["track_id"]
            start = tr["frames"][0]
            end = tr["frames"][-1]
            
            # Conflict Check
            conflict = False
            overlap_track = None
            
            for (Ts, Te, t_idx) in timeline:
                if start < Te and end > Ts:
                    conflict = True
                    overlap_track = t_idx
                    break
            
            if not conflict:
                timeline.append((start, end, tr))
                valid_tracks.append(tr)
            else:
                log(f"[Conflict] ID {tid} overlaps with ID {overlap_track['track_id']} ({key})")
                
                # Deep Re-Check
                crop_path = f"{CROP_DIR}/{tid}.jpg"
                enhanced = verifier.enhance(crop_path)
                
                if enhanced is not None:
                    new_num, conf = verifier.verify_identity(enhanced)
                    log(f"[Deep Check] ID {tid} Analysis -> Num: {new_num} (Conf: {conf})")
                    
                    if new_num is not None and str(new_num) != str(tr["jersey_number"]) and conf > 0.8:
                        # Scenario A: Correction - Move to correct group
                        # For this script we will log it. In a full system we'd re-queue it.
                        new_key = f"{tr['team']}_{new_num}"
                        log(f"[Result] Re-identified as {new_key}. (Skipping merge for safety).")
                    elif new_num == tr["jersey_number"] and conf > 0.9:
                         # Scenario B: Confirmation - Duplicate track error.
                         log("[Result] Duplicate confirmed. Discarding weaker track.")
                    else:
                        # Scenario C: Noise
                        log("[Result] Unsure. Discarding.")
                else:
                    log("[Deep Check] No crop found. Discarding.")

        # Step 4: Merge Stats
        if not valid_tracks: continue
        
        # Base entity
        base = valid_tracks[0]
        merged_stats = base["stats"].copy()
        merged_frames = base["frames"]
        
        for other in valid_tracks[1:]:
             # Sum numeric stats
             for k, v in other["stats"].items():
                if isinstance(v, (int, float)):
                    merged_stats[k] = merged_stats.get(k, 0) + v
             # Merge Trajectory
             merged_frames.extend(other["frames"])
        
        merged_frames.sort()
        
        final_players[str(base["jersey_number"])] = {
            "jersey_number": base["jersey_number"],
            "team": base["team"],
            "stats": merged_stats
        }

    # Step 5: Output
    with open(output_path, "w") as f:
        json.dump(final_players, f, indent=2)
    log(f"Saved merged stats to {output_path}")

if __name__ == "__main__":
    merge_identities()
