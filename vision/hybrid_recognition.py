import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from vision.resnet_recognition import ResNetRecognizer
from collections import defaultdict

from utils.device_utils import is_cuda_available

class HybridJNRService:
    """
    Hybrid Jersey Number Recognition Service (Phase 200).
    Combines:
    1. ResNet32 (Primary): Fast, high-recall, stable locking.
    2. Qwen2.5-VL (Verifier): Semantic review for ambiguous/low-conf detections.
    """
    
    def __init__(self):
        print("🔄 [HybridJNR] Initializing Hybrid Service...")
        
        # 1. Initialize Primary (ResNet)
        self.resnet = ResNetRecognizer()
        
        # 2. Initialize Verifier (Qwen2.5-VL)
        # Using Qwen2.5-VL-3B-Instruct
        self.vl_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"🔄 [HybridJNR] Loading Verifier: {self.vl_model_path}")
        
        try:
            self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vl_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                _attn_implementation="flash_attention_2" if is_cuda_available() else "eager"
            )
            self.vl_processor = AutoProcessor.from_pretrained(self.vl_model_path)
            self.has_verifier = True
        except Exception as e:
            print(f"⚠️ [HybridJNR] Failed to load Qwen: {e}. Running in ResNet-only mode.")
            self.has_verifier = False
        
        # Configuration
        self.review_threshold = 0.80  # User Request: "80% below that will be pass to QWEN"
        self.ambiguous_numbers = {11, 19, 29}  # Always review these numbers
        self.vl_prompt = "Identify the jersey number. Reply ONLY with the integer."
        
        # Phase 201: Voting Buffer
        self.vote_history = defaultdict(list)
        
        print("✅ [HybridJNR] Service Ready.")

    def run_qwen_verification(self, crops):
        """Run Qwen2.5-VL on a batch of crops."""
        results = []
        if not crops or not self.has_verifier:
            return [{"number": None, "confidence": 0.0}] * len(crops)

        for crop in crops:
            try:
                # Convert to RGB PIL
                if isinstance(crop, np.ndarray):
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                else:
                    pil_img = crop # Assume PIL

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": self.vl_prompt}
                        ]
                    }
                ]
                
                text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = self.vl_processor(
                    text=[text],
                    images=[pil_img],
                    padding=True,
                    return_tensors="pt"
                ).to("cuda")
                
                with torch.no_grad():
                    generated_ids = self.vl_model.generate(**inputs, max_new_tokens=10)
                    output_text = self.vl_processor.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )[0]
                
                # Parse Integer
                import re
                match = re.search(r'\d+', output_text)
                if match:
                    num = int(match.group())
                    if 0 <= num <= 99:
                        results.append({"number": num, "confidence": 0.95}) # High conf for VL
                    else:
                        results.append({"number": None, "confidence": 0.0})
                else:
                    results.append({"number": None, "confidence": 0.0})
                    
            except Exception as e:
                print(f"⚠️ [HybridJNR] Qwen Error: {e}")
                results.append({"number": None, "confidence": 0.0})
        
        return results

    def predict_batch(self, images, track_ids=None, reference_crops=None):
        """
        Two-Pass Prediction Pipeline with Sequence Voting (Phase 201).
        """
        if not images:
            return []
            
        final_results = []
        review_indices = []
        review_crops = []
        
        # --- PASS 1: ResNet on Full Sequences ---
        # Flatten sequences to one big batch for ResNet
        flat_inputs = []
        seq_map = [] # (original_index, crop_count)
        
        for idx, item in enumerate(images):
            if isinstance(item, list):
                # Valid crops only
                valid_seq = [img for img in item if img is not None and img.size > 0]
                flat_inputs.extend(valid_seq)
                seq_map.append((idx, len(valid_seq)))
            else:
                flat_inputs.append(item)
                seq_map.append((idx, 1))
                
        if not flat_inputs:
             return [{"number": None, "confidence": 0.0, "source": "empty"}] * len(images)
             
        # Bulk Predict
        flat_preds = self.resnet.predict_batch(flat_inputs)
        
        # Reconstruct & Vote
        cursor = 0
        for i, (orig_idx, count) in enumerate(seq_map):
            track_id = track_ids[i] if track_ids else f"TRK_{i}"
            res = {
                "number": None,
                "confidence": 0.0,
                "source": "resnet",
                "box": None
            }
            
            if count == 0:
                final_results.append(res)
                continue
                
            # Extract predictions for this track's sequence
            seq_preds = flat_preds[cursor : cursor + count]
            cursor += count
            
            # --- Voting Logic (3 Consecutive) ---
            # We have a list of predictions [P1, P2, P3, P4, P5]
            # Check for ANY run of 3 identical numbers
            locked_number = None
            
            numbers = [p.get("number") for p in seq_preds]
            
            # Sliding window of 3
            if len(numbers) >= 3:
                for j in range(len(numbers) - 2):
                    window = numbers[j : j+3]
                    # Check if all valid and equal
                    if window[0] is not None and all(n == window[0] for n in window):
                        locked_number = window[0]
                        break
            
            if locked_number is not None:
                # LOCKED BY VOTING
                res["number"] = int(locked_number)
                res["confidence"] = 0.99
                res["source"] = "resnet_vote"
                print(f"🗳️ [Voting] Track {track_id} Locked on #{locked_number} (Seq: {numbers})")
                final_results.append(res)
                continue # Skip Qwen Review
            
            # If not locked, fall back to the LAST prediction (most recent)
            last_pred = seq_preds[-1]
            p_num = last_pred.get("number")
            p_conf = last_pred.get("confidence", 0.0)
            
            res["number"] = p_num
            res["confidence"] = p_conf
            
            # Check for Qwen Review
            if p_num is not None:
                if (int(p_num) in self.ambiguous_numbers) or (p_conf < self.review_threshold):
                    review_indices.append(i)
                    review_crops.append(flat_inputs[cursor - 1]) # Last crop
            elif p_num is None:
                # If ResNet failed on last frame, maybe review? 
                # (Optional: Review if we have good crop but no ResNet)
                pass

            final_results.append(res)

        # --- PASS 2: Qwen Verifier ---
        if review_indices and self.has_verifier:
            print(f"🔎 [HybridJNR] Running Qwen Verification on {len(review_indices)} items...")
            qwen_preds = self.run_qwen_verification(review_crops)
            
            for i, q_res in enumerate(qwen_preds):
                original_idx = review_indices[i]
                old_res = final_results[original_idx]
                
                if q_res["number"] is not None:
                    # Merge Logic
                    if q_res["number"] != old_res["number"]:
                        # Anti-Hallucination 10
                        if q_res["number"] == 10 and old_res["number"] != 10:
                            print(f"   ► [Anti-10] REJECTED Qwen Hallucination (10). Keeping ResNet {old_res['number']}.")
                            continue

                        print(f"   ► Review [{track_ids[original_idx] if track_ids else original_idx}]: ResNet {old_res['number']} -> Qwen {q_res['number']}")
                        final_results[original_idx]["number"] = q_res["number"]
                        final_results[original_idx]["confidence"] = q_res["confidence"]
                        final_results[original_idx]["source"] = "qwen"
                    else:
                         # Confirmed
                         final_results[original_idx]["confidence"] = 0.99 
                
        return final_results
