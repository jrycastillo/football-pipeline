"""
QwenJNRService - Pure Qwen2.5-VL Jersey Number Recognition (Phase 203)
Uses only Qwen VLM for jersey recognition, no ResNet.
"""
import cv2
import torch
import numpy as np
import re
from PIL import Image
from collections import defaultdict, deque

class QwenJNRService:
    """
    Pure Qwen2.5-VL Jersey Number Recognition Service.
    Matches the interface of ResNetRecognizerV2 for drop-in replacement.
    """
    
    def __init__(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        print("🔄 [QwenJNR] Initializing Pure Qwen2.5-VL Service...")
        
        self.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"🔄 [QwenJNR] Loading: {self.model_path}")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            _attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Prompt
        self.prompt = "Identify the jersey number on the player's back or front. Reply ONLY with the integer (1-99). If not visible or unclear, reply 'none'."
        
        # Voting/Stability Buffer (matching existing interface)
        self.queue = deque()
        self.results = []
        self.vote_history = defaultdict(list)
        self.buffer_size = 5
        
        print("✅ [QwenJNR] Service Ready.")
    
    def _infer_single(self, crop):
        """Run Qwen on a single crop, return (number, confidence)."""
        try:
            # Convert to RGB PIL
            if isinstance(crop, np.ndarray):
                if crop.size == 0:
                    return None, 0.0
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            elif isinstance(crop, Image.Image):
                pil_img = crop
            else:
                return None, 0.0

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=[text],
                images=[pil_img],
                padding=True,
                return_tensors="pt"
            ).to("cuda")
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=10)
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0]
            
            # Parse Integer
            match = re.search(r'\d+', output_text)
            if match:
                num = int(match.group())
                if 1 <= num <= 99:
                    return num, 0.90
            
            return None, 0.0
                
        except Exception as e:
            print(f"⚠️ [QwenJNR] Inference Error: {e}")
            return None, 0.0
    
    def queue_request(self, track_id, crop, frame_idx):
        """Queue a crop for processing (matches JNRService interface)."""
        self.queue.append({
            "track_id": track_id,
            "crop": crop,
            "frame_idx": frame_idx
        })
    
    def get_results(self):
        """Process queued requests and return results."""
        results = []
        
        while self.queue:
            req = self.queue.popleft()
            track_id = req["track_id"]
            crop = req["crop"]
            
            number, confidence = self._infer_single(crop)
            
            # Voting for stability
            if number is not None:
                self.vote_history[track_id].append(number)
                if len(self.vote_history[track_id]) > self.buffer_size:
                    self.vote_history[track_id] = self.vote_history[track_id][-self.buffer_size:]
                
                # Check for consensus (2+ identical in recent 5)
                recent = self.vote_history[track_id]
                from collections import Counter
                counts = Counter(recent)
                most_common, freq = counts.most_common(1)[0]
                if freq >= 2:
                    number = most_common
                    confidence = min(0.99, confidence + 0.1)
            
            results.append({
                "track_id": track_id,
                "number": number,
                "raw_text": f"Qwen:{number}",
                "confidence": confidence
            })
        
        return results
