import cv2
import numpy as np
import math
from collections import defaultdict, Counter
import yaml
import os

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CLASS = CONFIG["classes"]

# ================= TEAM ASSIGNMENT =================

def _center_crop(img, fraction=0.5):
    if img is None or img.size == 0: return img
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    dy, dx = int(h * fraction / 2), int(w * fraction / 2)
    return img[cy-dy:cy+dy, cx-dx:cx+dx]

def _hue_to_basic_color(h):
    if h is None: return "unknown"
    if h < 10 or h >= 170: return "red"
    elif h < 25: return "orange"
    elif h < 35: return "gold"
    elif h < 55: return "yellow"
    elif h < 85: return "green"
    elif h < 105: return "cyan"
    elif h < 135: return "blue"
    elif h < 155: return "purple"
    else: return "magenta"

def _describe_color(h, s, v):
    if s < 0.25:
        if v > 0.75: return "white"
        elif v < 0.35: return "black"
        else: return "gray"
    base = _hue_to_basic_color(h)
    if v > 0.80: return f"light_{base}"
    elif v < 0.35: return f"dark_{base}"
    else: return base

def assign_teams(frames):
    print("[identity] Assigning teams...")
    id_colors = {} 
    
    # Collect colors
    for f in frames:
        if "crops" in f and f["crops"]:
            for c in f["crops"]:
                box_idx = c["box_idx"]
                if box_idx >= len(f["boxes"]): continue
                b = f["boxes"][box_idx]
                pid = b["id"]
                
                if pid is None: continue
                
                crop = c["img"]
                color_crop = _center_crop(crop, fraction=0.4)
                hsv = cv2.cvtColor(color_crop, cv2.COLOR_BGR2HSV)
                h = np.median(hsv[:,:,0])
                s = np.median(hsv[:,:,1]) / 255.0
                v = np.median(hsv[:,:,2]) / 255.0
                
                if pid not in id_colors: id_colors[pid] = []
                id_colors[pid].append([h, s, v])

    if len(id_colors) < 2:
        print("[identity] Not enough players for team assignment.")
        return {}, {}

    # Feature extraction for KMeans
    id_list = sorted(id_colors.keys())
    per_id_features = []
    per_id_hsv = {}
    
    for pid in id_list:
        colors = id_colors[pid]
        h_med = float(np.median([c[0] for c in colors]))
        s_med = float(np.median([c[1] for c in colors]))
        v_med = float(np.median([c[2] for c in colors]))
        per_id_hsv[pid] = (h_med, s_med, v_med)
        h_rad = h_med * (np.pi / 90.0)
        per_id_features.append([math.cos(h_rad), math.sin(h_rad), s_med, v_med])
        
    per_id_features = np.array(per_id_features, dtype=np.float32)
    
    # KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(per_id_features, 2, None, criteria, 10, flags)
    
    assign = {pid: int(labels[i][0]) for i, pid in enumerate(id_list)}
    
    # Ensure Team 0 is the larger team (usually home/dominant?) or just consistent
    counts = Counter(assign.values())
    if counts[0] < counts[1]:
        assign = {pid: (0 if lab == 1 else 1) for pid, lab in assign.items()}
        
    # Labeling
    cluster_hsv = {0: [], 1: []}
    for pid, team_id in assign.items():
        cluster_hsv[team_id].append(per_id_hsv[pid])
        
    team_labels = {}
    for team_id in (0, 1):
        if not cluster_hsv[team_id]:
            team_labels[team_id] = f"team_{team_id}"
            continue
        median_h = np.median([c[0] for c in cluster_hsv[team_id]])
        median_s = np.median([c[1] for c in cluster_hsv[team_id]])
        median_v = np.median([c[2] for c in cluster_hsv[team_id]])
        p10_s = np.percentile([c[1] for c in cluster_hsv[team_id]], 10)
        
        if p10_s < 0.25 and median_v > 0.70:
            label = "white"
        else:
            label = _describe_color(median_h, median_s, median_v)
        team_labels[team_id] = label
        
    print(f"[identity] Team Labels: {team_labels}")
    return assign, team_labels


# ================= JERSEY NUMBER RECOGNITION =================

# Placeholder for Qwen integration or Legacy JNR
# Since the user mentioned "Import the QwenWrapper (or create a bridge to the MLOps agent's inference script)"
# and "H100 Optimization: Ensure Qwen runs in bfloat16 precision".
# I should check if `training/jnr/inference_service.py` exists as per instructions.

# ================= JERSEY NUMBER RECOGNITION =================

import sys
import uuid
# Add project root to path to allow importing from training.jnr
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from training.jnr.inference_service import JNRService
    JNR_AVAILABLE = True
except ImportError:
    print("[identity] Could not import JNRService. Check path.")
    JNR_AVAILABLE = False
except Exception as e:
    print(f"[identity] Error importing JNRService: {e}")
    JNR_AVAILABLE = False

# Import PEFT for Fine-Tuning
try:
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class FineTunedJNRService:
    def __init__(self, adapter_path):
        print(f"Initializing FineTunedJNRService with adapter {adapter_path}...")
        max_mem = {0: "14GB"}
        
        try:
            # Load Base
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_mem,
            )
            # Load Adapter
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print("Adapter loaded successfully.")
        except Exception as e:
            print(f"Error loading model/adapter: {e}")
            raise e

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        print("Model loaded successfully.")
        
    def predict_number(self, image_path):
        return self.predict_batch([image_path])[0]

    def predict_batch(self, images):
        """
        Takes a list of image paths OR numpy arrays (BGR).
        Returns list of results: [{"number": "10", "confidence": "high"}, ...]
        """
        from qwen_vl_utils import process_vision_info
        from PIL import Image
        import numpy as np
        import cv2

        if not images: return []

        # Prepare messages
        messages = []
        for img in images:
            # Handle numpy array (convert to PIL)
            if isinstance(img, np.ndarray):
                # Check Laplacian Variance (Smart Stride)
                try:
                    gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur_score = cv2.Laplacian(gray_check, cv2.CV_64F).var()
                except:
                    blur_score = 100.0 # Fallback 

                if blur_score < 50:
                    messages.append(None) # Mark as skipped
                    continue

                # Convert BGR to RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
            else:
                # Assume path
                pil_img = img

            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": "Analyze the jersey number. Return a Strict JSON object: {\"number\": <int> or null, \"confidence\": \"high\"|\"medium\"|\"low\"}."},
                    ],
                }
            ])

        # Handle all skipped or empty
        if not messages or all(m is None for m in messages): 
            return [{"number": None, "confidence": None}] * len(images)

        # Filter out skipped messages
        valid_indices = [i for i, m in enumerate(messages) if m is not None]
        valid_messages = [messages[i] for i in valid_indices]
        
        # Prepare batch inputs
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in valid_messages
        ]
        
        # Helper for batch processing
        image_inputs, video_inputs = process_vision_info(valid_messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        valid_results = []
        for text in output_texts:
            clean_pred = text.replace("```json", "").replace("```", "").strip()
            try:
                # Simple JSON parse
                start = clean_pred.find("{")
                end = clean_pred.rfind("}") + 1
                if start != -1 and end != -1:
                    json_str = clean_pred[start:end]
                    data = json.loads(json_str)
                    results_dict = {"number": str(data.get("number")), "confidence": data.get("confidence")}
                    # Normalize None
                    if results_dict["number"] in [None, "None", "null", "NaN"]:
                         results_dict["number"] = None
                    valid_results.append(results_dict)
                else:
                    valid_results.append({"number": None, "confidence": None})
            except Exception:
                 valid_results.append({"number": None, "confidence": None})
                 
        # Reconstruct full results list
        final_results = [{"number": None, "confidence": None}] * len(images)
        for idx, res in zip(valid_indices, valid_results):
            final_results[idx] = res
            
        return final_results

class MergedJNRService:
    def __init__(self, model_path):
        print(f"Initializing MergedJNRService with model {model_path}...")
        max_mem = {0: "14GB"}
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_mem,
            )
            print("Merged model loaded successfully.")
        except Exception as e:
            print(f"Error loading merged model: {e}")
            raise e

        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Processor loaded successfully.")
        
    def predict_number(self, image_path):
        return self.predict_batch([image_path])[0]

    def predict_batch(self, images):
        """
        Takes a list of image paths OR numpy arrays (BGR).
        Returns list of results: [{"number": "10", "confidence": "high"}, ...]
        """
        from qwen_vl_utils import process_vision_info
        from PIL import Image
        import numpy as np
        import cv2

        if not images: return []

        # Prepare messages
        messages = []
        for img in images:
            # Handle numpy array (convert to PIL)
            if isinstance(img, np.ndarray):
                # Check Laplacian Variance (Smart Stride)
                try:
                    gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur_score = cv2.Laplacian(gray_check, cv2.CV_64F).var()
                except:
                    blur_score = 100.0 # Fallback 

                if blur_score < 50:
                    messages.append(None) # Mark as skipped
                    continue

                # Convert BGR to RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
            else:
                # Assume path
                pil_img = img

            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": "Analyze the jersey number. Return a Strict JSON object: {\"number\": <int> or null, \"confidence\": \"high\"|\"medium\"|\"low\"}."},
                    ],
                }
            ])

        # Handle all skipped or empty
        if not messages or all(m is None for m in messages): 
            return [{"number": None, "confidence": None}] * len(images)

        # Filter out skipped messages
        valid_indices = [i for i, m in enumerate(messages) if m is not None]
        valid_messages = [messages[i] for i in valid_indices]
        
        # Prepare batch inputs
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in valid_messages
        ]
        
        # Helper for batch processing
        image_inputs, video_inputs = process_vision_info(valid_messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        valid_results = []
        for text in output_texts:
            clean_pred = text.replace("```json", "").replace("```", "").strip()
            try:
                # Simple JSON parse
                start = clean_pred.find("{")
                end = clean_pred.rfind("}") + 1
                if start != -1 and end != -1:
                    json_str = clean_pred[start:end]
                    data = json.loads(json_str)
                    results_dict = {"number": str(data.get("number")), "confidence": data.get("confidence")}
                    # Normalize None
                    if results_dict["number"] in [None, "None", "null", "NaN"]:
                         results_dict["number"] = None
                    valid_results.append(results_dict)
                else:
                    valid_results.append({"number": None, "confidence": None})
            except Exception:
                 valid_results.append({"number": None, "confidence": None})
                 
        # Reconstruct full results list
        final_results = [{"number": None, "confidence": None}] * len(images)
        for idx, res in zip(valid_indices, valid_results):
            final_results[idx] = res
            
        return final_results

_jnr_service = None

def get_jnr_service():
    global _jnr_service
    if _jnr_service is None:
        jnr_weights = CONFIG["env"]["JNR_WEIGHTS"]
        
        if jnr_weights and os.path.isdir(jnr_weights):
            # Check if it's an adapter or full model
            is_adapter = os.path.exists(os.path.join(jnr_weights, "adapter_config.json"))
            
            if is_adapter and PEFT_AVAILABLE:
                try:
                    _jnr_service = FineTunedJNRService(jnr_weights)
                except Exception as e:
                    print(f"[identity] Failed to initialize FineTunedJNRService: {e}")
            else:
                # Assume full model (merged)
                try:
                    _jnr_service = MergedJNRService(jnr_weights)
                except Exception as e:
                    print(f"[identity] Failed to initialize MergedJNRService: {e}")
        
        # Fallback to Base JNRService
        if _jnr_service is None and JNR_AVAILABLE:
            try:
                _jnr_service = JNRService()
                print("[identity] Initialized Base JNRService.")
            except Exception as e:
                print(f"[identity] Failed to initialize JNRService: {e}")
                
    # Polyfill predict_batch if missing (Crucial for Phase 21)
    if _jnr_service and not hasattr(_jnr_service, "predict_batch"):
        print("[identity] Polyfilling predict_batch for Base JNRService...")
        # We can borrow the implementation from MergedJNRService concept via a helper
        # But since we can't easily reference the method bound to another class, we'll define a standalone wrapper
        # Or simpler: Just define it right here and bind it.
        
        def _predict_batch_polyfill(self, images):
            from qwen_vl_utils import process_vision_info
            from PIL import Image
            import numpy as np
            import cv2
            import json # Ensure json is imported
    
            if not images: return []
    
            messages = []
            for img_idx, img in enumerate(images): # Added index for debugging if needed
                if isinstance(img, np.ndarray):
                    # Phase 85: Signal Maximization (Torso + SR)
                    # REGRESSION TEST: DISABLED (Phase 28+79 Baseline)
                    # 1. Torso Crop
                    # t_crop = _torso_crop(img)
                    
                    # 2. Super Resolution
                    # sr_img = preprocess_sr(t_crop)

                    # rgb_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
                    
                    # BASELINE LOGIC (Phase 28)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                else:
                    pil_img = img
    
                messages.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": "You are a sports referee. Analyze the image of the jersey. Return a JSON object with two keys:\n'number': The visible number (integer). If you can see a number but it's blurry, output it. If no number is visible, return null.\n'confidence': 'high', 'medium', or 'low'."},
                        ],
                    }
                ])
    
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Move to device
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
    
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            results = []
            for text in output_texts:
                clean_pred = text.replace("```json", "").replace("```", "").strip()
                print(f"DEBUG RAW QWEN (Polyfill): {clean_pred}")
                try:
                    start = clean_pred.find("{")
                    end = clean_pred.rfind("}") + 1
                    if start != -1 and end != -1:
                        json_str = clean_pred[start:end]
                        data = json.loads(json_str)
                        
                        raw_num = data.get("number")
                        raw_conf = data.get("confidence")
                        
                        # VALIDATION (Same as class method)
                        if isinstance(raw_num, int):
                            final_num = str(raw_num)
                        elif isinstance(raw_num, str) and raw_num.isdigit():
                            final_num = raw_num
                        else:
                            final_num = None
                        
                        if final_num:
                             results.append({"number": final_num, "confidence": raw_conf})
                        else:
                             results.append({"number": None, "confidence": None})
                    else:
                        results.append({"number": None, "confidence": None})
                except Exception:
                     results.append({"number": None, "confidence": None})
            return results

        import types
        _jnr_service.predict_batch = types.MethodType(_predict_batch_polyfill, _jnr_service)

    return _jnr_service

import json

_upsampler = None

def get_upsampler():
    global _upsampler
    if _upsampler is None:
        try:
            # Phase 85: Re-enable Super-Resolution (Signal Maximization)
            # REGRESSION TEST: DISABLED (Phase 28+79 Baseline)
            # sr = cv2.dnn_superres.DnnSuperResImpl_create()
            # model_path = "models/EDSR_x4.pb"
            
            # if not os.path.exists(model_path):
            #     print(f"[identity] Model not found at {model_path}")
            #     _upsampler = False
            #     return _upsampler
                
            # sr.readModel(model_path)
            # sr.setModel("edsr", 4)
            # if torch.cuda.is_available():
            #     sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            #     sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                
            # _upsampler = sr
            # print("[identity] EDSR SuperRes initialized.")
            
            # Fallback to Bicubic (Simulating Phase 28)
            class BicubicUpsampler:
                def upsample(self, img):
                    if img is None or img.size == 0: return img
                    return cv2.resize(img, (img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_CUBIC)
            
            _upsampler = BicubicUpsampler()
            print("[identity] Bicubic Upsampler initialized (REGRESSION TEST MODE).")

        except Exception as e:
            print(f"[identity] Failed to init SuperRes: {e}")
            _upsampler = False # Flag as failed

        except Exception as e:
            print(f"[identity] Failed to init SuperRes: {e}")
            _upsampler = False # Flag as failed
            
    return _upsampler

def _torso_crop(img):
    """
    Heuristic: Crop to "Torso" (Top 15% to 65% of height).
    Used to focus Qwen on the jersey number, removing head and legs.
    """
    if img is None or img.size == 0: return img
    h, w = img.shape[:2]
    
    # Heuristic: y_start=0.15, y_end=0.65
    y1 = int(h * 0.15)
    y2 = int(h * 0.65)
    
    # Safety Check: ensure we have at least 50% of the crop or 30px
    if y2 - y1 < 20: 
        return img # Tool small to crop
        
    return img[y1:y2, 0:w]

def _finalize_processing(img):
    # 1. Upscale to min 224 (Bicubic) - if needed (Only if SR didn't already make it huge)
    TARGET_MIN = 224
    h, w = img.shape[:2]
    min_dim = min(h, w)
    
    if min_dim < TARGET_MIN:
        scale = TARGET_MIN / min_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
    # 2. Pad to Square
    h, w = img.shape[:2]
    if h != w:
        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        
    # 3. Sharpen (Mild)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img

def preprocess_sr(img):
    # Try Super-Resolution
    upsampler = get_upsampler()
    if upsampler:
        try:
            # Upscale
            img = upsampler.upsample(img)
        except Exception as e:
            print(f"[identity] SR Failed: {e}")
            
    return _finalize_processing(img)

def preprocess_bicubic(img):
    # Just standard processing (Bicubic upscale is inside _finalize_processing)
    return _finalize_processing(img)

def calculate_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def get_jersey_numbers(frames, team_map):
    service = get_jnr_service()
    if not service:
        print("[identity] JNR Service unavailable. Skipping.")
        return {}
        
    print("[identity] Running Qwen JNR on player crops (Optimization Mode)...")
    
    # 1. Collect ALL crops per player
    player_crops = defaultdict(list)
    
    for i, f in enumerate(frames):
        if i % 3 != 0: continue # Skip frames for speed
        
        if "crops" in f and f["crops"]:
            for c in f["crops"]:
                box_idx = c["box_idx"]
                if box_idx >= len(f["boxes"]): continue
                b = f["boxes"][box_idx]
                pid = b["id"]
                if pid is None: continue
                
                crop = c["img"]
                
                if crop.size == 0: continue
                
                # Filter 1: Size
                ch, cw = crop.shape[:2]
                if ch < 30: continue 
                
                # Filter 2: Blur
                blur_score = calculate_blur(crop)
                if blur_score < 50: continue
                
                player_crops[pid].append({"img": crop, "h": ch, "blur": blur_score})
            
    # 2. Process Top Crops per Player
    jersey_map = {}
    temp_dir = "temp_crops"
    os.makedirs(temp_dir, exist_ok=True)
    
    for pid, items in player_crops.items():
        # Sort by Height (Quality Proxy)
        items.sort(key=lambda x: x["h"], reverse=True)
        
        # Take Top 10 (Expanded Search)
        candidates = items[:10]
        votes = []
        
        print(f"[identity] Processing Player {pid}: {len(candidates)} best crops...")
        
        for i, item in enumerate(candidates):
            raw_crop = item["img"]
            
            # Helper to predict single image
            def predict_single(img_to_pred, suffix):
                fname = f"{temp_dir}/{pid}_{i}_{suffix}_{uuid.uuid4().hex[:8]}.jpg"
                cv2.imwrite(fname, img_to_pred)
                try:
                    pred = service.predict_number(fname)
                    clean_pred = pred.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean_pred)
                    return data.get("number"), data.get("confidence")
                except Exception:
                    return None, None
                finally:
                    if os.path.exists(fname): os.remove(fname)
 
            # Helper for Adaptive Rotation
            def predict_with_adaptive_rotation(base_img, base_suffix):
                # 1. Try 0 degrees
                num, conf = predict_single(base_img, f"{base_suffix}_0")
                
                # If High confidence, return immediately
                if num is not None and conf == "high":
                    return num, conf
                    
                # 2. If Low/Medium/Null, try rotations
                best_num = num
                best_conf = conf
                best_score = 0
                
                # Score mapping
                def get_score(c): 
                    if c == "high": return 3
                    if c == "medium": return 2
                    if c == "low": return 1
                    return 0
                
                if num is not None: best_score = get_score(conf)
                
                for angle in [-15, 15]:
                    rot_img = rotate_image(base_img, angle)
                    r_num, r_conf = predict_single(rot_img, f"{base_suffix}_{angle}")
                    
                    if r_num is not None:
                        r_score = get_score(r_conf)
                        if r_score > best_score:
                            best_num = r_num
                            best_conf = r_conf
                            best_score = r_score
                            
                return best_num, best_conf
 
            # Hybrid Fallback Strategy + Adaptive Rotation
            
            # Attempt 1: Super-Resolution
            crop_sr = preprocess_sr(raw_crop.copy())
            num, conf = predict_with_adaptive_rotation(crop_sr, "sr")
            
            # Attempt 2: Bicubic (Fallback if SR failed to find *anything* or result is low confidence?)
            # Let's stick to: Fallback if SR returned None.
            # Or maybe if SR returned Low confidence?
            # Let's be aggressive: If SR is not High/Medium, try Bicubic.
            
            if num is None:
                crop_bicubic = preprocess_bicubic(raw_crop.copy())
                num_bi, conf_bi = predict_with_adaptive_rotation(crop_bicubic, "bi")
                
                if num_bi is not None and conf_bi in ["high", "medium", "low"]:
                    # Fallback Succeeded!
                    num, conf = num_bi, conf_bi
                    print(f"  - Player {pid} Crop {i}: Fallback Success! {num} (Conf: {conf})")
            
            # Process Result
            if num is not None and conf in ["high", "medium", "low"]:
                val = str(num)
                print(f"  - Player {pid} Crop {i}: {val} (Conf: {conf}) [H={item['h']}, Blur={item['blur']:.1f}]")
                
                # Weighted Vote
                weight = 1
                if conf == "high": weight = 3
                elif conf == "medium": weight = 2
                
                votes.append({"val": val, "weight": weight})
            else:
                print(f"  - Player {pid} Crop {i}: Ignored (Num: {num}, Conf: {conf}) [H={item['h']}, Blur={item['blur']:.1f}]")
        
        # Weighted Voting Logic
        if not votes:
            final_num = "Unknown"
            conf = 0.0
        else:
            # Aggregate weights
            vote_counts = defaultdict(int)
            for v in votes:
                vote_counts[v["val"]] += v["weight"]
                
            # Find winner
            winner = max(vote_counts, key=vote_counts.get)
            total_weight = sum(vote_counts.values())
            winner_weight = vote_counts[winner]
            
            conf = winner_weight / total_weight
            
            # Debug
            print(f"[identity] Player {pid} Weighted Votes: {dict(vote_counts)} -> Winner: {winner} (Conf: {conf:.2f})")
            final_num = winner
 
        # Store result
        jersey_map[pid] = {"number": final_num, "conf": conf, "votes": votes}
 
    # Cleanup dir
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
    return jersey_map
