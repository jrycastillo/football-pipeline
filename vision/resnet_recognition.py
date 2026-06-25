
import os
import string
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import cv2

# --- Custom ResNet32 Implementation (Matches Training) ---
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.inplanes = 16 # Start low? No, stem outputs 64 in my training script.
        # Training script used standard stem:
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet32(num_classes=100):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)

# --- Phase 224: ResNet34 Grayscale (New Model) ---
def create_resnet34_grayscale(num_classes=100, pretrained=False):
    """Create ResNet34 adapted for grayscale input."""
    # Note: pretrained=False because we will load our own weights
    model = models.resnet34(weights=None)
    
    # Modify first conv layer for 1 channel input
    # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


class ResNetRecognizerV2:
    def __init__(self, weights_path="output_models/resnet34_jnr_manual_rgb_strict/best_model.pt", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = [str(i) for i in range(100)] # 0-99
        self.pending_queue = [] # Queue for JNR
        self.idx_to_jersey = None  # None = identity mapping (old 100-class models)
        # Torso crop bounds (fraction of bounding-box height)
        self.torso_top = 0.0   # overridden for new-format models
        self.torso_bot = 0.60  # legacy: top 60%

        # --- Peek at checkpoint to detect new-format models ---
        checkpoint = None
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # New format saved by train_resnet_reshuffled.py / train on clean data
            n_classes      = checkpoint.get("n_classes", 54)
            idx_to_jersey  = checkpoint.get("idx_to_jersey", {})
            # Keys may be ints or strings depending on how torch serialised them
            self.idx_to_jersey = {int(k): int(v) for k, v in idx_to_jersey.items()}

            from torchvision import models as _tvm
            self.model = _tvm.resnet34(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)
            self.is_grayscale = False
            self.size = 128
            # ImageNet RGB normalisation (matches training)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # Training used torso_crop(top=0.15, bot=0.52) on the full bounding-box crop
            self.torso_top = 0.15
            self.torso_bot = 0.52

            self.model.load_state_dict(checkpoint["state_dict"])
            val_acc = checkpoint.get("val_acc", None)
            val_str = f"{val_acc:.1f}%" if val_acc is not None else "?"
            print(f"✅ [JNR] Loaded clean ResNet34 ({n_classes} classes, val={val_str}) from {weights_path}")

        else:
            # Legacy path-name-based detection
            if "rgb" in weights_path.lower():
                 # RGB Model (Phase 227)
                 from torchvision import models as _tvm
                 self.model = _tvm.resnet34(weights=None)
                 self.model.fc = nn.Linear(self.model.fc.in_features, 100)
                 self.is_grayscale = False
                 self.size = 128
                 self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                 print("✅ [JNR] Using RGB Color-Aware Model")
            elif "resnet34" in weights_path or "grayscale" in weights_path:
                 self.model = create_resnet34_grayscale(num_classes=100)
                 self.is_grayscale = True
                 self.size = 128
                 self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
            else:
                 # Legacy ResNet32 RGB
                 self.model = resnet32(num_classes=100)
                 self.is_grayscale = False
                 self.size = 224
                 self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            if checkpoint is not None:
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif isinstance(checkpoint, dict) and "state_dict" not in checkpoint:
                    # raw state_dict
                    self.model.load_state_dict(checkpoint)
                # else already handled above
                print(f"✅ Loaded JNR Model from {weights_path}")
            else:
                print(f"⚠️ Warning: Weights not found at {weights_path}. Model unsupervised.")

        self.model.to(self.device)
        self.model.eval()

        # --- Legibility gate ---
        self.leg_model = None
        self.leg_threshold = 0.65
        self.leg_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._init_legibility_model()

        # Phase 219: OpenCV DNN Super Resolution for tiny crops
        self.upscaler = None
        self._init_upscaler()
    
    def _init_legibility_model(self, path="models/legibility_resnet18.pt"):
        """Load ResNet18 binary legibility classifier (0=not_legible, 1=legible)."""
        if not os.path.exists(path):
            print(f"⚠️ [JNR] Legibility model not found at {path} — gate disabled")
            return
        from torchvision import models as _tvm
        leg = _tvm.resnet18(weights=None)
        leg.fc = nn.Linear(512, 2)
        leg.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        leg.eval()
        self.leg_model = leg.to(self.device)
        print(f"✅ [JNR] Legibility gate loaded from {path}")

    def _legibility_scores(self, torso_crops_bgr):
        """
        Score a list of BGR torso crops for legibility.
        Returns list of float (prob of being legible, 0-1).
        """
        if self.leg_model is None:
            return [1.0] * len(torso_crops_bgr)  # gate disabled → pass all through
        tensors = []
        for crop in torso_crops_bgr:
            try:
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensors.append(self.leg_transform(pil))
            except Exception:
                tensors.append(torch.zeros(3, 128, 128))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.leg_model(batch), dim=1)
        return probs[:, 1].cpu().tolist()  # prob of class 1 = legible

    def _init_upscaler(self):
        """Initialize OpenCV DNN 4x upscaler (ESPCN - fast and effective)."""
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            # Try ESPCN 4x first (fast and good quality)
            model_paths = [
                "models/ESPCN_x4.pb",
                "/home/ubuntu/football/models/ESPCN_x4.pb",
            ]
            for path in model_paths:
                if os.path.exists(path):
                    sr.readModel(path)
                    sr.setModel("espcn", 4)
                    self.upscaler = sr
                    print("✅ [JNR] OpenCV SR 4x upscaler initialized (ESPCN)")
                    return
            
            # If no model, try downloading
            print("⚠️ [JNR] SR model not found, using bicubic fallback")
            self.upscaler = None
        except Exception as e:
            print(f"⚠️ [JNR] Super-resolution not available: {e}")
            self.upscaler = None
    
    def _upscale_crop(self, img_bgr):
        """Upscale tiny crops using OpenCV DNN SR 4x."""
        h, w = img_bgr.shape[:2]
        # Only upscale small crops (under 60px in smallest dimension)
        if min(h, w) >= 60:
            return img_bgr
        
        if self.upscaler is not None:
            try:
                return self.upscaler.upsample(img_bgr)
            except Exception as e:
                pass
        
        # Fallback to bicubic 4x
        return cv2.resize(img_bgr, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

    def queue_request(self, track_id, crop, frame_idx):
        self.pending_queue.append((track_id, crop, frame_idx))

    def get_results(self):
        if not self.pending_queue: return []
        batch = self.pending_queue[:]
        self.pending_queue = [] # Clear queue
        
        crops = [x[1] for x in batch]
        tids = [x[0] for x in batch]
        
        # Batch Predict
        preds = self.predict_batch(crops)
        
        # Merge TIDs
        results = []
        for i, pred in enumerate(preds):
            pred["track_id"] = tids[i]
            results.append(pred)
        return results

    def _smart_pad(self, img_pil):
        w, h = img_pil.size
        # Use simple resize if aspect ratio is close to square? No, stick to padding
        # to preserve aspect ratio logic from training (augmentation used RandomAffine/Rotation which distorts, 
        # but training input was Resize((128,128)) which STRETCHES).
        # WAIT! My training transform was `transforms.Resize((128, 128))`.
        # This STRETCHES the image to square, distorting aspect ratio.
        # So I should probably STRETCH here too for consistency with training.
        # But `_smart_pad` preserved AR.
        # If I trained with AR distortion, I should inference with AR distortion.
        # Let's check training script: `transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))`
        # Yes, training STRETCHED.
        # So I should switch to Resize directly to match training distribution.
        
        return img_pil.resize((self.size, self.size), Image.Resampling.BILINEAR)

    def _preprocess_crop(self, img_bgr):
        """Phase 218: Training-Aligned Pre-processing with Torso Cropping
        
        Key insight: Full-body crops are too small for JNR. Focus on upper body/torso
        where jersey numbers appear (~top 60% of crop).
        """
        if img_bgr is None or img_bgr.size == 0: return None
        
        h, w = img_bgr.shape[:2]

        # Torso crop: use bounds set at init time to match training distribution
        # New model: 15-52% (excludes head, keeps jersey area)
        # Legacy model: 0-60% (top 60%)
        y1 = int(h * self.torso_top)
        y2 = int(h * self.torso_bot)
        if y2 - y1 > 20:
            img_bgr = img_bgr[y1:y2, :]
            h = y2 - y1
        
        # 0. Minimum crop size filter - reject crops too small to recognize
        if min(h, w) < 15:  # Very small crops are hopeless
            return None
        
        # Phase 219: Real-ESRGAN 4x upscaling for tiny crops
        # This provides AI-based super-resolution instead of bicubic interpolation
        if min(h, w) < 60:
            img_bgr = self._upscale_crop(img_bgr)
            h, w = img_bgr.shape[:2]
        
        # 2. Light Gaussian blur (MATCHES training augmentation)
        # Training used blur as data augmentation - align with that distribution
        blurred = cv2.GaussianBlur(img_bgr, (3, 3), 0.5)
        
        # 3. Lighter CLAHE for contrast (reduced from 2.0 to 1.5)
        # If Grayscale model, CLAHE on L channel is still good concept
        if self.is_grayscale:
             gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
             clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
             enhanced = clahe.apply(gray)
             return enhanced # Returns single channel gray
        else:
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))  # Lighter CLAHE
            l = clahe.apply(l)
            enhanced_lab = cv2.merge((l, a, b))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return enhanced_bgr # Returns BGR

    def predict_batch(self, images):
        """
        Args:
            images: List of BGR numpy arrays (crops).
        Returns:
            List of dicts: {"number": int, "confidence": float, "raw_text": str}
        """
        if not images: return []

        # --- Step 1: extract raw torso crops for legibility gate ---
        raw_torsos = []   # parallel to images; None if unusable
        for img in images:
            if isinstance(img, list):
                img = img[-1]
            if not isinstance(img, np.ndarray) or img.size == 0:
                raw_torsos.append(None)
                continue
            h = img.shape[0]
            y1, y2 = int(h * self.torso_top), int(h * self.torso_bot)
            torso = img[y1:y2, :] if (y2 - y1) > 6 else img
            raw_torsos.append(torso if torso.size > 0 else None)

        # --- Step 2: batch legibility check ---
        valid_torsos    = [(i, t) for i, t in enumerate(raw_torsos) if t is not None]
        leg_scores      = [0.0] * len(images)
        if valid_torsos:
            idxs, torsos = zip(*valid_torsos)
            scores = self._legibility_scores(list(torsos))
            for i, s in zip(idxs, scores):
                leg_scores[i] = s

        # --- Step 3: build JNR batch (legible crops only) ---
        batch_tensors = []
        valid_indices = []

        for i, img in enumerate(images):
            # Skip crops the legibility model flagged as unreadable
            if leg_scores[i] < self.leg_threshold:
                continue

            try:
                if isinstance(img, list):
                    img = img[-1]

                if not isinstance(img, np.ndarray) or img.size == 0:
                    continue

                # --- Smart Pre-processing (Low Res -> Super Res -> Contrast) ---
                img_proc = self._preprocess_crop(img)
                if img_proc is None: continue

                # Conversion to PIL
                if self.is_grayscale:
                    img_pil = Image.fromarray(img_proc, mode="L")
                else:
                    img_pil = Image.fromarray(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))

                # Resize / Pad
                img_padded = self._smart_pad(img_pil)

                # Transform
                batch_tensors.append(self.transform(img_padded))
                valid_indices.append(i)
                
                # Debug: Save Enhanced Crop for User Verification
                if os.environ.get("SAVE_ENHANCED_CROPS") == "1":
                     debug_dir = "output_production_ikorodu_resnet34/enhanced_crops"
                     os.makedirs(debug_dir, exist_ok=True)
                     import uuid
                     fname = f"{debug_dir}/{uuid.uuid4().hex[:8]}.jpg"
                     img_pil.save(fname)
                     
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                
        # Pre-fill results: crops that failed legibility gate get status="not_legible"
        results = []
        for i in range(len(images)):
            if leg_scores[i] < self.leg_threshold:
                results.append({"number": None, "confidence": 0.0, "status": "not_legible"})
            else:
                results.append({"number": None, "confidence": 0.0})

        if not batch_tensors:
            return results

        batch_stack = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_stack)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            # Entropy Calculation (User Request for "Unknown" Filter)
            # H(x) = -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10) # Avoid log(0)
            entropy = -torch.sum(probs * log_probs, dim=1)
            
            # Phase 224: Relaxed Heuristics for Better Model
            # Since this model is trained on 733k balanced images, we trust it more.
            # Reduced penalties and thresholds.
            
            CONFIDENCE_THRESHOLD = 0.40 # Keep moderate threshold
            ENTROPY_THRESHOLD = 4.0     # Little stricter entropy
            MIN_REJECT_CONF = 0.20      # Reject very low conf
            
            # Simplified Class Penalties (Only slight nudge against #1 bias if it persists)
            # The new model should have less bias.
            CLASS_PENALTIES = {1: 0.10} 
            
            for idx, conf, pred, ent, prob_row in zip(valid_indices, confs, preds, entropy, probs):
                pred_idx = int(pred.item())
                confidence = float(conf.item())
                ent_val = float(ent.item())

                # Map class index → actual jersey number (new-format models only)
                if self.idx_to_jersey is not None:
                    num = self.idx_to_jersey.get(pred_idx, pred_idx)
                else:
                    num = pred_idx

                # Apply slight penalty to jersey #1 (historically over-predicted)
                if num in CLASS_PENALTIES:
                     confidence *= (1.0 - CLASS_PENALTIES[num])

                # Logic: Only mark as Unknown if confidence is very low OR entropy very high
                if confidence < MIN_REJECT_CONF or ent_val > ENTROPY_THRESHOLD:
                    results[idx] = {
                        "number": None,
                        "confidence": 0.0,
                        "status": "unknown",
                        "entropy": ent_val,
                        "raw_text": "Unknown"
                    }
                else:
                    results[idx] = {
                        "number": num,
                        "confidence": confidence,
                        "status": "valid",
                        "entropy": ent_val,
                        "raw_text": str(num)
                    }

        return results


class PARSeqRecognizer:
    """
    Drop-in replacement for ResNetRecognizerV2 using PARSeq scene-text model.
    Same queue_request / get_results interface as ResNetRecognizerV2.
    """

    TORSO_TOP = 0.15
    TORSO_BOT = 0.52

    def __init__(self, weights_path="models/parseq_local_v5.pt", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pending_queue = []

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parseq_dir = os.path.join(base, "parseq")
        if parseq_dir not in sys.path:
            sys.path.insert(0, parseq_dir)

        from strhub.models.utils import create_model
        from strhub.data.module import SceneTextDataModule

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model = create_model("parseq", pretrained=False,
                             charset_train=string.digits, charset_test=string.digits,
                             max_label_length=2)
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        self.model = model.to(self.device)
        self.transform = SceneTextDataModule.get_transform((32, 128))

        self.leg_model = None
        self.leg_threshold = 0.65
        self.leg_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._init_legibility_model(os.path.join(base, "models/legibility_resnet18.pt"))
        print(f"✅ [JNR] PARSeq loaded from {weights_path}")

    def _init_legibility_model(self, path):
        if not os.path.exists(path):
            print(f"⚠️ [JNR] Legibility model not found at {path} — gate disabled")
            return
        leg = models.resnet18(weights=None)
        leg.fc = nn.Linear(512, 2)
        leg.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        leg.eval()
        self.leg_model = leg.to(self.device)
        print(f"✅ [JNR] Legibility gate loaded from {path}")

    def _torso(self, img_bgr):
        h = img_bgr.shape[0]
        y1, y2 = int(h * self.TORSO_TOP), int(h * self.TORSO_BOT)
        return img_bgr[y1:y2, :] if y2 - y1 >= 6 else img_bgr

    def queue_request(self, track_id, crop, frame_idx):
        self.pending_queue.append((track_id, crop, frame_idx))

    def get_results(self):
        if not self.pending_queue:
            return []
        batch = self.pending_queue[:]
        self.pending_queue = []
        crops = [x[1] for x in batch]
        tids  = [x[0] for x in batch]
        preds = self.predict_batch(crops)
        for i, pred in enumerate(preds):
            pred["track_id"] = tids[i]
        return preds

    def predict_batch(self, images):
        if not images:
            return []

        torsos = [self._torso(img) for img in images]

        leg_scores = [1.0] * len(torsos)
        if self.leg_model is not None:
            tensors = []
            for t in torsos:
                pil = Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
                tensors.append(self.leg_transform(pil))
            with torch.no_grad():
                leg_scores = torch.softmax(
                    self.leg_model(torch.stack(tensors).to(self.device)), dim=1
                )[:, 1].tolist()

        valid_idx = [i for i, s in enumerate(leg_scores) if s >= self.leg_threshold]

        if not valid_idx:
            return [{"number": None, "confidence": 0.0, "status": "not_legible"}] * len(images)

        tensors = []
        for i in valid_idx:
            pil = Image.fromarray(cv2.cvtColor(torsos[i], cv2.COLOR_BGR2RGB))
            tensors.append(self.transform(pil))

        with torch.no_grad():
            logits = self.model(torch.stack(tensors).to(self.device))
            probs  = logits[:, :3, :11].softmax(-1)
            preds_text, _ = self.model.tokenizer.decode(probs)
            confs = probs.max(-1).values.min(-1).values.tolist()

        parseq_results = {}
        for rank, i in enumerate(valid_idx):
            pred = preds_text[rank].strip()
            conf = confs[rank]
            if pred.isdigit() and 1 <= int(pred) <= 99 and conf >= 0.50:
                parseq_results[i] = {"number": int(pred), "confidence": conf,
                                     "status": "valid", "raw_text": pred}
            else:
                parseq_results[i] = {"number": None, "confidence": 0.0,
                                     "status": "unknown", "raw_text": pred}

        return [
            parseq_results.get(i, {"number": None, "confidence": 0.0, "status": "not_legible"})
            for i in range(len(images))
        ]
