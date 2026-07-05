"""
OSNet ReID module — adapted from Nabeel's reid_extractor.py
Replaces SigLIP for player re-identification using OSNet x0.25.
"""
import cv2
import torch
import numpy as np
from torchvision import transforms

try:
    import torchreid
    _TORCHREID_OK = True
except ImportError:
    _TORCHREID_OK = False

WEIGHTS_PATH = "models/osnet_x0_25.pth"
SIMILARITY_THRESHOLD = 0.82  # cosine similarity — same as Nabeel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def _apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def _normalize(embedding):
    if embedding is None: return None
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else None


class OSNetReID:
    """
    Lightweight OSNet-based ReID for player tracking.
    Matches new detections to known players by appearance embedding.
    """

    def __init__(self):
        self._model = None
        self._memory = {}  # track_id -> embedding
        self._gallery = {}  # track_id -> [(frame, raw embedding snapshot), ...]
        self._update_counts = {}  # track_id -> number of update() calls
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if not _TORCHREID_OK:
            print("⚠️ [OSNetReID] torchreid not installed — ReID disabled")
            return
        try:
            import os
            if not os.path.exists(WEIGHTS_PATH):
                print(f"⚠️ [OSNetReID] Weights not found at {WEIGHTS_PATH} — ReID disabled")
                return
            self._model = torchreid.models.build_model(
                name="osnet_x0_25",
                num_classes=4101,
                pretrained=False
            )
            checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
            self._model.load_state_dict(checkpoint)
            self._model.to(device)
            self._model.eval()
            self._loaded = True
            print(f"✅ [OSNetReID] OSNet x0.25 loaded on {device}")
        except Exception as e:
            print(f"⚠️ [OSNetReID] Failed to load: {e}")

    def extract_embedding(self, crop):
        """Extract appearance embedding from a player crop (BGR image)."""
        if self._model is None:
            return None
        try:
            if crop is None or crop.size == 0:
                return None
            h, w = crop.shape[:2]
            if h < 80 or w < 30:
                return None

            crop = cv2.resize(crop, (256, 512))
            crop = _apply_clahe(crop)
            crop = cv2.GaussianBlur(crop, (3, 3), 0)

            # Multi-region weighted embedding (Nabeel's approach)
            regions = [
                (crop[0:int(512*0.45), int(256*0.15):int(256*0.85)], 1.40),  # upper body
                (crop[int(512*0.18):int(512*0.82), int(256*0.10):int(256*0.90)], 1.25),  # middle
                (crop[int(512*0.45):int(512*0.98), int(256*0.10):int(256*0.90)], 0.85),  # lower
                (crop, 1.10),  # full body
            ]

            embeddings = []
            for region, weight in regions:
                if region.size == 0:
                    continue
                rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                tensor = transform(rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = self._model(tensor)
                emb = feat.cpu().numpy().flatten()
                emb = _normalize(emb)
                if emb is not None:
                    embeddings.append(emb * weight)

            if not embeddings:
                return None
            final = _normalize(np.mean(embeddings, axis=0))
            return final.astype(np.float32) if final is not None else None

        except Exception as e:
            return None

    def find_match(self, embedding):
        """Find best matching track_id from memory. Returns (track_id, similarity) or (None, 0)."""
        if embedding is None or not self._memory:
            return None, 0.0
        best_id, best_sim = None, 0.0
        for tid, stored_emb in self._memory.items():
            sim = float(np.dot(embedding, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = tid
        if best_sim >= SIMILARITY_THRESHOLD:
            return best_id, best_sim
        return None, best_sim

    # Gallery snapshots per track: the EMA above smooths appearance into ONE
    # vector, which blurs pose/lighting variation — offline identity merging
    # needs multiple raw snapshots spread over the track's life. Exponential
    # spacing front-loads early looks and still covers long tracks.
    _GALLERY_SNAPSHOT_AT = (1, 4, 10, 25, 60, 140, 320, 700)

    def update(self, track_id, embedding, frame=None):
        """Update memory with new embedding for a track."""
        if embedding is None:
            return
        if track_id in self._memory:
            # Exponential moving average — smooths appearance over time
            self._memory[track_id] = _normalize(
                0.7 * self._memory[track_id] + 0.3 * embedding
            )
        else:
            self._memory[track_id] = embedding

        cnt = self._update_counts.get(track_id, 0) + 1
        self._update_counts[track_id] = cnt
        if cnt in self._GALLERY_SNAPSHOT_AT:
            self._gallery.setdefault(track_id, []).append((frame, embedding))

    def get_gallery(self, track_id):
        """[(frame_or_None, embedding), ...] raw snapshots for a track."""
        return self._gallery.get(track_id, [])

    def remove(self, track_id):
        self._memory.pop(track_id, None)
        self._gallery.pop(track_id, None)
        self._update_counts.pop(track_id, None)

    def reset(self):
        self._memory.clear()
        self._gallery.clear()
        self._update_counts.clear()

    @classmethod
    def create(cls):
        """Factory — loads model lazily."""
        instance = cls()
        instance._load()
        return instance
