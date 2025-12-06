# ============================================================
# FULL FOOTBALL STATS + xG PIPELINE → PLAYERS_FLAT + CSV
# Memory-aware version (limits stored images)
# ============================================================

import os, json, math, csv, pymysql, hashlib, uuid, sys
from collections import defaultdict, Counter
from pymysql.cursors import DictCursor

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ======================= ENV SETTINGS =======================

SOURCE_VIDEO_PATH = os.getenv("SRC_VIDEO", None)

DET_WEIGHTS = os.getenv(
"DET_WEIGHTS",
"/home/ubuntu/new/models/detect.pt"
)

BALL_MODEL_PATH = os.getenv(
    "BALL_MODEL_PATH",
    None
)

POSE_WEIGHTS = os.getenv(
"POSE_WEIGHTS",
"/home/ubuntu/new/models/pose.pt"
)

JNR_WEIGHTS = os.getenv(
"JNR_WEIGHTS",
"/home/ubuntu/new/models/jersey.pt"
)

JNR_IMG_SIZE = int(os.getenv("JNR_IMG_SIZE", "160"))
JNR_GATE = float(os.getenv("JNR_GATE", "0.35"))

CONF_DET = float(os.getenv("DET_CONF", "0.25"))
IOU_DET = float(os.getenv("DET_IOU", "0.50"))

MAX_FRAMES_TRACK = os.getenv("MAX_TRACK_FRAMES", "") # "3000" or ""

OUT_JSON = os.getenv("OUT_JSON", "players_stats.json")
OUT_CSV = os.getenv("OUT_CSV", "players_stats.csv")

# new: image size / stride / memory control
DET_IMG_SIZE = int(os.getenv("DET_IMG_SIZE", "832")) # 0 = let YOLO decide
VID_STRIDE = int(os.getenv("VID_STRIDE", "1")) # 1 = every frame
STORE_IMAGES = int(os.getenv("STORE_IMAGES", "1")) # 1 = keep some imgs, 0 = no orig_img
STORE_IMAGES_UP_TO = int(os.getenv("STORE_IMAGES_UP_TO", "1500")) # how many frames keep orig_img

# DB Settings
MYSQL_HOST = os.getenv("MYSQL_HOST", "db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "25060"))
MYSQL_USER = os.getenv("MYSQL_USER", "scoutbridge")
MYSQL_PASS = os.getenv("MYSQL_PASSWORD", "***REMOVED_SECRET***")
MYSQL_DB   = os.getenv("MYSQL_DB", "footballgallery")
ANALYSIS_TABLE = os.getenv("TABLE_NAME", "MatchesVideoAnalysis_test")

# SBG Settings
SBG_BASE      = os.getenv("SBG_BASE", "https://api-staging.scoutbridge.net/football-gallery/api").rstrip("/")
SBG_LIST_URL  = f"{SBG_BASE}/v2/files/list/video/for-match-analysis"
SBG_TOKEN     = os.getenv("SBG_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjQzMzAzNDAsInN1YiI6ImFudG9uaW9qaGFuY2Vkcmljays1QGdtYWlsLmNvbSIsInVzZXJfaWQiOiI0MjRkYzI2ZiJ9.vwl3-QkomqmN3TIs-TFfJTe5wZ-B4o9EDMFk_q_ZTI0")

CLASS = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}


# ===================== BASIC HELPERS =====================

def _conn():
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS, database=MYSQL_DB,
        cursorclass=DictCursor, autocommit=True
    )

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def upsert_status_row(matches_video_id, user_id, source_url, status, task_id,
                      validation_status_id=None, analysis=None, error=None):
    unique_id = _sha1(source_url or f"{matches_video_id or ''}")
    mv_id_num = int(matches_video_id) if (matches_video_id is not None and str(matches_video_id).isdigit()) else None
    payload = analysis.copy() if isinstance(analysis, dict) else (analysis or {})
    if isinstance(payload, dict):
        payload.setdefault("matches_video_key", str(matches_video_id) if matches_video_id is not None else None)
        payload.setdefault("source_url", source_url)
        payload.setdefault("user_id", user_id)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",",":")) if payload is not None else None

    try:
        sel_sql = f"SELECT id FROM {ANALYSIS_TABLE} WHERE unique_id=%s LIMIT 1"
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(sel_sql, (unique_id,))
            row = cur.fetchone()
            if row:
                upd_sql = f"""
                UPDATE {ANALYSIS_TABLE}
                   SET matches_video_id=%s, user_id=%s, validation_status_id=%s,
                       source_url=%s, task_id=%s, status=%s, analysis=CAST(%s AS JSON),
                       error=%s, updated_at=NOW()
                 WHERE id=%s
                """
                cur.execute(upd_sql, (mv_id_num, user_id, validation_status_id, source_url,
                                      task_id, status, payload_json, error, row["id"]))
            else:
                ins_sql = f"""
                INSERT INTO {ANALYSIS_TABLE}
                  (matches_video_id, user_id, unique_id, validation_status_id,
                   source_url, task_id, status, analysis, error, created_at, updated_at)
                VALUES
                  (%s, %s, %s, %s, %s, %s, %s, CAST(%s AS JSON), %s, NOW(), NOW())
                """
                cur.execute(ins_sql, (mv_id_num, user_id, unique_id, validation_status_id,
                                      source_url, task_id, status, payload_json, error))
    except Exception as e:
        print(f"[db] Failed to upsert status: {e}")

def save_analysis_to_db(payload):
    # Extract metadata from payload
    matches_video_id = payload.get("matches_video_id")
    user_id = payload.get("user_id")
    source_url = payload.get("source_url") or payload.get("video_path")
    # Generate a task_id if not present
    task_id = uuid.uuid4().hex[:32]
    
    if not matches_video_id:
        print("[db] No matches_video_id provided. Skipping DB save (Debug Mode).")
        return

    print(f"[db] Saving analysis for mv_id={matches_video_id}...")
    upsert_status_row(
        matches_video_id=matches_video_id,
        user_id=user_id,
        source_url=source_url,
        status="finished",
        task_id=task_id,
        analysis=payload
    )

def exist(p):
    if p.startswith("http://") or p.startswith("https://"):
        print(f"[check] URL detected: {p}")
        return True
    ok = os.path.exists(p)
    print(f"[check] {'OK' if ok else 'MISSING'}: {p}")
    return ok




def pick_device(user_device=0):
    if torch.cuda.is_available():
        print(f"[env] CUDA available -> using device {user_device}")
        return user_device
    print("[env] CUDA not available -> using CPU")
    return "cpu"


def get_fps(path, default=25):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = int(round(fps or default))
    print(f"[video] RAW FPS: {fps}")
    return fps


def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _inc(d, k, v=1):
    d[k] = d.get(k, 0) + v


# ================= HSV FEATURES FOR TEAM COLORS =================

def _hsv_feat(hsv_patch):
    if hsv_patch.size == 0:
        return np.empty((0, 4), np.float32)
    H, S, V = cv2.split(hsv_patch)
    grass = (H > 35) & (H < 90) & (S > 70) & (V > 60)
    dark = (V < 40)
    mask = ~(grass | dark)
    if mask.sum() < 30:
        mask[:] = True
    h = H[mask].astype(np.float32) * (np.pi / 180.0)
    s = S[mask].astype(np.float32) / 255.0
    v = V[mask].astype(np.float32) / 255.0
    hc = np.cos(h); hs = np.sin(h)
    return np.stack([hc, hs, s, v], axis=1).astype(np.float32)


def _hue_from_cos_sin(hc, hs):
    return float((np.arctan2(hs, hc) * 180 / np.pi) % 180)


def _circ_dist(h1, h2):
    d = abs(h1 - h2)
    return min(d, 180 - d)


# ===================== TRACKING (YOLO.track) =====================

def track_video_with_yolo(video_path, weights_path, conf=0.25, iou=0.5, device=0, max_frames=None, vid_stride=1, imgsz=None, store_images=True, store_images_up_to=1500, store_interval=30, store_crops=True, crop_interval=1, ball_model_path=None):
    model = YOLO(weights_path)
    ball_model = YOLO(ball_model_path) if ball_model_path else None
    
    use_half = torch.cuda.is_available() and str(device).lower() != "cpu"
    print(f"[track] vid_stride={vid_stride}, imgsz={imgsz}, half={use_half}, store_crops={store_crops}, crop_interval={crop_interval}, ball_model={ball_model_path}")
    
    kwargs = dict(source=video_path, tracker="bytetrack.yaml", conf=conf, iou=iou, stream=True, verbose=False, device=device, persist=True, vid_stride=vid_stride, half=use_half)
    if imgsz not in (None, 0):
        kwargs["imgsz"] = imgsz
    kwargs["save"] = store_images
        
    stream = model.track(**kwargs)
    
    # If using ball model, we need a separate stream for it
    ball_stream = None
    if ball_model:
        ball_kwargs = kwargs.copy()
        # Use lower confidence for ball if specified, or default to same
        # For now, let's hardcode a lower default for ball if not passed?
        # Or better, accept ball_conf arg.
        # Since I can't easily change the signature in all calls, I'll just lower it here if it's the generic model.
        # Actually, let's just use 0.15 for ball if conf is higher.
        if conf > 0.15:
            ball_kwargs["conf"] = 0.15
            print(f"[track] Lowering ball model confidence to {ball_kwargs['conf']}")
        
        ball_stream = ball_model.track(**ball_kwargs)

    frames, n, with_ball, with_players = [], 0, 0, 0
    unique_ids = set()
    max_frames_int = int(max_frames) if (max_frames and str(max_frames).isdigit()) else None
    
    # Iterate streams
    # If ball_stream exists, we zip.
    iterator = zip(stream, ball_stream) if ball_stream else stream
    
    for item in iterator:
        if ball_stream:
            res, res_ball = item
        else:
            res = item
            
        n += 1
        if store_images:
            plotted = res.plot()
            if plotted is not None:
                # cv2.imwrite(f"debug_frames/frame_{n}.jpg", plotted)
                pass
            else:
                print(f"[debug] Frame {n}: plotted is None")
            # Memory management
            if n % 1000 == 0:
                import gc
                gc.collect()
                
            if n % 100 == 0:
                # print(f"[debug] Processing frame {n}, store_images={store_images}")
                pass
        if n % 100 == 0:
             print(f"[debug] Processing frame {n}, store_images={store_images}")
        if max_frames_int and n > max_frames_int:
            break
        if max_frames_int and n > max_frames_int:
            break
        img = res.orig_img
        frame = {"orig_shape": img.shape[:2], "path": getattr(res, "path", None), "boxes": [], "crops": []}
        
        # Only store full image if explicitly requested and within limits
        if store_images and (n <= store_images_up_to or n % store_interval == 0):
            frame["orig_img"] = img
            
        # Process Main Model (Players/GK/Ref)
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            b = res.boxes
            ids = (b.id.cpu().numpy().astype(int).tolist() if b.id is not None else [None] * len(b))
            cls = b.cls.cpu().numpy().astype(int).tolist()
            confs = b.conf.cpu().numpy().astype(float).tolist()
            xyxy = b.xyxy.cpu().numpy().astype(float).tolist()
            for i in range(len(cls)):
                # Skip ball from main model if we are using a dedicated ball model
                if ball_model and cls[i] == CLASS["ball"]:
                    continue
                    
                box = {"id": ids[i], "cls": cls[i], "conf": confs[i], "xyxy": xyxy[i]}
                frame["boxes"].append(box)
                
                # Store crop logic...
                if store_crops and box["cls"] in (CLASS["player"], CLASS["goalkeeper"]):
                    if (n - 1) % crop_interval == 0:
                        crop = _torso_crop(img, box["xyxy"])
                        if crop is not None and crop.size > 0:
                            frame["crops"].append({"box_idx": len(frame["boxes"])-1, "img": crop})
                            if len(frame["crops"]) == 1 and n % 100 == 0:
                                print(f"[debug] Stored crops for frame {n}")

        # Process Ball Model
        if ball_stream and hasattr(res_ball, "boxes") and res_ball.boxes is not None and len(res_ball.boxes) > 0:
            b = res_ball.boxes
            ids = (b.id.cpu().numpy().astype(int).tolist() if b.id is not None else [None] * len(b))
            cls = b.cls.cpu().numpy().astype(int).tolist()
            confs = b.conf.cpu().numpy().astype(float).tolist()
            xyxy = b.xyxy.cpu().numpy().astype(float).tolist()
            for i in range(len(cls)):
                if cls[i] == 32: # Sports ball
                    # Map to our internal class 0
                    # We need a unique ID for the ball. 
                    # If YOLOv8x tracks it, use that ID + offset to avoid collision?
                    # Or just use the raw ID? Player IDs are usually 1-1000.
                    # Let's add 10000 to ball IDs to be safe.
                    ball_id = ids[i] + 10000 if ids[i] is not None else None
                    box = {"id": ball_id, "cls": CLASS["ball"], "conf": confs[i], "xyxy": xyxy[i]}
                    frame["boxes"].append(box)

        if any(bb["cls"] == CLASS["ball"] for bb in frame["boxes"]):
            with_ball += 1
        if any(bb["cls"] in (CLASS["player"], CLASS["goalkeeper"]) for bb in frame["boxes"]):
            with_players += 1
        
        # Update unique IDs
        for bb in frame["boxes"]:
            if bb["id"] is not None: unique_ids.add(bb["id"])
            
        if n % 100 == 0:
            print(f"[track] frames={n} | with ball={with_ball} | with players={with_players} | unique IDs={len(unique_ids)}")
            
        frames.append(frame)
    print(f"[track] frames={n} | with ball={with_ball} | with players={with_players} | unique IDs={len(unique_ids)}")
    if n == 0:
        raise RuntimeError("No frames were read. Check the video path/codec.")
    return frames, model


# ================= TEAM ASSIGNMENT (COLORS) =================

def _hue_to_basic_color(h):
    if h is None:
        return "unknown"
    if h < 10 or h >= 170:
        return "red"
    elif h < 25:
        return "orange"
    elif h < 35:
        return "gold"
    elif h < 55:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 105:
        return "cyan"
    elif h < 135:
        return "blue"
    elif h < 155:
        return "purple"
    else:
        return "magenta"


def _describe_color(h, s, v):
    if s < 0.25:
        if v > 0.75:
            return "white"
        elif v < 0.35:
            return "black"
        else:
            return "gray"
    base = _hue_to_basic_color(h)
    if v > 0.80:
        return f"light_{base}"
    elif v < 0.35:
        return f"dark_{base}"
    else:
        return base


def _center_crop(img, fraction=0.5):
    """
    Take the central fraction of the image to avoid background.
    """
    if img is None or img.size == 0: return img
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    dy, dx = int(h * fraction / 2), int(w * fraction / 2)
    return img[cy-dy:cy+dy, cx-dx:cx+dx]

def assign_teams_to_ids_from_frames(frames, sample_frames=1000, class_players=(2,)):
    """
    1. Collect color samples from player crops (using KMeans to find 2 dominant colors).
    2. Assign each ID to a team (0 or 1).
    3. Determine label (color name) for each team.
    """
    # We'll collect (h,s,v) for each person ID
    id_colors = {} # id -> list of (h,s,v)

    # Gather crops
    # We want to sample 'sample_frames' evenly from the video
    total_frames = len(frames)
    if total_frames == 0: return {}, {}
    
    step = max(1, total_frames // sample_frames)
    
    for i in range(0, total_frames, step):
        frame_data = frames[i]
        
        # Check for pre-stored crops (memory optimized)
        if "crops" in frame_data and frame_data["crops"]:
            for crop_data in frame_data["crops"]:
                box_idx = crop_data["box_idx"]
                if box_idx < len(frame_data["boxes"]):
                    b = frame_data["boxes"][box_idx]
                    pid = b["id"]
                    cls = b["cls"]
                    
                    if pid is None: continue
                    pid = int(pid)
                    if pid < 0: continue
                    if int(cls) not in class_players: continue
                    
                    crop = crop_data["img"]
                    if crop is None or crop.size == 0: continue
                    
                    # Use central crop for color to avoid background
                    color_crop = _center_crop(crop, fraction=0.4)
                    
                    # Get dominant color
                    hsv = cv2.cvtColor(color_crop, cv2.COLOR_BGR2HSV)
                    h = np.median(hsv[:,:,0])
                    s = np.median(hsv[:,:,1]) / 255.0
                    v = np.median(hsv[:,:,2]) / 255.0
                    
                    if pid not in id_colors: id_colors[pid] = []
                    id_colors[pid].append([h, s, v])
                    
        elif "orig_img" in frame_data:
            img = frame_data["orig_img"]
            if img is None: continue

            # For each detection
            # boxes: list of dicts with id, cls, conf, xyxy
            if "boxes" not in frame_data: 
                continue
            
            for b in frame_data["boxes"]:
                pid = b["id"]
                cls = b["cls"]
                xyxy = b["xyxy"]
                
                if pid is None: continue
                pid = int(pid)
                if pid < 0: continue
                if int(cls) not in class_players: continue
                
                # Crop
                crop = _torso_crop(img, xyxy)
                if crop is None or crop.size == 0: continue
                
                # Use central crop for color to avoid background
                color_crop = _center_crop(crop, fraction=0.4)
                
                # Get dominant color
                # fast way: mean or median of center
                # Convert to HSV
                hsv = cv2.cvtColor(color_crop, cv2.COLOR_BGR2HSV)
                # median
                h = np.median(hsv[:,:,0])
                s = np.median(hsv[:,:,1]) / 255.0
                v = np.median(hsv[:,:,2]) / 255.0
                
                if pid not in id_colors: id_colors[pid] = []
                id_colors[pid].append([h, s, v])
            
    # print(f"[teams] Collected colors for {len(id_colors)} IDs.")

    hues_by_id = defaultdict(list)
    sv_by_id = defaultdict(list)
    for pid, colors in id_colors.items():
        for h, s, v in colors:
            hues_by_id[pid].append(h)
            sv_by_id[pid].append((s, v))

    if len(hues_by_id) < 2:
        print("[teams] Not enough player hues.")
        return {}, {}
    id_list = sorted(hues_by_id.keys())
    per_id_features = []
    per_id_hsv = {}
    for pid in id_list:
        h_med = float(np.median(hues_by_id[pid]))
        s_med = float(np.median([sv[0] for sv in sv_by_id[pid]])) if sv_by_id[pid] else 0.5
        v_med = float(np.median([sv[1] for sv in sv_by_id[pid]])) if sv_by_id[pid] else 0.5
        per_id_hsv[pid] = (h_med, s_med, v_med)
        h_rad = h_med * (np.pi / 90.0)
        per_id_features.append([math.cos(h_rad), math.sin(h_rad), s_med, v_med])
    per_id_features = np.array(per_id_features, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(per_id_features, 2, None, criteria, 10, flags)
    assign = {pid: int(labels[i][0]) for i, pid in enumerate(id_list)}
    counts = Counter(assign.values())
    if counts[0] < counts[1]:
        assign = {pid: (0 if lab == 1 else 1) for pid, lab in assign.items()}
    cluster_hsv = {0: [], 1: []}
    for pid, team_id in assign.items():
        cluster_hsv[team_id].append(per_id_hsv[pid])
    team_labels = {}
    for team_id in (0, 1):
        if not cluster_hsv[team_id]:
            team_labels[team_id] = f"team_{team_id}"
            continue
        # hs = np.array(cluster_hsv[team_id]) # Original line, not needed with list comprehensions
        median_h = np.median([c[0] for c in cluster_hsv[team_id]])
        median_s = np.median([c[1] for c in cluster_hsv[team_id]])
        median_v = np.median([c[2] for c in cluster_hsv[team_id]])
        
        # Use 10th percentile saturation to detect White teams that have noisy samples
        p10_s = np.percentile([c[1] for c in cluster_hsv[team_id]], 10)
        
        print(f"[teams] Team {team_id}: h={median_h:.1f}, s={median_s:.2f} (p10={p10_s:.2f}), v={median_v:.2f}")
        
        # If p10_s indicates white, use it
        if p10_s < 0.25 and median_v > 0.70:
            label = "white"
        else:
            label = _describe_color(median_h, median_s, median_v)
            
        team_labels[team_id] = label
    print(f"[teams] assignment: {len(assign)} ids; labels={team_labels}")
    return assign, team_labels


# ===================== CUSTOM JNR (KERAS) =====================

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from itertools import groupby

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def ctc_decoder(predictions):
    text_list = []
    pred_indcies = np.argmax(predictions, axis=-1)
    for i in range(pred_indcies.shape[0]):
        ans = ""
        merged_list = [k for k,_ in groupby(pred_indcies[i])]
        for p in merged_list:
            if p != len(char_list):
                ans += char_list[int(p)]
        text_list.append(ans)
    return text_list

def get_model():
    # Re-implementation of the model architecture from src/models/model.py
    # to avoid loading config issues.
    inputs = layers.Input(shape=(32, 64, 1), name="image")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    conv_1 = layers.Conv2D(32, (3,3), kernel_initializer="he_uniform" ,activation = "selu", padding='same')(inputs)
    pool_1 = layers.MaxPool2D(pool_size=(2, 2))(conv_1)

    conv_2 = layers.Conv2D(64, (3,3), activation = "selu", padding='same')(pool_1)
    pool_2 = layers.MaxPool2D(pool_size=(2, 2))(conv_2)

    conv_3 = layers.Conv2D(128, (3,3), activation = "selu", padding='same')(pool_2)
    conv_4 = layers.Conv2D(128, (3,3), activation = "selu", padding='same')(conv_3)

    pool_4 = layers.MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = layers.Conv2D(256, (3,3), activation = "selu", padding='same')(pool_4)

    batch_norm_5 = layers.BatchNormalization()(conv_5)

    conv_6 = layers.Conv2D(256, (3,3), activation = "selu", padding='same')(batch_norm_5)
    batch_norm_6 = layers.BatchNormalization()(conv_6)
    pool_6 = layers.MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = layers.Conv2D(64, (2,2), activation = "selu")(pool_6)
    # squeezed = layers.Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    # Use Reshape instead of Lambda to avoid shape inference issues
    squeezed = layers.Reshape((-1, 64))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = layers.Bidirectional(layers.GRU(128, return_sequences=True , dropout=0.3))(squeezed)
    blstm_2 = layers.Bidirectional(layers.GRU(128, return_sequences=True , dropout=0.3))(blstm_1)

    softmax_output = layers.Dense(len(char_list) + 1, activation = 'softmax', name="dense")(blstm_2)

    output = CTCLayer(name="ctc_loss")(labels, softmax_output)

    # optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
    model = keras.models.Model(inputs=[inputs, labels], outputs=output)
    # model.compile(optimizer = optimizer)
    return model

class CustomJNR:
    def __init__(self, weights_path, gate=0.5):
        self.ok = False
        self.gate = gate
        self.model = None
        if not os.path.exists(weights_path):
            print(f"[jnr] Custom model weights not found: {weights_path}")
            return
        try:
            # Instantiate model first
            full_model = get_model()
            # Load weights
            full_model.load_weights(weights_path)
            
            # Extract inference model
            # full_model.inputs should be [image_input, label_input]
            # We want image_input -> dense_output
            
            img_input = full_model.inputs[0]
            dense_output = full_model.get_layer("dense").output
            
            self.model = keras.models.Model(inputs=img_input, outputs=dense_output)
            
            self.ok = True
            print(f"[jnr] Custom Keras model loaded (inference only) from {weights_path}")
        except Exception as e:
            print(f"[jnr] Failed to load Custom Keras model: {e}")

    def preprocess(self, img):
        if img is None or img.size == 0:
            return None
        # Resize to (64, 32) -> (W, H) for cv2.resize
        img = cv2.resize(img, (64, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=-1) # (32, 64, 1)
        # Normalize to 0-1 range if the model expects it. 
        # Based on typical CRNN training, often inputs are normalized.
        # The provided data loader didn't show explicit normalization, but let's try /255.0
        img = img / 255.0
        return img

    def predict_batch(self, crops_bgr):
        if not self.ok or not crops_bgr:
            return [{"number": None, "conf": 0.0} for _ in crops_bgr]
        
        batch_imgs = []
        valid_indices = []
        results = [{"number": None, "conf": 0.0} for _ in crops_bgr]
        
        for i, img in enumerate(crops_bgr):
            p_img = self.preprocess(img)
            if p_img is not None:
                batch_imgs.append(p_img)
                valid_indices.append(i)
                
        if not batch_imgs:
            return results
            
        batch_imgs = np.array(batch_imgs)
        try:
            preds = self.model.predict(batch_imgs, verbose=0)
            decoded_texts = ctc_decoder(preds)
            
            for idx, text in enumerate(decoded_texts):
                original_idx = valid_indices[idx]
                probs = preds[idx]
                max_probs = np.max(probs, axis=-1)
                conf = float(np.mean(max_probs)) if len(max_probs) > 0 else 0.0
                
                if text == "" or text == "-1":
                    results[original_idx] = {"number": None, "conf": conf}
                else:
                    results[original_idx] = {"number": text, "conf": conf}
        except Exception as e:
            print(f"[jnr] Prediction error: {e}")
            
        return results

# ===================== PADDLE JNR (FALLBACK) =====================

class PaddleJNR:
    def __init__(self, weights=None, device="cpu", img_size=160, gate=0.70):
        self.ok = False
        self.gate = float(gate)
        use_gpu = False
        # if isinstance(device, int):
        #     use_gpu = True
        # elif isinstance(device, str):
        #     use_gpu = (device.lower() != "cpu")
        print(f"[jnr] Forcing PaddleOCR to CPU to avoid CUDA conflicts.")
        try:
            import logging
            logging.getLogger("ppocr").setLevel(logging.ERROR)
            from paddleocr import PaddleOCR
            self.ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')
            self.ok = True
            print(f"[jnr] PaddleOCR initialized. use_gpu={use_gpu}")
        except Exception as e:
            print(f"[jnr] Failed to initialize PaddleOCR: {e}")
            self.ok = False

    def preprocess_for_ocr(self, img):
        if img is None or img.size == 0:
            return img
        h, w = img.shape[:2]
        target_h = 128
        if h < target_h:
            scale = target_h / float(h)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img

    def predict_batch(self, crops_bgr):
        if not self.ok or not crops_bgr:
            return [{"number": None, "conf": 0.0} for _ in crops_bgr]
        results = []
        for raw_img in crops_bgr:
            if raw_img is None or raw_img.size == 0:
                results.append({"number": None, "conf": 0.0})
                continue
            img_processed = self.preprocess_for_ocr(raw_img)
            candidates = []
            gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bin_img_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            imgs_to_try = [img_processed, cv2.bitwise_not(img_processed), gray_bgr, bin_img_bgr]
            for img in imgs_to_try:
                try:
                    res_list = self.ocr_model.ocr(img)
                except:
                    res_list = None
                if res_list and isinstance(res_list, list):
                    for res in res_list:
                        if isinstance(res, dict):
                            texts = res.get('rec_texts', [])
                            scores = res.get('rec_scores', [])
                            if texts and scores:
                                for text, score in zip(texts, scores):
                                    digits = "".join(filter(str.isdigit, text))
                                    if len(digits) > 0:
                                        candidates.append((digits, float(score)))
                        elif isinstance(res, list) and len(res) > 0:
                             for line in res:
                                if len(line) == 2 and isinstance(line[1], (list, tuple)):
                                    text, conf = line[1]
                                    digits = "".join(filter(str.isdigit, text))
                                    if len(digits) > 0:
                                        candidates.append((digits, float(conf)))
            best_number = None
            best_conf = 0.0
            for num, conf in candidates:
                if conf > best_conf:
                    best_conf = conf
                    best_number = num
            if best_number is not None and best_conf < self.gate:
                best_number = None
            results.append({"number": best_number, "conf": float(best_conf)})
        return results

class EasyOCRJNR:
    def __init__(self, gate=0.5):
        self.ok = False
        self.gate = float(gate)
        try:
            import easyocr
            # 'en' for English. gpu=False to avoid conflict for now
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.ok = True
            print("[jnr] EasyOCR initialized.")
        except Exception as e:
            print(f"[jnr] Failed to initialize EasyOCR: {e}")
            self.ok = False

    def preprocess(self, img):
        if img is None or img.size == 0: return img
        # Upscale significantly for OCR
        h, w = img.shape[:2]
        scale = 4.0 
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        
        return img

    def predict_batch(self, crops_bgr):
        if not self.ok or not crops_bgr:
            return [{"number": None, "conf": 0.0} for _ in crops_bgr]
        results = []
        for raw_img in crops_bgr:
            if raw_img is None or raw_img.size == 0:
                results.append({"number": None, "conf": 0.0})
                continue
                
            img = self.preprocess(raw_img)
            
            try:
                # allowlist='0123456789' to restrict to digits
                res = self.reader.readtext(img, allowlist='0123456789')
            except:
                res = []
            
            best_number = None
            best_conf = 0.0
            
            # res is list of (bbox, text, prob)
            for _, text, conf in res:
                if conf > best_conf:
                    best_conf = conf
                    best_number = text
            
            if best_number is not None and best_conf < self.gate:
                best_number = None
                
            results.append({"number": best_number, "conf": float(best_conf)})
        return results

class EnsembleJNR:
    def __init__(self, paddle_gate=0.5, easy_gate=0.5):
        # self.paddle = PaddleJNR(gate=paddle_gate)
        # self.ok = self.paddle.ok or self.easy.ok
        
        # PaddleOCR causes hangs on some environments (CPU/Memory issues)
        # Disabling it for stability, relying on EasyOCR.
        self.paddle = None 
        self.easy = EasyOCRJNR(gate=easy_gate)
        self.ok = self.easy.ok
        
    def predict_batch(self, crops_bgr):
        if not self.ok:
             return [{"number": None, "conf": 0.0} for _ in crops_bgr]
             
        # res_paddle = self.paddle.predict_batch(crops_bgr)
        res_easy = self.easy.predict_batch(crops_bgr)
        
        final_results = []
        # for p, e in zip(res_paddle, res_easy):
        for e in res_easy:
            # p_num = p["number"]
            # p_conf = p["conf"]
            
            e_num = e["number"]
            e_conf = e["conf"]
            
            # Fallback logic simplified to just EasyOCR
            final_results.append(e)
                
        return final_results


def _torso_crop(img, xyxy):
    Himg, Wimg = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    H = y2 - y1; W = x2 - x1
    if H <= 0 or W <= 0:
        return None
    yA = y1 + int(0.20 * H); yB = y1 + int(0.70 * H)
    xA = x1 + int(0.20 * W); xB = x1 + int(0.80 * W)
    yA = max(0, yA); yB = min(Himg, yB)
    xA = max(0, xA); xB = min(Wimg, xB)
    if yB <= yA or xB <= xA:
        return None
    return img[yA:yB, xA:xB]


def jersey_numbers_from_frames_jnr(
    frames,
    jnr_model,
    class_players=(CLASS["player"], CLASS["goalkeeper"]),
    sample_every=5,
    top_k=12,
    min_box_h=30
):
    if jnr_model is None or not getattr(jnr_model, "ok", False):
        print("[jersey] JNR disabled; all jerseys = Unknown.")
        return {}
    buckets = defaultdict(list)
    print(f"[jersey] Starting JNR on {len(frames)} frames...")
    crops_found_count = 0
    for t, f in enumerate(frames):
        # Option A: Use pre-stored crops (Memory efficient)
        if "crops" in f and f["crops"]:
            crops_found_count += 1
            for c in f["crops"]:
                box_idx = c["box_idx"]
                if box_idx >= len(f["boxes"]): continue
                b = f["boxes"][box_idx]
                if b["id"] is None: continue
                
                # Check box height (score)
                x1, y1, x2, y2 = map(int, b["xyxy"])
                box_h = (y2 - y1)
                if box_h < min_box_h: continue
                
                crop = c["img"]
                score = float(box_h)
                buckets[b["id"]].append((score, crop))
            continue

        # Option B: Re-crop from orig_img (Legacy / Full Image Mode)
        if "orig_img" not in f:
            continue
        if t % sample_every != 0:
            continue
        img = f["orig_img"]
        for b in f["boxes"]:
            if b["id"] is None or b["cls"] not in class_players:
                continue
            x1, y1, x2, y2 = map(int, b["xyxy"])
            box_h = (y2 - y1)
            if box_h < min_box_h:
                continue
            crop = _torso_crop(img, b["xyxy"])
            if crop is None or crop.size == 0:
                continue
            score = float(box_h)
            buckets[b["id"]].append((score, crop))
            
    print(f"[jersey] Found pre-stored crops in {crops_found_count} frames.")

    # keep best top_k crops per player
    for pid in list(buckets.keys()):
        buckets[pid].sort(key=lambda z: z[0], reverse=True)
        buckets[pid] = [p for _, p in buckets[pid][:top_k]]

    jersey_map = {}
    debug_stats = {"total_ids": len(buckets), "with_nums": 0, "low_conf": 0, "no_preds": 0}
    
    for pid, crops in buckets.items():
        preds = jnr_model.predict_batch(crops)
        nums = [pr["number"] for pr in preds if pr["number"] is not None]
        if not nums:
            debug_stats["no_preds"] += 1
            continue
        
        cnt = Counter(nums)
        best_num, votes = cnt.most_common(1)[0]
        best_conf = max(pr["conf"] for pr in preds if pr["number"] == best_num)
        
        jersey_map[pid] = {
            "number": best_num,
            "conf": round(float(best_conf), 3),
            "votes": int(votes),
            "samples": int(len(nums)),
        }
        debug_stats["with_nums"] += 1

    print(f"[jersey] JNR stats: {debug_stats}")
    print(f"[jersey] JNR detected jerseys for {len(jersey_map)} player ids.")
    return jersey_map


# ===================== POSE → PITCH POLYGON =====================

def infer_pitch_polygon_from_pose(
    frames,
    pose_weights,
    device=0,
    conf=0.25,
    sample_every=10,
    max_sample_frames=None
):
    if not (pose_weights and os.path.exists(pose_weights)):
        print("[pose] POSE_WEIGHTS missing; pitch polygon disabled.")
        return None
    try:
        pose_model = YOLO(pose_weights)
    except Exception as e:
        print(f"[pose] failed to load pose model: {e}")
        return None
    pts = []
    
    # Iterate through frames. If max_sample_frames is set, use it as a cap, 
    # otherwise look at all frames (that have images).
    limit = min(len(frames), max_sample_frames) if max_sample_frames else len(frames)
    
    for idx in range(0, limit, sample_every):
        f = frames[idx]
        if "orig_img" not in f:
            continue
        img = f["orig_img"]
        res_list = pose_model.predict(img, conf=conf, device=device, verbose=False)
        if not res_list:
            continue
        r = res_list[0]
        if not hasattr(r, "keypoints") or r.keypoints is None:
            continue
        kps = r.keypoints
        if hasattr(kps, "xy"):
            arr = kps.xy.cpu().numpy()
            arr = arr.reshape(-1, 2)
            for p in arr:
                x, y = float(p[0]), float(p[1])
                if not (np.isnan(x) or np.isnan(y)):
                    pts.append([x, y])
    if len(pts) < 10:
        print("[pose] Not enough keypoints to build pitch polygon.")
        return None
    pts = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    print(f"[pose] pitch polygon with {len(hull)} vertices inferred.")
    return hull


def apply_pitch_polygon_to_ball(frames, pitch_poly):
    if pitch_poly is None or len(pitch_poly) < 3:
        print("[pose] No pitch polygon -> ball not filtered.")
        return
    for f in frames:
        new_boxes = []
        for b in f["boxes"]:
            if b["cls"] != CLASS["ball"]:
                new_boxes.append(b)
                continue
            c = bbox_center(b["xyxy"])
            inside = cv2.pointPolygonTest(
                pitch_poly.astype(np.float32),
                (float(c[0]), float(c[1])),
                False
            )
            if inside >= 0:
                new_boxes.append(b)
        f["boxes"] = new_boxes
    print("[pose] Applied pitch polygon filter to ball detections.")


# ===================== xG + SHOTS HELPERS =====================

def _sigmoid(x):
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _goal_mouth_centers(w, h):
    # Relaxed to 12% to catch balls when camera pans and goal is not at absolute edge
    return (np.array([0.12 * w, 0.50 * h]), np.array([0.88 * w, 0.50 * h]))


def _goal_posts(w, h):
    left_post, right_post = _goal_mouth_centers(w, h)
    return left_post, right_post


def _angle_to_goal(pt, w, h):
    lp, rp = _goal_posts(w, h)
    v1 = lp - np.array(pt); v2 = rp - np.array(pt)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 1.0
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.acos(cosang)
    return ang / math.pi


def _distance_to_nearest_goal(pt, w, h):
    gl, gr = _goal_mouth_centers(w, h)
    return min(np.linalg.norm(np.array(pt) - gl), np.linalg.norm(np.array(pt) - gr))


def _nearest_opponent_distance(frame, pid, team_map):
    my_team = team_map.get(pid, None)
    me_c = None
    for b in frame["boxes"]:
        if b["id"] == pid:
            me_c = bbox_center(b["xyxy"]); break
    if me_c is None:
        return 1e9
    best = 1e9
    for b in frame["boxes"]:
        if b["cls"] not in (CLASS["player"], CLASS["goalkeeper"]):
            continue
        if b["id"] is None or b["id"] == pid:
            continue
        if team_map.get(b["id"], None) == my_team:
            continue
        c = bbox_center(b["xyxy"])
        d = math.hypot(me_c[0] - c[0], me_c[1] - c[1])
        best = min(best, d)
    return best


def _is_header(frame, pid, ball_xy):
    for b in frame["boxes"]:
        if b["id"] == pid:
            x1, y1, x2, y2 = map(int, b["xyxy"])
            head_y = y1 + int(0.15 * (y2 - y1))
            return ball_xy is not None and ball_xy[1] <= head_y
    return False


def _estimate_xg_from_features(
    dist_px,
    angle_norm,
    header=False,
    under_pressure=False,
    w=1920
):
    dist_norm = max(0.0, min(1.0, dist_px / max(1.0, 0.6 * w)))
    base = _sigmoid(-3.2 + 4.8 * (1.0 - dist_norm) + 1.8 * angle_norm)
    if header:
        base *= 0.80
    if under_pressure:
        base *= 0.75
    return float(max(0.02, min(0.75, base)))


def detect_shots_and_xg(
    frames,
    ownership,
    team_map,
    fps=25,
    pose_weights=None,
    speed_px_thr_frac=0.015,
    near_goal_frac=0.50,
    opp_thr_px=90
):
    h, w = frames[0]["orig_shape"]
    ball = [None] * len(frames)
    for t, f in enumerate(frames):
        balls = [b for b in f["boxes"] if b["cls"] == CLASS["ball"]]
        ball[t] = bbox_center(balls[0]["xyxy"]) if balls else None

    vx, vy, v = [0.0] * len(frames), [0.0] * len(frames), [0.0] * len(frames)
    for t in range(1, len(frames)):
        if ball[t] is None or ball[t - 1] is None:
            v[t] = 0.0
            continue
        vx[t] = ball[t][0] - ball[t - 1][0]
        vy[t] = ball[t][1] - ball[t - 1][1]
        v[t] = math.hypot(vx[t], vy[t])

    speed_thr = speed_px_thr_frac * w
    shots = []

    def _in_goal_strip(pt):
        if pt is None:
            return False
        x, y = pt
        # Relaxed threshold to 12% of width to handle camera panning
        in_left = (x <= 0.12 * w and 0.15 * h <= y <= 0.85 * h)
        in_right = (x >= 0.88 * w and 0.15 * h <= y <= 0.85 * h)
        return in_left or in_right

    # Pose model for jumping detection (lazy load if needed, but here we might just skip for speed or use heuristic)
    # For now, we'll use a heuristic based on bounding box height/position relative to goal.
    
    start = 0
    for t in range(1, len(ownership) + 1):
        if t == len(ownership) or ownership[t] != ownership[t - 1]:
            pid = ownership[t - 1]
            end = t - 1
            if pid is not None:
                lo = max(start, 1)
                # Look ahead slightly (e.g. 3 frames) to catch velocity spike after leaving foot
                hi = min(end + 3, len(frames) - 1)
                if hi >= lo:
                    t_star = max(range(lo, hi + 1), key=lambda k: v[k])
                    if ball[t_star] is not None and v[t_star] >= speed_thr:
                        dist = _distance_to_nearest_goal(ball[t_star], w, h)
                        if dist <= near_goal_frac * w:
                            ang = _angle_to_goal(ball[t_star], w, h)
                            under_pressure = (_nearest_opponent_distance(frames[t_star], pid, team_map) <= opp_thr_px)
                            is_header = _is_header(frames[t_star], pid, ball[t_star])

                            on_target = False
                            goal_t = None
                            end_t = min(len(frames) - 1, t_star + int(0.6 * fps))
                            for tt in range(t_star, end_t + 1):
                                if _in_goal_strip(ball[tt]):
                                    on_target = True
                                    goal_t = tt
                                    break
                            
                            is_goal = False
                            if on_target and goal_t is not None:
                                # Heuristic: check frames after entering goal. 
                                # If ball stays in goal or disappears, likely GOAL.
                                # If ball comes back to pitch (x > 0.05w and x < 0.95w) quickly, likely SAVE/POST.
                                check_duration = int(1.5 * fps) # 1.5 seconds
                                ch_end = min(len(frames)-1, goal_t + check_duration)
                                returned_to_pitch = False
                                for c_idx in range(goal_t + 1, ch_end + 1):
                                    b_pos = ball[c_idx]
                                    if b_pos is not None:
                                        bx = b_pos[0]
                                        # Must return comfortably to pitch (buffer zone > 15%)
                                        if bx > 0.15 * w and bx < 0.85 * w:
                                            returned_to_pitch = True
                                            break
                                
                                if not returned_to_pitch:
                                    is_goal = True

                            xg = _estimate_xg_from_features(
                                dist, ang,
                                header=is_header,
                                under_pressure=under_pressure,
                                w=w
                            )
                            
                            # GK Stats Logic
                            is_save = on_target and not is_goal
                            save_details = {
                                "range": "mid", # default
                                "jumping": False,
                                "type": "regular",
                                "gk_id": None
                            }
                            
                            if is_save:
                                # 1. Range
                                # dist is in pixels. w is width (e.g. 1920). Pitch is ~105m.
                                # 15m ~= 15/105 * w ~= 0.14 * w
                                # 30m ~= 30/105 * w ~= 0.28 * w
                                if dist < 0.14 * w:
                                    save_details["range"] = "close"
                                elif dist < 0.28 * w:
                                    save_details["range"] = "mid"
                                else:
                                    save_details["range"] = "long"
                                    
                                # 2. Type (Penalty/Freekick/Corner)
                                # Based on shot start position ball[t_star]
                                bx, by = ball[t_star]
                                nx, ny = bx/w, by/h
                                if (nx < 0.05 or nx > 0.95) and (ny < 0.05 or ny > 0.95):
                                    save_details["type"] = "corner"
                                elif 0.45 <= nx <= 0.55 and (0.10 <= ny <= 0.15 or 0.85 <= ny <= 0.90):
                                    # Rough penalty spot check
                                    save_details["type"] = "penalty"
                                elif not under_pressure and dist > 0.20 * w:
                                    # Stationary-ish start? Hard to tell. Assume freekick if far and no pressure?
                                    save_details["type"] = "freekick"
                                    
                                # 3. Find GK and Jumping
                                # Look at frame at goal_t (save time)
                                if goal_t:
                                    gf = frames[goal_t]
                                    # Find GK closest to ball
                                    best_gk = None
                                    min_gk_dist = 1e9
                                    ball_at_save = ball[goal_t]
                                    if ball_at_save:
                                        for b in gf["boxes"]:
                                            if b["cls"] == CLASS["goalkeeper"]:
                                                c = bbox_center(b["xyxy"])
                                                d = math.hypot(c[0] - ball_at_save[0], c[1] - ball_at_save[1])
                                                if d < min_gk_dist:
                                                    min_gk_dist = d
                                                    best_gk = b
                                    
                                    if best_gk:
                                        save_details["gk_id"] = best_gk["id"]
                                        # Jumping Heuristic:
                                        # If GK bbox bottom is significantly higher than... ?
                                        # Or if bbox aspect ratio is different?
                                        # Let's use a simple height check relative to image height? No.
                                        # Let's assume if the save is "high" (ball y is high/low depending on view)
                                        # Assuming side view? No, usually broadcast view.
                                        # Jumping usually means y2 (bottom) is less than some ground baseline.
                                        # Hard to know ground baseline.
                                        # Let's skip jumping for now or default to False unless we add pose.
                                        pass

                            shots.append({
                                "type": "shot",
                                "t": t_star,
                                "pid": pid,
                                "team": team_map.get(pid, None),
                                "xg": xg,
                                "header": bool(is_header),
                                "opponent_present": bool(under_pressure),
                                "on_target": bool(on_target),
                                "is_penalty": save_details["type"] == "penalty",
                                "is_goal": is_goal,
                                "is_save": is_save,
                                "save_details": save_details
                            })
            start = t

    xg_player = defaultdict(lambda: defaultdict(float))
    for s in shots:
        if s["header"] and not s["opponent_present"]:
            key = "xg_header_no_opponent"
        elif s["header"] and s["opponent_present"]:
            key = "xg_header_opponent_present"
        elif (not s["header"]) and (not s["opponent_present"]):
            key = "xg_foot_no_opponent"
        else:
            key = "xg_foot_opponent_present"
        xg_player[s["pid"]][key] += float(s["xg"])

    xg_team = defaultdict(lambda: defaultdict(float))
    for pid, buckets in xg_player.items():
        tid = team_map.get(pid, None)
        for k, vv in buckets.items():
            xg_team[tid][k] += vv

    return shots, xg_player, xg_team


# ===================== PHASE 1/2 BASIC STATS =====================

def phase1_stats(frames, team_map, fps=25, max_owner_px=300, tackle_px=100, challenge_px=100):
    def _nearest_player_local(ball_xy, player_boxes):
        best, best_d = None, 1e9
        for p in player_boxes:
            c = bbox_center(p["xyxy"])
            d = math.hypot(ball_xy[0] - c[0], ball_xy[1] - c[1])
            if d < best_d: best, best_d = p, d
        return (best["id"] if best else None), best_d
    def _center_at(f, pid):
        for b in f["boxes"]:
            if b["id"] == pid: return bbox_center(b["xyxy"])
        return None
    events, ownership = [], []
    player_stats = defaultdict(lambda: defaultdict(int))
    team_stats = defaultdict(lambda: defaultdict(int))
    
    for t, f in enumerate(frames):
        players = [b for b in f["boxes"] if b["cls"] in (CLASS["player"], CLASS["goalkeeper"]) and b["id"] is not None]
        balls = [b for b in f["boxes"] if b["cls"] == CLASS["ball"]]
        owner = None
        if players and balls:
            # Find the ball that is closest to ANY player
            best_ball_idx = -1
            best_pid = None
            min_dist = 1e9
            
            for b_idx, b in enumerate(balls):
                x1, y1, x2, y2 = b["xyxy"]
                b_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                pid, d = _nearest_player_local(b_xy, players)
                if d < min_dist:
                    min_dist = d
                    best_pid = pid
                    best_ball_idx = b_idx
            
            if best_ball_idx != -1:
                # Use the best ball for stats
                x1, y1, x2, y2 = balls[best_ball_idx]["xyxy"]
                ball_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                pid = best_pid
                d = min_dist
                
                if pid is not None:
                    if max_owner_px is None or d <= max_owner_px: owner = pid
            
            # Challenges Logic: Check for opposing players near the ball
            nearby_players = []
            for p in players:
                c = bbox_center(p["xyxy"])
                dist = math.hypot(ball_xy[0] - c[0], ball_xy[1] - c[1])
                if dist <= challenge_px:
                    nearby_players.append(p["id"])
            
            teams_present = set()
            for p_id in nearby_players:
                tid = team_map.get(p_id, None)
                if tid is not None: teams_present.add(tid)
            
            if len(teams_present) >= 2:
                # Challenge occurring
                for p_id in nearby_players:
                    _inc(player_stats[p_id], "challenges")
                    tid = team_map.get(p_id, None)
                    tid = team_map.get(p_id, None)
                    if tid is not None: _inc(team_stats[tid], "challenges")
        
        ownership.append(owner)

    # Smooth ownership to reduce flickering
    # 1. Forward fill None values (up to 5 frames)
    smoothed_ownership = list(ownership)
    last_owner = None
    gap = 0
    for i in range(len(smoothed_ownership)):
        if smoothed_ownership[i] is not None:
            last_owner = smoothed_ownership[i]
            gap = 0
        elif last_owner is not None and gap < 5:
            smoothed_ownership[i] = last_owner
            gap += 1
    
    # 2. Remove short segments (length < 3)
    # If a segment is short, replace it with the previous owner
    merged_ownership = []
    if smoothed_ownership:
        current_owner = smoothed_ownership[0]
        current_len = 0
        for pid in smoothed_ownership:
            if pid == current_owner:
                current_len += 1
            else:
                # Segment ended
                if current_len < 3 and merged_ownership: # Short segment, merge to prev
                    pass
                current_owner = pid
                current_len = 1
    
    # Simpler approach: Iterate and build segments. If short, convert to prev owner.
    final_ownership = []
    if smoothed_ownership:
        segments = []
        curr = smoothed_ownership[0]
        count = 1
        for pid in smoothed_ownership[1:]:
            if pid == curr:
                count += 1
            else:
                segments.append({"pid": curr, "count": count})
                curr = pid
                count = 1
        segments.append({"pid": curr, "count": count})
        
        # Merge short segments
        for i in range(1, len(segments)):
            if segments[i]["count"] < 3:
                # Merge into previous ONLY if previous is not None
                if segments[i-1]["pid"] is not None:
                    segments[i]["pid"] = segments[i-1]["pid"]
                # Else, keep it (or merge into next? forward fill should have handled this)
        
        # Reconstruct
        for seg in segments:
            final_ownership.extend([seg["pid"]] * seg["count"])
            
    ownership = final_ownership if final_ownership else smoothed_ownership
    
    prev = None
    for t, own in enumerate(ownership):
        if own is None: continue
        if prev is None: prev = own; continue
        if own != prev:
            ft = team_map.get(prev, None)
            tt = team_map.get(own, None)
            e = {"type": "pass_attempt", "t": t, "from": prev, "to": own, "from_team": ft, "to_team": tt, "completed": ft == tt}
            events.append(e)
            if (e["from_team"] is not None and e["to_team"] is not None and e["from_team"] != e["to_team"]):
                f = frames[min(t, len(frames) - 1)]
                c_from = _center_at(f, e["from"]); c_to = _center_at(f, e["to"])
                if c_from and c_to:
                    d = math.hypot(c_from[0] - c_to[0], c_from[1] - c_to[1])
                    if d <= tackle_px:
                        events.append({"type": "tackle", "t": t, "by": e["to"], "on": e["from"], "by_team": e["to_team"]})
                        # Also count as challenge won for the tackler
                        _inc(player_stats[e["to"]], "challenges_won")
                        tid = team_map.get(e["to"], None)
                        if tid is not None: _inc(team_stats[tid], "challenges_won")
                    else:
                        # print(f"[debug] Tackle rejected: dist {d:.1f} > {tackle_px}")
                        pass
        prev = own
    
    for pid in ownership:
        if pid is None: continue
        _inc(player_stats[pid], "touch_frames")
        tid = team_map.get(pid, None)
        if tid is not None: _inc(team_stats[tid], "touch_frames")

    for e in list(events):
        if e["type"] != "pass_attempt": continue
        
        # GK Pass Stats
        # Check if 'from' player is a goalkeeper
        # We need to look up the class of the player ID.
        # We can do this by checking the frame at e["t"]
        f_pass = frames[min(e["t"], len(frames)-1)]
        is_gk = False
        for b in f_pass["boxes"]:
            if b["id"] == e["from"] and b["cls"] == CLASS["goalkeeper"]:
                is_gk = True
                break
        
        if is_gk:
            _inc(player_stats[e["from"]], "gk_passes_total")
            if e["completed"]:
                _inc(player_stats[e["from"]], "gk_passes_accurate")
            
            # Long pass check (> 35m approx)
            # 35m ~= 35/105 * w ~= 0.33 * w
            f = frames[min(e["t"], len(frames) - 1)]
            c_from = _center_at(f, e["from"])
            c_to = _center_at(f, e["to"])
            if c_from and c_to:
                dist_px = math.hypot(c_from[0] - c_to[0], c_from[1] - c_to[1])
                w = f["orig_shape"][1]
                if dist_px > 0.33 * w:
                    _inc(player_stats[e["from"]], "gk_passes_long_total")
                    if e["completed"]:
                        _inc(player_stats[e["from"]], "gk_passes_long_accurate")

    for e in events:
        if e["type"] == "pass_attempt":
            _inc(player_stats[e["from"]], "passes_total")
            if e["completed"]: _inc(player_stats[e["from"]], "passes_completed")
            _inc(team_stats[e["from_team"]], "passes_total")
            if e["completed"]: _inc(team_stats[e["from_team"]], "passes_completed")
        elif e["type"] == "interception":
            _inc(player_stats[e["by"]], "interceptions_total")
            _inc(team_stats[e.get("by_team")], "interceptions_total")
        elif e["type"] == "tackle":
            _inc(player_stats[e["by"]], "tackles_total")
            _inc(team_stats[e.get("by_team")], "tackles_total")
    return {"events": events, "player_stats": player_stats, "team_stats": team_stats, "ownership": ownership}

def phase2_stats(frames, base, team_map, H=None, fps=25, dribble_min_px=25, opp_near_px=150, max_gap=25):
    events = list(base["events"])
    player_stats = defaultdict(lambda: defaultdict(int), {pid: dict(s) for pid, s in base["player_stats"].items()})
    team_stats = defaultdict(lambda: defaultdict(int), {tid: dict(s) for tid, s in base["team_stats"].items()})
    h, w = frames[0]["orig_shape"]
    def in_wide(pt): return (pt[0] < 0.15 * w) or (pt[0] > 0.85 * w)
    def in_box(pt): return (0.35 * w < pt[0] < 0.65 * w) and (0.25 * h < pt[1] < 0.85 * h)
    def _center_at(f, pid):
        for b in f["boxes"]:
            if b["id"] == pid: return bbox_center(b["xyxy"])
        return None
    def _nearest_opponent_dist(f, pid):
        my_team = team_map.get(pid, None)
        best = 1e9
        pc = _center_at(f, pid)
        if pc is None: return best
        for b in f["boxes"]:
            if b["cls"] not in (CLASS["player"], CLASS["goalkeeper"]): continue
            if b["id"] is None or b["id"] == pid: continue
            if team_map.get(b["id"], None) == my_team: continue
            c = bbox_center(b["xyxy"])
            d = math.hypot(pc[0] - c[0], pc[1] - c[1])
            best = min(best, d)
        return best
    for e in base["events"]:
        if e["type"] != "pass_attempt": continue
        t = e["t"]; f = frames[min(t, len(frames) - 1)]
        def center(pid):
            for b in f["boxes"]:
                if b["id"] == pid: return bbox_center(b["xyxy"])
            return None
        p_from = center(e["from"]); p_to = center(e["to"])
        if p_from and p_to and in_wide(p_from) and in_box(p_to):
            events.append({"type": "cross", "t": t, "from": e["from"], "to": e["to"], "completed": bool(e["completed"])})
            tid = team_map.get(e["from"], None)
            _inc(player_stats[e["from"]], "crosses_total")
            _inc(team_stats[tid], "crosses_total")
            if e["completed"]:
                _inc(player_stats[e["from"]], "crosses_accurate")
                _inc(team_stats[tid], "crosses_accurate")
    own = base.get("ownership", [])
    start = 0
    for t in range(1, len(own) + 1):
        if t == len(own) or own[t] != own[t - 1]:
            pid = own[t - 1]
            end = t - 1
            if pid is not None and end - start + 1 >= 3:
                f0 = frames[start]; f1 = frames[end]
                def _center_at_local(fr, pid_):
                    for bb in fr["boxes"]:
                        if bb["id"] == pid_: return bbox_center(bb["xyxy"])
                    return None
                c0 = _center_at_local(f0, pid); c1 = _center_at_local(f1, pid)
                if c0 and c1:
                    disp = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
                    near0 = _nearest_opponent_dist(f0, pid)
                    near1 = _nearest_opponent_dist(f1, pid)
                    if disp >= dribble_min_px and (near0 <= opp_near_px or near1 <= opp_near_px) and (end - start) <= max_gap:
                        conf = max(0.1, min(1.0, disp / 120.0))
                        events.append({"type": "dribble", "t_start": start, "t_end": end, "pid": pid, "conf": round(float(conf), 3)})
                        _inc(player_stats[pid], "dribbles_total")
                        _inc(team_stats[team_map.get(pid, None)], "dribbles_total")
                        _inc(player_stats[pid], "dribbles_successful")
                        _inc(team_stats[team_map.get(pid, None)], "dribbles_successful")
                        player_stats[pid]["dribbles_conf_sum"] = player_stats[pid].get("dribbles_conf_sum", 0.0) + float(conf)
        start = t
    return {"events": events, "player_stats": player_stats, "team_stats": team_stats, "ownership": base.get("ownership", [])}


# ===================== MAIN ORCHESTRATION =====================

def run_pipeline_to_players_json(
    video_path,
    weights_path,
    out_json="players_stats.json",
    out_csv="players_stats.csv",
    conf=0.25,
    iou=0.5,
    device=0,
    sample_frames=1200,
    H=None,
    max_track_frames=None,
    vid_stride=1,
    imgsz=None,
    store_images=True,
    store_images_up_to=1500,
    model=None, # Added model argument for memory cleanup
    matches_video_id=None,
    user_id=None,
    source_url=None,
    ball_model_path=None,
):
    if not (exist(video_path) and exist(weights_path)):
        raise FileNotFoundError("Video or weights path is wrong.")

    device = pick_device(device)
    raw_fps = get_fps(video_path)
    eff_fps = max(1, raw_fps // max(1, vid_stride))
    print(f"[video] effective FPS for stats: {eff_fps} (stride={vid_stride})")

    frames, model = track_video_with_yolo(
        video_path, weights_path,
        conf=conf,
        iou=iou,
        device=device,
        max_frames=max_track_frames,
        vid_stride=vid_stride,
        imgsz=imgsz,
        store_images=store_images,
        store_images_up_to=store_images_up_to,
        store_interval=30,
        store_crops=True,
        crop_interval=30,
        ball_model_path=ball_model_path,
    )

    # pitch polygon only if we still have some images OR if we can read from video
    pitch_poly = None
    if any("orig_img" in f for f in frames) or (video_path and exist(video_path)):
        # Look through the entire video (max_sample_frames=None) to find pitch keypoints
        # We rely on the fact that we stored images every `store_interval` frames.
        pitch_poly = infer_pitch_polygon_from_pose(
            frames, POSE_WEIGHTS, device=device,
            conf=0.25, sample_every=15, max_sample_frames=None
        )
        if pitch_poly is not None:
            apply_pitch_polygon_to_ball(frames, pitch_poly)

    # team colors – require some frames with orig_img OR crops
    if any("orig_img" in f for f in frames) or any("crops" in f and f["crops"] for f in frames):
        # Use a high sample_frames to ensure we get enough data
        team_map, team_labels = assign_teams_to_ids_from_frames(frames, sample_frames=20000)
    else:
        team_map, team_labels = {}, {}
        print("[teams] No frames with orig_img or crops; team assignment disabled.")

    if not team_labels:
        team_labels = {0: "team_A", 1: "team_B"}

    # jersey numbers only if we have images + weights
    jersey_map = {}
    skip_jnr = os.getenv("SKIP_JNR", "0") == "1"
    if not skip_jnr and (any("orig_img" in f for f in frames) or any("crops" in f and f["crops"] for f in frames)):
        # Memory cleanup before loading OCR models
        # Aggressively delete the YOLO model if passed
        if model is not None:
            print("[memory] Deleting YOLO model to free VRAM for JNR...")
            del model
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Use EnsembleJNR for best results
        print("[jersey] Initializing EnsembleJNR...")
        jnr_model = EnsembleJNR(paddle_gate=0.5, easy_gate=0.5)
        if jnr_model.ok:
            jersey_map = jersey_numbers_from_frames_jnr(frames, jnr_model)
            
            # --- CONFLICT RESOLUTION ---
            print("[jersey] Resolving number conflicts...")
            # Group by (team, number)
            # team_map maps pid -> team_id (0 or 1)
            # jersey_map maps pid -> {number, conf, ...}
            
            conflicts = defaultdict(list)
            for pid, info in jersey_map.items():
                num = info.get("number")
                if num in (None, "Unknown", ""): continue
                
                tid = team_map.get(pid)
                if tid is None: continue # Should not happen if team assignment worked
                
                conflicts[(tid, num)].append(pid)
            
            resolved_count = 0
            for (tid, num), pids in conflicts.items():
                if len(pids) > 1:
                    # Sort by confidence (descending)
                    # We use the 'conf' field from jersey_map
                    pids.sort(key=lambda p: jersey_map[p].get("conf", 0.0), reverse=True)
                    
                    winner = pids[0]
                    losers = pids[1:]
                    
                    print(f"[jersey] Conflict for Team {tid} Number {num}: Winner={winner} ({jersey_map[winner]['conf']}), Losers={losers}")
                    
                    for l in losers:
                        # Revert loser to Unknown
                        jersey_map[l]["number"] = "Unknown"
                        jersey_map[l]["original_number"] = num # Keep track for debug
                        resolved_count += 1
            
            print(f"[jersey] Resolved {resolved_count} conflicts.")
            # ---------------------------
            
        else:
            print("[jersey] EnsembleJNR failed to initialize.")
    else:
        print("[jersey] No frames with orig_img or crops; JNR skipped.")

    # basic stats (phase 1 & 2)
    p1 = phase1_stats(frames, team_map, fps=eff_fps, max_owner_px=300, tackle_px=100, challenge_px=100)
    p2 = phase2_stats(frames, p1, team_map, H=H, fps=eff_fps, dribble_min_px=25, opp_near_px=150, max_gap=25)

    # xG + shots
    shots, xg_player_map, xg_team_map = detect_shots_and_xg(
        frames, p2["ownership"], team_map, fps=eff_fps, pose_weights=POSE_WEIGHTS
    )
    
    # Aggregate GK Stats from Shots
    for s in shots:
        if s.get("is_save"):
            # If we identified a GK ID, attribute to them
            gk_id = s["save_details"].get("gk_id")
            
            # If no GK ID found (e.g. occlusion), try to infer from team?
            # If shot was by Team A, save was by Team B's GK.
            # But we might not know who Team B's GK is if not detected.
            # Let's stick to explicit GK detection for now.
            
            if gk_id is not None:
                _inc(p2["player_stats"][gk_id], "saves_total")
                
                # Range
                rng = s["save_details"].get("range")
                if rng == "close": _inc(p2["player_stats"][gk_id], "saves_close_range")
                elif rng == "mid": _inc(p2["player_stats"][gk_id], "saves_mid_range")
                elif rng == "long": _inc(p2["player_stats"][gk_id], "saves_long_range")
                
                # Jumping
                if s["save_details"].get("jumping"):
                    _inc(p2["player_stats"][gk_id], "saves_jumping")
                else:
                    _inc(p2["player_stats"][gk_id], "saves_no_jump")
                    
                # Type
                stype = s["save_details"].get("type")
                if stype == "penalty": _inc(p2["player_stats"][gk_id], "saves_penalty")
                elif stype == "freekick": _inc(p2["player_stats"][gk_id], "saves_freekick")
                elif stype == "corner": _inc(p2["player_stats"][gk_id], "saves_corner")
                
                # Fouls (Placeholder)
                # _inc(p2["player_stats"][gk_id], "fouls_total")

    # shot counts
    shot_counts = defaultdict(lambda: {"on": 0, "wide": 0, "pen": 0, "goals": 0})
    for s in shots:
        pid = s.get("pid")
        if pid is None:
            continue
        if s.get("is_goal", False):
            shot_counts[pid]["goals"] += 1
            # Also increment "goals_final" in player stats
            _inc(p2["player_stats"][pid], "goals_final")

        if s.get("is_penalty", False):
            shot_counts[pid]["pen"] += 1
        if s.get("on_target", False):
            shot_counts[pid]["on"] += 1
        else:
            shot_counts[pid]["wide"] += 1

    # ensure all tracked players appear
    all_player_ids = {
        b["id"]
        for f in frames
        for b in f["boxes"]
        if b["cls"] in (CLASS["player"], CLASS["goalkeeper"]) and b["id"] is not None
    }
    for pid in all_player_ids:
        if pid not in p2["player_stats"]:
            p2["player_stats"][pid] = defaultdict(int)

    players_flat = []
    # Filter and Deduplicate
    # We want to remove Unknowns and ensure unique jersey numbers per team.
    # Strategy: Collect all candidates, sort by "importance" (e.g. time on pitch or confidence), then pick unique.
    
    candidates = []
    for pid in sorted(p2["player_stats"].keys()):
        stats = p2["player_stats"][pid]
        j = jersey_map.get(pid, {"number": "Unknown", "conf": 0.0})
        jnum = None
        if j.get("number") not in (None, "Unknown", ""):
            s = "".join(ch for ch in j.get("number") if ch.isdigit())
            s = (s.lstrip("0") or "0")[-2:] if s else None
            if s and s not in ("0", "00"):
                jnum = s
        
        # SKIP if Unknown
        if jnum is None:
            continue
            
        jersey_number = jnum
        team_id = team_map.get(pid, None)
        team_name = team_labels.get(team_id, "unknown") if team_id is not None else "unknown"
        
        candidates.append({
            "pid": pid,
            "stats": stats,
            "jersey_number": jersey_number,
            "team_name": team_name,
            "team_id": team_id,
            "conf": j.get("conf", 0.0),
            "touch_frames": int(stats.get("touch_frames", 0))
        })

    # Sort candidates by touch_frames (descending) as a proxy for importance/correctness
    candidates.sort(key=lambda x: x["touch_frames"], reverse=True)
    
    seen_jerseys = set() # (team_id, jersey_number)
    
    players_flat = []
    for c in candidates:
        key = (c["team_id"], c["jersey_number"])
        if key in seen_jerseys:
            continue # Skip duplicate
        seen_jerseys.add(key)
        
        pid = c["pid"]
        stats = c["stats"]
        jersey_number = c["jersey_number"]
        team_name = c["team_name"]
        
        passes_total = int(stats.get("passes_total", 0))
        passes_completed = int(stats.get("passes_completed", 0))
        acc_pct = 100.0 * passes_completed / passes_total if passes_total > 0 else 0.0

        tackles_total = int(stats.get("tackles_total", 0))
        interceptions_total = int(stats.get("interceptions_total", 0))
        challenges = int(stats.get("challenges", 0))
        challenges_won = int(stats.get("challenges_won", 0))
        tackles_successful = tackles_total # Simplify for now
        ball_interceptions = interceptions_total

        drib_n = int(stats.get("dribbles_total", 0))
        drib_success = int(stats.get("dribbles_successful", drib_n))

        crosses_total = int(stats.get("crosses_total", 0))
        crosses_acc = int(stats.get("crosses_accurate", 0))

        goals = int(stats.get("goals_final", 0))

        touch_frames = int(stats.get("touch_frames", 0))
        time_on_ball_s = float(touch_frames / eff_fps) if eff_fps > 0 else 0.0

        xg_buckets = xg_player_map.get(pid, {})
        xg_foot_no_opponent = float(xg_buckets.get("xg_foot_no_opponent", 0.0))
        xg_header_no_opponent = float(xg_buckets.get("xg_header_no_opponent", 0.0))
        xg_foot_opponent_present = float(xg_buckets.get("xg_foot_opponent_present", 0.0))
        xg_header_opponent_present = float(xg_buckets.get("xg_header_opponent_present", 0.0))

        sc = shot_counts.get(pid, {"on": 0, "wide": 0, "pen": 0})

        row = {
            "team": team_name,
            "player_id": int(pid),
            "jersey_number": jersey_number,

            "shots_on_target": int(sc["on"]),
            "shots_wide": int(sc["wide"]),
            "penalty": int(sc["pen"]),

            "crosses": crosses_total,
            "crosses_accurate": crosses_acc,

            "dribbles": drib_n,
            "dribbles_successful": drib_success,

            "passes": passes_total,
            "accurate_passes_%": float(acc_pct),

            "challenges": challenges,
            "challenges_won": challenges_won,

            "tackles": tackles_total,
            "tackles_successful": tackles_successful,

            "ball_interceptions": ball_interceptions,
            "fouls": 0, # Placeholder

            "goals": goals,

            "xg_foot_no_opponent": xg_foot_no_opponent,
            "xg_header_no_opponent": xg_header_no_opponent,
            "xg_foot_opponent_present": xg_foot_opponent_present,
            "xg_header_opponent_present": xg_header_opponent_present,
            
            # GK Stats
            "saves_total": int(stats.get("saves_total", 0)),
            "saves_close_range": int(stats.get("saves_close_range", 0)),
            "saves_mid_range": int(stats.get("saves_mid_range", 0)),
            "saves_long_range": int(stats.get("saves_long_range", 0)),
            "saves_jumping": int(stats.get("saves_jumping", 0)),
            "saves_no_jump": int(stats.get("saves_no_jump", 0)),
            "saves_penalty": int(stats.get("saves_penalty", 0)),
            "saves_freekick": int(stats.get("saves_freekick", 0)),
            "saves_corner": int(stats.get("saves_corner", 0)),
            "gk_passes_total": int(stats.get("gk_passes_total", 0)),
            "gk_passes_accurate": int(stats.get("gk_passes_accurate", 0)),
            "gk_passes_long_accurate": int(stats.get("gk_passes_long_accurate", 0)),
            "fouls_total": int(stats.get("fouls_total", 0)),
        }
        players_flat.append(row)

    payload = {
        "players_flat": players_flat,
        "class_map": CLASS,
        "team_labels": team_labels,
        "video_path": video_path,
        "model_weights": str(weights_path),
        "fps": int(eff_fps),
        "matches_video_id": matches_video_id,
        "user_id": user_id,
        "source_url": source_url,
    }

    # Save to DB if enabled
    if MYSQL_HOST:
        save_analysis_to_db(payload)

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[write] {out_json}")

    if players_flat:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(players_flat[0].keys()))
            writer.writeheader()
            writer.writerows(players_flat)
        print(f"[write] {out_csv}")

    print(f"[summary] players={len(players_flat)}")
    return payload



MYSQL_DB   = os.getenv("MYSQL_DB", "footballgallery")
ANALYSIS_TABLE = os.getenv("TABLE_NAME", "MatchesVideoAnalysis_test")

# ===================== POLLING LOGIC =====================

import requests
import time

def fetch_pending_videos():
    print(f"[poll] Fetching from {SBG_LIST_URL}...")
    headers = {
        "Authorization": f"Bearer {SBG_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.get(SBG_LIST_URL, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("items", [])
        else:
            print(f"[poll] Error fetching videos: {resp.status_code} - {resp.text}")
            return []
    except Exception as e:
        print(f"[poll] Exception fetching videos: {e}")
        return []

def is_video_processed(matches_video_id, source_url):
    unique_id = _sha1(source_url or f"{matches_video_id or ''}")
    try:
        with _conn() as conn, conn.cursor() as cur:
            sql = f"SELECT status FROM {ANALYSIS_TABLE} WHERE unique_id=%s LIMIT 1"
            cur.execute(sql, (unique_id,))
            row = cur.fetchone()
            if row and row["status"] == "finished":
                return True
    except Exception as e:
        print(f"[db] Error checking status: {e}")
    return False

def poll_and_process():
    print("[poll] Starting polling loop...")
    while True:
        videos = fetch_pending_videos()
        print(f"[poll] Found {len(videos)} videos.")
        
        for v in videos:
            # print(f"[poll] Item keys: {list(v.keys())}") # Debug
            # API returns 'id' (hex string) and no userID
            raw_id = v.get("id")
            
            # Filter for specific IDs for this test
            TARGET_IDS = ["db70ab0188d14cf", "2fb9b21ef0fb442"]
            if raw_id not in TARGET_IDS:
                continue

            user_id = v.get("userID") # Likely None
            url = v.get("spacesURL")
            
            # Convert hex ID to int for DB
            mv_id = None
            if raw_id:
                try:
                    # Try as int first, then hex
                    if isinstance(raw_id, int) or raw_id.isdigit():
                        mv_id = int(raw_id)
                    else:
                        mv_id = int(raw_id, 16)
                except ValueError:
                    print(f"[poll] Could not convert ID {raw_id} to int. Using None.")
                    mv_id = None
            
            if not url:
                continue
                
            if is_video_processed(mv_id, url):
                print(f"[poll] Video {mv_id} already processed. Skipping.")
                continue
                
            print(f"[poll] Processing video {mv_id} from {url}")
            
            # Generate task_id
            task_id = uuid.uuid4().hex[:32]
            
            # Update status to running
            upsert_status_row(mv_id, user_id, url, "running", task_id=task_id, analysis={})
            
            try:
                # Download video locally to avoid streaming issues
                local_video_path = f"temp_{mv_id}.mp4"
                print(f"[poll] Downloading {url} to {local_video_path}...")
                try:
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        with open(local_video_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    print(f"[poll] Download complete.")
                except Exception as e:
                    print(f"[poll] Download failed: {e}")
                    upsert_status_row(mv_id, user_id, url, "failed", task_id=task_id, error=f"Download failed: {e}")
                    continue

                # Use unique CSV name for local dump
                unique_csv_name = f"output_{mv_id}_{task_id}.csv"
                
                payload = run_pipeline_to_players_json(
                    video_path=local_video_path,
                    weights_path=DET_WEIGHTS,
                    out_json=OUT_JSON, # JSON can be overwritten or we can make it unique too, but user asked for CSV
                    out_csv=unique_csv_name,
                    conf=CONF_DET,
                    iou=IOU_DET,
                    device=0,
                    sample_frames=2000, # Adjust as needed
                    H=None,
                    max_track_frames=MAX_FRAMES_TRACK,
                    vid_stride=VID_STRIDE,
                    imgsz=imgsz_arg,
                    store_images=bool(STORE_IMAGES),
                    store_images_up_to=STORE_IMAGES_UP_TO,
                    matches_video_id=mv_id,
                    user_id=user_id,
                    source_url=url,
                    ball_model_path=BALL_MODEL_PATH,
                )
                
                # Cleanup
                if os.path.exists(local_video_path):
                    os.remove(local_video_path)
                    print(f"[poll] Removed temp file {local_video_path}")
                
                print(f"[poll] Finished processing {mv_id}")
                
                # Log processed video
                with open("processed_videos.log", "a") as log_f:
                    log_f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ID: {mv_id} | Task: {task_id} | CSV: {unique_csv_name} | URL: {url}\n")
                
            except Exception as e:
                print(f"[poll] Error processing {mv_id}: {e}")
                import traceback
                traceback.print_exc()
                upsert_status_row(mv_id, user_id, url, "failed", task_id=task_id, error=str(e))
                
        print("[poll] Sleeping for 60 seconds...")
        time.sleep(60)

# ===================== SCRIPT ENTRYPOINT =====================

if __name__ == "__main__":
    # If SRC_VIDEO is set, run in single mode (legacy/debug)
    # Otherwise, run in polling mode
    
    if SOURCE_VIDEO_PATH:
        video_path_to_use = SOURCE_VIDEO_PATH
        is_url = SOURCE_VIDEO_PATH.startswith("http")
        
        if is_url:
            print(f"[main] SRC_VIDEO is a URL. Downloading to temp file...")
            temp_filename = f"temp_manual_{uuid.uuid4().hex[:8]}.mp4"
            try:
                with requests.get(SOURCE_VIDEO_PATH, stream=True) as r:
                    r.raise_for_status()
                    with open(temp_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"[main] Downloaded to {temp_filename}")
                video_path_to_use = temp_filename
            except Exception as e:
                print(f"[main] Failed to download video: {e}")
                sys.exit(1)
        elif not os.path.exists(SOURCE_VIDEO_PATH):
             print(f"[main] Local file {SOURCE_VIDEO_PATH} not found.")
             # Fallback to polling if not found? Or exit?
             # Let's exit to be clear.
             sys.exit(1)

        print(f"[main] Running in single file mode for {video_path_to_use}")
        imgsz_arg = DET_IMG_SIZE if DET_IMG_SIZE > 0 else None

        try:
            payload = run_pipeline_to_players_json(
                video_path=video_path_to_use,
                weights_path=DET_WEIGHTS,
                out_json=OUT_JSON,
                out_csv=OUT_CSV,
                conf=CONF_DET,
                iou=IOU_DET,
                device=0,
                sample_frames=2000,
                H=None,
                max_track_frames=MAX_FRAMES_TRACK,
                vid_stride=VID_STRIDE,
                imgsz=imgsz_arg,
                store_images=bool(STORE_IMAGES),
                store_images_up_to=STORE_IMAGES_UP_TO,
                ball_model_path=BALL_MODEL_PATH,
            )
        finally:
            if is_url and os.path.exists(video_path_to_use):
                print(f"[main] Cleaning up temp file {video_path_to_use}")
                os.remove(video_path_to_use)

    else:
        print("[main] No local SRC_VIDEO found or not set. Starting polling mode.")
        poll_and_process()
    print("\nDone.")
