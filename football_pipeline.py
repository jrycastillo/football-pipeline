# ============================================================
# FULL FOOTBALL STATS + xG PIPELINE → PLAYERS_FLAT + CSV
# Memory-aware version (limits stored images)
# ============================================================

import os, json, math, csv
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ======================= ENV SETTINGS =======================

SOURCE_VIDEO_PATH = os.getenv(
"SRC_VIDEO",
"/home/ronan/exp/test_videos/tunisia_vs_algeria_clip.mp4"
)

DET_WEIGHTS = os.getenv(
"DET_WEIGHTS",
"/home/ronan/runs/detect/train2/weights/best.pt"
)

POSE_WEIGHTS = os.getenv(
"POSE_WEIGHTS",
"/home/ronan/runs/pose/train/weights/best.pt"
)

JNR_WEIGHTS = os.getenv(
"JNR_WEIGHTS",
"/home/ronan/runs/soccernet_jersey/best_resnet18_temporal.pt"
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

CLASS = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}


# ===================== BASIC HELPERS =====================

def exist(p):
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

def track_video_with_yolo(
    video_path,
    weights_path,
    conf=0.25,
    iou=0.5,
    device=0,
    max_frames=None,
    vid_stride=1,
    imgsz=None,
    store_images=True,
    store_images_up_to=1500,
    store_interval=30,
):
    """
    Uses Ultralytics YOLO .track with:
    - vid_stride for frame skipping
    - imgsz for resizing (only passed if not None/0)
    - half precision when CUDA is available
    - stores orig_img if (n <= store_images_up_to) OR (n % store_interval == 0)
    """
    model = YOLO(weights_path)

    use_half = torch.cuda.is_available() and str(device).lower() != "cpu"
    print(f"[track] vid_stride={vid_stride}, imgsz={imgsz}, half={use_half}")

    kwargs = dict(
        source=video_path,
        tracker="bytetrack.yaml",
        conf=conf,
        iou=iou,
        stream=True,
        verbose=False,
        device=device,
        persist=True,
        vid_stride=vid_stride,
        half=use_half,
    )
    if imgsz not in (None, 0):
        kwargs["imgsz"] = imgsz

    stream = model.track(**kwargs)

    frames, n, with_ball, with_players = [], 0, 0, 0
    unique_ids = set()
    max_frames_int = int(max_frames) if (max_frames and str(max_frames).isdigit()) else None

    for res in stream:
        n += 1
        if max_frames_int and n > max_frames_int:
            break

        img = res.orig_img
        frame = {
            "orig_shape": img.shape[:2],
            "path": getattr(res, "path", None),
            "boxes": []
        }
        if store_images and (n <= store_images_up_to or n % store_interval == 0):
            frame["orig_img"] = img

        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            b = res.boxes
            ids = (b.id.cpu().numpy().astype(int).tolist() if b.id is not None else [None] * len(b))
            cls = b.cls.cpu().numpy().astype(int).tolist()
            confs = b.conf.cpu().numpy().astype(float).tolist()
            xyxy = b.xyxy.cpu().numpy().astype(float).tolist()
            for i in range(len(cls)):
                frame["boxes"].append({
                    "id": ids[i],
                    "cls": cls[i],
                    "conf": confs[i],
                    "xyxy": xyxy[i],
                })
            if any(bb["cls"] == CLASS["ball"] for bb in frame["boxes"]):
                with_ball += 1
            if any(bb["cls"] in (CLASS["player"], CLASS["goalkeeper"]) for bb in frame["boxes"]):
                with_players += 1
            for bb in frame["boxes"]:
                if bb["id"] is not None and bb["cls"] in (CLASS["player"], CLASS["goalkeeper"]):
                    unique_ids.add(bb["id"])
        frames.append(frame)

    print(f"[track] frames={n} | with ball={with_ball} | with players={with_players} | unique IDs={len(unique_ids)}")
    if n == 0:
        raise RuntimeError("No frames were read. Check the video path/codec.")
    return frames


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


def assign_teams_to_ids_from_frames(
    frames,
    sample_frames=1200,
    class_players=(CLASS["player"], CLASS["goalkeeper"])
):
    """
    Use HSV jersey colors from frames that still have orig_img.
    """
    hues_by_id = defaultdict(list)
    sv_by_id = defaultdict(list)

    take = min(sample_frames, len(frames))
    for f in frames[:take]:
        if "orig_img" not in f:
            continue
        hsv_img = cv2.cvtColor(f["orig_img"], cv2.COLOR_BGR2HSV)
        for b in f["boxes"]:
            tid, c = b["id"], b["cls"]
            if tid is None or c not in class_players:
                continue
            x1, y1, x2, y2 = map(int, b["xyxy"])
            h = y2 - y1; w = x2 - x1
            if h <= 0 or w <= 0:
                continue
            yA = y1 + int(0.20 * h); yB = y1 + int(0.65 * h)
            xA = x1 + int(0.15 * w); xB = x1 + int(0.85 * w)
            yA = max(0, yA); yB = min(hsv_img.shape[0], yB)
            xA = max(0, xA); xB = min(hsv_img.shape[1], xB)
            if yB <= yA or xB <= xA:
                continue
            patch = hsv_img[yA:yB, xA:xB]
            feats = _hsv_feat(patch)
            if feats.size == 0:
                continue
            m = feats.mean(axis=0)
            hc, hs, s_mean, v_mean = m[0], m[1], float(m[2]), float(m[3])
            hue = _hue_from_cos_sin(hc, hs)
            hues_by_id[tid].append(hue)
            sv_by_id[tid].append((s_mean, v_mean))

    if len(hues_by_id) < 2:
        print("[teams] Not enough player hues → empty map.")
        return {}, {}

    id_list = sorted(hues_by_id.keys())
    per_id_features = []
    per_id_hsv = {}

    for pid in id_list:
        h_med = float(np.median(hues_by_id[pid]))
        s_med = float(np.median([sv[0] for sv in sv_by_id[pid]])) if sv_by_id[pid] else 0.5
        v_med = float(np.median([sv[1] for sv in sv_by_id[pid]])) if sv_by_id[pid] else 0.5
        per_id_hsv[pid] = (h_med, s_med, v_med)

        # Features for clustering: cos(h), sin(h), s, v
        # Weight s and v slightly less or more depending on importance?
        # For now, equal weight (0-1 range).
        h_rad = h_med * (np.pi / 90.0) # H is 0-180 in OpenCV
        per_id_features.append([
            math.cos(h_rad),
            math.sin(h_rad),
            s_med,
            v_med
        ])

    per_id_features = np.array(per_id_features, dtype=np.float32)

    # K-Means clustering (k=2) using OpenCV
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
        hs = np.array(cluster_hsv[team_id])
        h_med = float(np.median(hs[:, 0]))
        s_med = float(np.median(hs[:, 1]))
        v_med = float(np.median(hs[:, 2]))
        team_labels[team_id] = _describe_color(h_med, s_med, v_med)

    print(f"[teams] assignment: {len(assign)} ids; labels={team_labels}")
    for tid, lab in team_labels.items():
        cnt = sum(1 for _pid, t in assign.items() if t == tid)
        print(f"  team_id={tid} ({lab}) → {cnt} player ids")
    return assign, team_labels


# ===================== JNR MODEL (ResNet18Temporal) =====================

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


class ResNet18Temporal(torch.nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()
        from torchvision import models
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        self.backbone = backbone
        self.classifier = torch.nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        f = self.backbone(x)
        logit = self.classifier(f).view(B, T, -1).mean(1)
        return logit


class SimpleJNR:
    def __init__(self, weights, device="cpu", img_size=160, gate=0.70):
        self.ok = False
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.img_size = img_size
        self.gate = float(gate)

        if not (weights and os.path.exists(weights)):
            print("[jnr] weights not found; JNR disabled.")
            return

        try:
            from torchvision import transforms
            self.tf = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMNET_MEAN, IMNET_STD),
            ])
            self.model = ResNet18Temporal(num_classes=100, pretrained=False).to(self.device).eval()
            state = torch.load(weights, map_location=self.device)
            if isinstance(state, dict):
                state = state.get("state_dict") or state.get("model") or state
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=False)
            self.ok = True
            print(f"[jnr] loaded weights: {weights}")
        except Exception as e:
            print(f"[jnr] failed to load JNR model: {e}")
            self.ok = False

    @torch.inference_mode()
    def predict_batch(self, crops_bgr):
        if not self.ok or not crops_bgr:
            return [{"number": None, "conf": 0.0} for _ in crops_bgr]
        from PIL import Image
        xs = []
        for c in crops_bgr:
            if c is None or c.size == 0:
                xs.append(None)
            else:
                rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                xs.append(self.tf(Image.fromarray(rgb)))
        idx = [i for i, x in enumerate(xs) if x is not None]
        if not idx:
            return [{"number": None, "conf": 0.0} for _ in crops_bgr]
        B = torch.stack([xs[i] for i in idx], 0).unsqueeze(1).to(self.device)
        prob = torch.softmax(self.model(B), dim=1).cpu().numpy()
        out = [{"number": None, "conf": 0.0} for _ in crops_bgr]
        for k, i in enumerate(idx):
            p = prob[k]
            j = int(np.argmax(p))
            c = float(np.max(p))
            if j > 0 and c >= self.gate:
                num = f"{j:02d}"
            else:
                num = None
            out[i] = {"number": num, "conf": c}
        return out


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
    for t, f in enumerate(frames):
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
                            shots.append({
                                "type": "shot",
                                "t": t_star,
                                "pid": pid,
                                "team": team_map.get(pid, None),
                                "xg": xg,
                                "header": bool(is_header),
                                "opponent_present": bool(under_pressure),
                                "on_target": bool(on_target),
                                "is_penalty": False,
                                "is_goal": is_goal,
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

def phase1_stats(frames, team_map, fps=25, max_owner_px=None, tackle_px=55):
    def _nearest_player_local(ball_xy, player_boxes):
        best, best_d = None, 1e9
        for p in player_boxes:
            c = bbox_center(p["xyxy"])
            d = math.hypot(ball_xy[0] - c[0], ball_xy[1] - c[1])
            if d < best_d:
                best, best_d = p, d
        return (best["id"] if best else None), best_d

    def _center_at(f, pid):
        for b in f["boxes"]:
            if b["id"] == pid:
                return bbox_center(b["xyxy"])
        return None

    events, ownership = [], []
    for f in frames:
        players = [b for b in f["boxes"]
                   if b["cls"] in (CLASS["player"], CLASS["goalkeeper"]) and b["id"] is not None]
        balls = [b for b in f["boxes"] if b["cls"] == CLASS["ball"]]
        owner = None
        if players and balls:
            x1, y1, x2, y2 = balls[0]["xyxy"]
            ball_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            pid, d = _nearest_player_local(ball_xy, players)
            if pid is not None:
                if max_owner_px is None or d <= max_owner_px:
                    owner = pid
        ownership.append(owner)

    player_stats = defaultdict(lambda: defaultdict(int))
    team_stats = defaultdict(lambda: defaultdict(int))

    # touches (frames in possession)
    for pid in ownership:
        if pid is None:
            continue
        _inc(player_stats[pid], "touch_frames")
        tid = team_map.get(pid, None)
        if tid is not None:
            _inc(team_stats[tid], "touch_frames")

    prev = None
    for t, own in enumerate(ownership):
        if own is None:
            continue
        if prev is None:
            prev = own
            continue

        if own != prev:
            e = {
                "type": "pass_attempt",
                "t": t,
                "from": prev,
                "to": own,
                "from_team": team_map.get(prev, None),
                "to_team": team_map.get(own, None),
                "completed": team_map.get(prev, None) == team_map.get(own, None),
            }
            events.append(e)

            # possible tackle
            if (e["from_team"] is not None and e["to_team"] is not None and
                    e["from_team"] != e["to_team"]):
                f = frames[min(t, len(frames) - 1)]
                c_from = _center_at(f, e["from"])
                c_to = _center_at(f, e["to"])
                if c_from and c_to:
                    d = math.hypot(c_from[0] - c_to[0], c_from[1] - c_to[1])
                    if d <= tackle_px:
                        events.append({
                            "type": "tackle",
                            "t": t,
                            "by": e["to"],
                            "on": e["from"],
                            "by_team": e["to_team"]
                        })

        prev = own

    for e in list(events):
        if e["type"] != "pass_attempt":
            continue
        if (e["from_team"] is not None and e["to_team"] is not None and
                e["from_team"] != e["to_team"] and not e["completed"]):
            events.append({
                "type": "interception",
                "t": e["t"],
                "by": e["to"],
                "from": e["from"],
                "by_team": team_map.get(e["to"], None)
            })

    for e in events:
        if e["type"] == "pass_attempt":
            _inc(player_stats[e["from"]], "passes_total")
            if e["completed"]:
                _inc(player_stats[e["from"]], "passes_completed")
            _inc(team_stats[e["from_team"]], "passes_total")
            if e["completed"]:
                _inc(team_stats[e["from_team"]], "passes_completed")

        elif e["type"] == "interception":
            _inc(player_stats[e["by"]], "interceptions_total")
            _inc(team_stats[e.get("by_team")], "interceptions_total")

        elif e["type"] == "tackle":
            _inc(player_stats[e["by"]], "tackles_total")
            _inc(team_stats[e.get("by_team")], "tackles_total")

    return {"events": events, "player_stats": player_stats,
            "team_stats": team_stats, "ownership": ownership}


def phase2_stats(frames, base, team_map, H=None, fps=25,
                 dribble_min_px=35, opp_near_px=75, max_gap=25):
    events = list(base["events"])
    player_stats = defaultdict(lambda: defaultdict(int),
                               {pid: dict(s) for pid, s in base["player_stats"].items()})
    team_stats = defaultdict(lambda: defaultdict(int),
                             {tid: dict(s) for tid, s in base["team_stats"].items()})
    h, w = frames[0]["orig_shape"]

    def in_wide(pt): return (pt[0] < 0.15 * w) or (pt[0] > 0.85 * w)

    def in_box(pt): return (0.35 * w < pt[0] < 0.65 * w) and (0.25 * h < pt[1] < 0.85 * h)

    def _center_at(f, pid):
        for b in f["boxes"]:
            if b["id"] == pid:
                return bbox_center(b["xyxy"])
        return None

    def _nearest_opponent_dist(f, pid):
        my_team = team_map.get(pid, None)
        best = 1e9
        pc = _center_at(f, pid)
        if pc is None:
            return best
        for b in f["boxes"]:
            if b["cls"] not in (CLASS["player"], CLASS["goalkeeper"]):
                continue
            if b["id"] is None or b["id"] == pid:
                continue
            if team_map.get(b["id"], None) == my_team:
                continue
            c = bbox_center(b["xyxy"])
            d = math.hypot(pc[0] - c[0], pc[1] - c[1])
            best = min(best, d)
        return best

    # crosses
    for e in base["events"]:
        if e["type"] != "pass_attempt":
            continue
        t = e["t"]; f = frames[min(t, len(frames) - 1)]

        def center(pid):
            for b in f["boxes"]:
                if b["id"] == pid:
                    return bbox_center(b["xyxy"])
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

    # dribbles
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
                        if bb["id"] == pid_:
                            return bbox_center(bb["xyxy"])
                    return None

                c0 = _center_at_local(f0, pid); c1 = _center_at_local(f1, pid)
                if c0 and c1:
                    disp = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
                    near0 = _nearest_opponent_dist(f0, pid)
                    near1 = _nearest_opponent_dist(f1, pid)
                    if disp >= dribble_min_px and (near0 <= opp_near_px or near1 <= opp_near_px) and (end - start) <= max_gap:
                        conf = max(0.1, min(1.0, disp / 120.0))
                        events.append({"type": "dribble", "t_start": start, "t_end": end, "pid": pid,
                                       "conf": round(float(conf), 3)})
                        _inc(player_stats[pid], "dribbles_total")
                        _inc(team_stats[team_map.get(pid, None)], "dribbles_total")
                        _inc(player_stats[pid], "dribbles_successful")
                        _inc(team_stats[team_map.get(pid, None)], "dribbles_successful")
                        player_stats[pid]["dribbles_conf_sum"] = player_stats[pid].get("dribbles_conf_sum", 0.0) + float(conf)
        start = t

    return {"events": events, "player_stats": player_stats,
            "team_stats": team_stats, "ownership": base.get("ownership", [])}


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
):
    if not (exist(video_path) and exist(weights_path)):
        raise FileNotFoundError("Video or weights path is wrong.")

    device = pick_device(device)
    raw_fps = get_fps(video_path)
    eff_fps = max(1, raw_fps // max(1, vid_stride))
    print(f"[video] effective FPS for stats: {eff_fps} (stride={vid_stride})")

    frames = track_video_with_yolo(
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
    )

    # pitch polygon only if we still have some images
    pitch_poly = None
    if any("orig_img" in f for f in frames):
        # Look through the entire video (max_sample_frames=None) to find pitch keypoints
        # We rely on the fact that we stored images every `store_interval` frames.
        pitch_poly = infer_pitch_polygon_from_pose(
            frames, POSE_WEIGHTS, device=device,
            conf=0.25, sample_every=15, max_sample_frames=None
        )
        if pitch_poly is not None:
            apply_pitch_polygon_to_ball(frames, pitch_poly)

    # team colors – require some frames with orig_img
    if any("orig_img" in f for f in frames):
        team_map, team_labels = assign_teams_to_ids_from_frames(frames, sample_frames=sample_frames)
    else:
        team_map, team_labels = {}, {}
        print("[teams] No frames with orig_img; team assignment disabled.")

    if not team_labels:
        team_labels = {0: "team_A", 1: "team_B"}

    # jersey numbers only if we have images + weights
    jersey_map = {}
    if any("orig_img" in f for f in frames):
        jnr_model = SimpleJNR(JNR_WEIGHTS, device=device, img_size=JNR_IMG_SIZE, gate=JNR_GATE)
        jersey_map = jersey_numbers_from_frames_jnr(frames, jnr_model)
    else:
        print("[jersey] No frames with orig_img; JNR skipped.")

    # basic stats (phase 1 & 2)
    p1 = phase1_stats(frames, team_map, fps=eff_fps, max_owner_px=150, tackle_px=55)
    p2 = phase2_stats(frames, p1, team_map, H=H, fps=eff_fps)

    # xG + shots
    shots, xg_player_map, xg_team_map = detect_shots_and_xg(
        frames, p2["ownership"], team_map, fps=eff_fps,
        speed_px_thr_frac=0.015, near_goal_frac=0.50, opp_thr_px=90
    )

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
    for pid in sorted(p2["player_stats"].keys()):
        stats = p2["player_stats"][pid]

        j = jersey_map.get(pid, {"number": "Unknown", "conf": 0.0})
        jnum = None
        if j.get("number") not in (None, "Unknown", ""):
            s = "".join(ch for ch in j.get("number") if ch.isdigit())
            s = (s.lstrip("0") or "0")[-2:] if s else None
            if s and s not in ("0", "00"):
                jnum = s
        jersey_number = jnum if jnum else "Unknown"

        team_id = team_map.get(pid, None)
        team_name = team_labels.get(team_id, "unknown") if team_id is not None else "unknown"

        passes_total = int(stats.get("passes_total", 0))
        passes_completed = int(stats.get("passes_completed", 0))
        acc_pct = 100.0 * passes_completed / passes_total if passes_total > 0 else 0.0

        tackles_total = int(stats.get("tackles_total", 0))
        interceptions_total = int(stats.get("interceptions_total", 0))
        challenges = tackles_total + interceptions_total
        challenges_won = challenges
        tackles_successful = tackles_total
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

            "time_on_ball_s": time_on_ball_s,
            "touch_frames": touch_frames,

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
            "fouls": 0,

            "goals": goals,

            "xg_foot_no_opponent": xg_foot_no_opponent,
            "xg_header_no_opponent": xg_header_no_opponent,
            "xg_foot_opponent_present": xg_foot_opponent_present,
            "xg_header_opponent_present": xg_header_opponent_present,
        }
        players_flat.append(row)

    payload = {
        "players_flat": players_flat,
        "class_map": CLASS,
        "team_labels": team_labels,
        "video_path": video_path,
        "model_weights": str(weights_path),
        "fps": int(eff_fps),
    }

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


# ===================== SCRIPT ENTRYPOINT =====================

if __name__ == "__main__":
    imgsz_arg = DET_IMG_SIZE if DET_IMG_SIZE > 0 else None

    payload = run_pipeline_to_players_json(
        video_path=SOURCE_VIDEO_PATH,
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
    )

    print("\n=== players_flat (first 15 rows) ===")
    for row in payload["players_flat"][:15]:
        print(row)
    print("\nDone.")
