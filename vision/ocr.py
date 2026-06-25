import cv2
import numpy as np
import os
from collections import defaultdict, Counter
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from itertools import groupby

# Load Config
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# ===================== CUSTOM JNR (KERAS) =====================

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
    squeezed = layers.Reshape((-1, 64))(conv_7)

    blstm_1 = layers.Bidirectional(layers.GRU(128, return_sequences=True , dropout=0.3))(squeezed)
    blstm_2 = layers.Bidirectional(layers.GRU(128, return_sequences=True , dropout=0.3))(blstm_1)

    softmax_output = layers.Dense(len(char_list) + 1, activation = 'softmax', name="dense")(blstm_2)

    output = CTCLayer(name="ctc_loss")(labels, softmax_output)
    model = keras.models.Model(inputs=[inputs, labels], outputs=output)
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
            full_model = get_model()
            full_model.load_weights(weights_path)
            img_input = full_model.inputs[0]
            dense_output = full_model.get_layer("dense").output
            self.model = keras.models.Model(inputs=img_input, outputs=dense_output)
            self.ok = True
            print(f"[jnr] Custom Keras model loaded from {weights_path}")
        except Exception as e:
            print(f"[jnr] Failed to load Custom Keras model: {e}")

    def preprocess(self, img):
        if img is None or img.size == 0: return None
        img = cv2.resize(img, (64, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=-1)
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
                
        if not batch_imgs: return results
            
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

class EasyOCRJNR:
    def __init__(self, gate=0.5):
        self.ok = False
        self.gate = float(gate)
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.ok = True
            print("[jnr] EasyOCR initialized.")
        except Exception as e:
            print(f"[jnr] Failed to initialize EasyOCR: {e}")

    def preprocess(self, img):
        if img is None or img.size == 0: return img
        h, w = img.shape[:2]
        scale = 4.0 
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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
                res = self.reader.readtext(img, allowlist='0123456789')
            except:
                res = []
            best_number = None
            best_conf = 0.0
            for _, text, conf in res:
                if conf > best_conf:
                    best_conf = conf
                    best_number = text
            if best_number is not None and best_conf < self.gate:
                best_number = None
            results.append({"number": best_number, "conf": float(best_conf)})
        return results

class EnsembleJNR:
    def __init__(self, weights_path=CONFIG["env"]["JNR_WEIGHTS"], easy_gate=0.5):
        self.custom = CustomJNR(weights_path)
        self.easy = EasyOCRJNR(gate=easy_gate)
        self.ok = self.custom.ok or self.easy.ok
        
    def predict_batch(self, crops_bgr):
        if not self.ok:
             return [{"number": None, "conf": 0.0} for _ in crops_bgr]
             
        # Prefer Custom Model
        if self.custom.ok:
            return self.custom.predict_batch(crops_bgr)
        
        # Fallback to EasyOCR
        return self.easy.predict_batch(crops_bgr)

# ===================== JERSEY ANCHOR LOGIC =====================

def run_jnr_and_anchor(frames, jnr_model, top_k=10, min_box_h=30):
    """
    Collects crops, runs JNR, and anchors identities.
    """
    if jnr_model is None or not jnr_model.ok:
        print("[jersey] JNR disabled.")
        return {}

    # 1. Collect Crops per ID
    buckets = defaultdict(list)
    for f in frames:
        if "crops" in f and f["crops"]:
            for c in f["crops"]:
                box_idx = c["box_idx"]
                if box_idx >= len(f["boxes"]): continue
                b = f["boxes"][box_idx]
                if b["id"] is None: continue
                
                x1, y1, x2, y2 = map(int, b["xyxy"])
                box_h = (y2 - y1)
                if box_h < min_box_h: continue
                
                crop = c["img"]
                score = float(box_h)
                buckets[b["id"]].append((score, crop))

    # 2. Filter Top K
    for pid in buckets:
        buckets[pid].sort(key=lambda z: z[0], reverse=True)
        buckets[pid] = [p for _, p in buckets[pid][:top_k]]

    # 3. Predict & Vote (Anchor)
    jersey_map = {}
    for pid, crops in buckets.items():
        # Optimization: In a real streaming scenario, we would check if pid is already anchored.
        # Here we process the whole video batch, so we just compute it.
        
        preds = jnr_model.predict_batch(crops)
        nums = [pr["number"] for pr in preds if pr["number"] is not None]
        
        if not nums:
            continue
        
        cnt = Counter(nums)
        best_num, votes = cnt.most_common(1)[0]
        
        # Confidence check?
        # If votes < 3 and len(nums) > 5? Maybe.
        # For now, just take the winner.
        
        best_conf = max(pr["conf"] for pr in preds if pr["number"] == best_num)
        
        jersey_map[pid] = {
            "number": best_num,
            "conf": round(float(best_conf), 3),
            "votes": int(votes),
            "samples": int(len(nums)),
        }
        
    print(f"[jersey] Anchored {len(jersey_map)} IDs.")
    return jersey_map
