import os
import re
import glob
import random
import cv2
import numpy as np
import tensorflow as tf

# =========================
# User config
# =========================
VALIDATION_ROOT = "/Users/zhangyanyu/Desktop/Engagement-recognition-using-DAISEE-dataset-master/dataset/DAiSEE/DataSet/Validation"
EXAMPLE_VIDEO = "/Users/zhangyanyu/Desktop/Engagement-recognition-using-DAISEE-dataset-master/dataset/DAiSEE/DataSet/Validation/799402/7994020110/7994020110.mp4"

CHECKPOINT_DIR = "checkpoints/"
USE_PRETRAINED = True
DATA_AUGMENTATION = True
PRETRAINED_NAME = "mobilenet"

FRAME_STRIDE = 5
MAX_SAMPLED_FRAMES = None
USE_RANDOM_VIDEO = True

# 你要的輸出順序（注意：模型原本第3維通常是 Confused）
OUT_ORDER = ["bored", "engaged", "frustrated", "interest"]

# ====== B 檢查：debug 印前幾筆 ======
DBG_N = 10   # 印前 10 個取樣幀
DBG_HASH_BYTES = 4096  # 只取前 4096 bytes 做 hash（更快）


# =========================
# Helpers
# =========================
def build_checkpoint_dir():
    ckpt = CHECKPOINT_DIR
    if USE_PRETRAINED:
        ckpt = os.path.join(ckpt, PRETRAINED_NAME)
    else:
        ckpt = os.path.join(ckpt, "scratch")

    if DATA_AUGMENTATION:
        ckpt = ckpt + "_aug"

    if not ckpt.endswith("/"):
        ckpt += "/"
    return ckpt


def pick_random_video(validation_root: str) -> str:
    patterns = ["**/*.mp4", "**/*.avi", "**/*.mov", "**/*.mkv", "**/*.MP4", "**/*.AVI"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(validation_root, p), recursive=True))
    if not files:
        raise FileNotFoundError(f"No videos found under: {validation_root}")
    return random.choice(files)


def load_latest_model_from_checkpoint_dir(checkpoint_dir: str):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith((".h5", ".keras"))]
    if not files:
        raise FileNotFoundError(f"No .h5/.keras model files found in: {checkpoint_dir}")

    def get_epoch(fn: str) -> int:
        m = re.search(r"Epoch_(\d+)_model", fn)
        return int(m.group(1)) if m else -1

    files.sort(key=get_epoch)
    chosen = files[-1]
    path = os.path.join(checkpoint_dir, chosen)
    print("[INFO] Loading model:", path)
    return tf.keras.models.load_model(path, compile=False)


def load_face_cascade():
    face_cascade_path = "dataset/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(face_cascade_path)
    if cascade.empty():
        raise FileNotFoundError(f"Cannot load haarcascade: {face_cascade_path}")
    return cascade


def preprocess_frame_like_daisee(frame_bgr, face_cascade, img_size=(224, 224)):
    """
    對齊 daisee_data_preprocessing.py：
    - BGR->RGB
    - detectMultiScale(彩色)
    - faces[0] crop
    - resize
    - 不 /255（保持 0~255）
    - float32 + expand dims
    """
    if frame_bgr is None:
        return None

    frame_rgb = frame_bgr[..., ::-1]  # BGR->RGB

    faces = face_cascade.detectMultiScale(frame_rgb, 1.3, 5)
    if faces is None or len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    roi = frame_rgb[y:y + h, x:x + w]
    roi = cv2.resize(roi, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA)

    x_in = roi.astype(np.float32)          # 0~255
    x_in = np.expand_dims(x_in, axis=0)    # (1,224,224,3)
    return x_in, (int(x), int(y), int(w), int(h))

def sanity_check_model(model):
    a = np.zeros((1,224,224,3), dtype=np.float32)
    b = (np.random.rand(1,224,224,3) * 255).astype(np.float32)

    oa = model(a, training=False).numpy()[0]
    ob = model(b, training=False).numpy()[0]
    print("[SANITY] out(zeros) :", oa)
    print("[SANITY] out(random):", ob)
    print("[SANITY] max abs diff:", float(np.max(np.abs(oa - ob))))

def print_scores_in_order(scores_4):
    """
    依你要求：bored, engaged, frustrated, interest
    - bored = out[0]
    - engaged = out[1]
    - frustrated = out[3]
    - interest = out[2]  (注意：這通常其實是 Confused)
    """
    bored = float(scores_4[0])
    engaged = float(scores_4[1])
    interest = float(scores_4[2])
    frustrated = float(scores_4[3])
    print(f"{bored:.6f} {engaged:.6f} {frustrated:.6f} {interest:.6f}")


def roi_fast_hash(x_in: np.ndarray) -> int:
    """
    快速判斷 ROI 是否真的不同：
    - 用 x_in.tobytes() 的前 DBG_HASH_BYTES bytes 做 hash（避免太慢）
    """
    b = x_in.tobytes()
    return hash(b[:DBG_HASH_BYTES])


def run_inference_on_video(model, video_path: str, frame_stride: int = 5, max_sampled_frames=None):
    face_cascade = load_face_cascade()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] FPS={fps:.2f}, total_frames={total_frames}")
    print(f"[INFO] frame_stride={frame_stride} (≈ {frame_stride/fps:.3f} sec per sampled frame)")
    print(f"[INFO] Output order: {', '.join(OUT_ORDER)}")
    print("[INFO] One line per sampled frame: bored engaged frustrated interest\n")

    frame_idx = 0
    sampled = 0
    printed_dbg = 0

    prev_hash = None
    prev_out = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 每 frame_stride 幀取 1 幀
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # OpenCV 目前的「實際讀到第幾幀」（最重要的 B 檢查）
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        prep = preprocess_frame_like_daisee(frame, face_cascade, img_size=(224, 224))
        if prep is None:
            # 沒臉就跳過（不推論、不輸出）
            if printed_dbg < DBG_N:
                print(f"[DBG] pos={pos} frame_idx={frame_idx} -> NO FACE (skip)")
                printed_dbg += 1
            frame_idx += 1
            continue

        x_in, (x, y, w, h) = prep

        # ===== B 檢查：確認輸入真的在變 =====
        # 1) 原始 frame 的均值/標準差（BGR）
        frame_mean = float(frame.mean())
        frame_std = float(frame.std())

        # 2) ROI (x_in) 的均值/標準差（RGB, 0~255）
        roi_mean = float(x_in.mean())
        roi_std = float(x_in.std())

        # 3) ROI hash
        hsh = roi_fast_hash(x_in)
        same_as_prev = (prev_hash == hsh)

        # 推論
        out = model(x_in, training=False).numpy()[0].copy()

        # 4) 輸出差值（如果輸入變了但輸出不變，這裡會顯示）
        if prev_out is None:
            delta = None
        else:
            delta = float(np.max(np.abs(out - prev_out)))

        # Debug 印前 DBG_N 次取樣幀
        if printed_dbg < DBG_N:
            print(
                f"[DBG] pos={pos:6d} frame_idx={frame_idx:6d} "
                f"frame_mean={frame_mean:8.3f} frame_std={frame_std:8.3f} "
                f"face=({x},{y},{w},{h}) "
                f"roi_mean={roi_mean:8.3f} roi_std={roi_std:8.3f} "
                f"roi_hash={hsh} same_prev={same_as_prev} "
                f"{'delta=' + str(delta) if delta is not None else 'delta=None'}"
            )
            print(f"[DBG] out={out}")
            printed_dbg += 1

        # 正式輸出（你要的那行）
        print_scores_in_order(out)

        prev_hash = hsh
        prev_out = out.copy()

        sampled += 1
        frame_idx += 1

        if max_sampled_frames is not None and sampled >= max_sampled_frames:
            break

    cap.release()
    print(f"\n[INFO] Done. Printed {sampled} sampled frames.")


# =========================
# Main
# =========================
if __name__ == "__main__":
    
    checkpoint_dir = build_checkpoint_dir()
    model = load_latest_model_from_checkpoint_dir(checkpoint_dir)
    sanity_check_model(model)
    video_path = pick_random_video(VALIDATION_ROOT) if USE_RANDOM_VIDEO else EXAMPLE_VIDEO
    run_inference_on_video(
        model=model,
        video_path=video_path,
        frame_stride=FRAME_STRIDE,
        max_sampled_frames=MAX_SAMPLED_FRAMES
    )
