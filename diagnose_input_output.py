import os
import re
import glob
import random
import csv

import cv2
import numpy as np
import tensorflow as tf

# =========================
# Preprocessing (保持原樣)
# =========================
def load_face_cascade():
    face_cascade_path = "dataset/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise FileNotFoundError(f"Cannot load haarcascade: {face_cascade_path}")
    return face_cascade

def resize_image(image_rgb, img_size=(224, 224)):
    return cv2.resize(image_rgb, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA)

def preprocess_frame_face_like_daisee(frame_bgr, face_cascade, img_size=(224, 224)):
    if frame_bgr is None:
        return None
    frame_rgb = frame_bgr[..., ::-1]
    faces = face_cascade.detectMultiScale(frame_rgb, 1.3, 5)
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    roi = frame_rgb[y:y + h, x:x + w]
    roi = resize_image(roi, img_size=img_size)
    x_in = roi.astype("uint8")
    x_in = np.expand_dims(x_in, axis=0)
    return x_in, (int(x), int(y), int(w), int(h))

def pick_random_video(validation_root: str):
    patterns = ["**/*.mp4", "**/*.avi", "**/*.mov", "**/*.mkv", "**/*.MP4", "**/*.AVI"]
    video_files = []
    for p in patterns:
        video_files.extend(glob.glob(os.path.join(validation_root, p), recursive=True))
    if not video_files:
        raise FileNotFoundError(f"No video files found under: {validation_root}")
    video_path = random.choice(video_files)
    video_dir = os.path.dirname(video_path)
    return video_dir, video_path


# =========================
# 🔬 詳細的輸入分析
# =========================

def analyze_input_diversity(video_path, face_cascade, num_frames=32, frame_stride=10):
    """
    詳細分析輸入的多樣性
    """
    print("\n" + "="*80)
    print("🔍 ANALYZING INPUT DIVERSITY")
    print("="*80)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    # 收集數據
    raw_frames = []
    face_positions = []
    roi_before_resize = []
    roi_after_resize = []
    final_inputs = []
    
    idx = 0
    collected = 0
    
    print(f"\nCollecting {num_frames} frames from video...")
    
    while collected < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        
        if idx % frame_stride != 0:
            idx += 1
            continue
        idx += 1
        
        # 保存原始幀
        raw_frames.append(frame.copy())
        
        # 預處理
        frame_rgb = frame[..., ::-1]
        faces = face_cascade.detectMultiScale(frame_rgb, 1.3, 5)
        
        if faces is None or len(faces) == 0:
            continue
        
        x, y, w, h = faces[0]
        face_positions.append((x, y, w, h))
        
        # 裁切 ROI (resize 前)
        roi = frame_rgb[y:y + h, x:x + w]
        roi_before_resize.append(roi.copy())
        
        # Resize
        roi_resized = resize_image(roi, img_size=(224, 224))
        roi_after_resize.append(roi_resized.copy())
        
        # 最終輸入
        x_in = roi_resized.astype("uint8")
        x_in = np.expand_dims(x_in, axis=0)
        final_inputs.append(x_in.copy())
        
        collected += 1
    
    cap.release()
    
    print(f"Collected {collected} valid frames with faces.\n")
    
    if collected < 2:
        print("❌ Not enough frames to analyze!")
        return None
    
    # =========================================================================
    # 分析 1: 原始幀的差異
    # =========================================================================
    print("\n" + "-"*80)
    print("[ANALYSIS 1] RAW FRAME DIVERSITY")
    print("-"*80)
    
    raw_frame_stats = []
    for i, frame in enumerate(raw_frames):
        stats = {
            'mean': frame.mean(),
            'std': frame.std(),
            'min': frame.min(),
            'max': frame.max()
        }
        raw_frame_stats.append(stats)
        if i < 5:
            print(f"Frame {i:2d}: mean={stats['mean']:6.2f}, std={stats['std']:5.2f}, "
                  f"range=[{stats['min']:3d}, {stats['max']:3d}]")
    
    # 計算幀間差異
    frame_means = [s['mean'] for s in raw_frame_stats]
    frame_stds = [s['std'] for s in raw_frame_stats]
    
    print(f"\nRaw frame statistics:")
    print(f"  Mean range: {min(frame_means):.2f} ~ {max(frame_means):.2f}")
    print(f"  Mean variance: {np.var(frame_means):.4f}")
    print(f"  Std range: {min(frame_stds):.2f} ~ {max(frame_stds):.2f}")
    
    if np.var(frame_means) < 1.0:
        print("  ⚠️  WARNING: Very low variance in frame means!")
        print("     Frames may be too similar (static video content).")
    
    # =========================================================================
    # 分析 2: 臉部位置的差異
    # =========================================================================
    print("\n" + "-"*80)
    print("[ANALYSIS 2] FACE POSITION DIVERSITY")
    print("-"*80)
    
    face_positions = np.array(face_positions)
    
    for i in range(min(5, len(face_positions))):
        x, y, w, h = face_positions[i]
        print(f"Frame {i:2d}: face at ({x:3d}, {y:3d}), size ({w:3d} x {h:3d})")
    
    print(f"\nFace position statistics:")
    print(f"  X position: mean={face_positions[:, 0].mean():.1f}, std={face_positions[:, 0].std():.2f}, "
          f"range=[{face_positions[:, 0].min()}, {face_positions[:, 0].max()}]")
    print(f"  Y position: mean={face_positions[:, 1].mean():.1f}, std={face_positions[:, 1].std():.2f}, "
          f"range=[{face_positions[:, 1].min()}, {face_positions[:, 1].max()}]")
    print(f"  Width:      mean={face_positions[:, 2].mean():.1f}, std={face_positions[:, 2].std():.2f}, "
          f"range=[{face_positions[:, 2].min()}, {face_positions[:, 2].max()}]")
    print(f"  Height:     mean={face_positions[:, 3].mean():.1f}, std={face_positions[:, 3].std():.2f}, "
          f"range=[{face_positions[:, 3].min()}, {face_positions[:, 3].max()}]")
    
    if face_positions[:, 0].std() < 5 and face_positions[:, 1].std() < 5:
        print("  ⚠️  WARNING: Face positions are almost identical!")
        print("     Person in video is very still.")
    
    # =========================================================================
    # 分析 3: ROI (resize 前) 的差異
    # =========================================================================
    print("\n" + "-"*80)
    print("[ANALYSIS 3] ROI DIVERSITY (before resize)")
    print("-"*80)
    
    roi_stats_before = []
    for i, roi in enumerate(roi_before_resize):
        stats = {
            'shape': roi.shape,
            'mean': roi.mean(),
            'std': roi.std(),
            'hash': hash(roi.tobytes()) % 1000000
        }
        roi_stats_before.append(stats)
        if i < 5:
            print(f"Frame {i:2d}: shape={stats['shape']}, mean={stats['mean']:6.2f}, "
                  f"std={stats['std']:5.2f}, hash={stats['hash']:6d}")
    
    # 檢查 hash 唯一性
    hashes_before = [s['hash'] for s in roi_stats_before]
    unique_hashes_before = len(set(hashes_before))
    
    print(f"\nROI (before resize) uniqueness:")
    print(f"  Total frames: {len(roi_before_resize)}")
    print(f"  Unique hashes: {unique_hashes_before}")
    
    if unique_hashes_before == 1:
        print("  ❌ CRITICAL: All ROIs are IDENTICAL before resize!")
    elif unique_hashes_before < len(roi_before_resize) * 0.5:
        print(f"  ⚠️  WARNING: Many duplicate ROIs ({unique_hashes_before}/{len(roi_before_resize)})")
    else:
        print(f"  ✅ ROIs are diverse before resize")
    
    # =========================================================================
    # 分析 4: ROI (resize 後) 的差異
    # =========================================================================
    print("\n" + "-"*80)
    print("[ANALYSIS 4] ROI DIVERSITY (after resize to 224x224)")
    print("-"*80)
    
    roi_stats_after = []
    for i, roi in enumerate(roi_after_resize):
        stats = {
            'mean': roi.mean(),
            'std': roi.std(),
            'hash': hash(roi.tobytes()) % 1000000
        }
        roi_stats_after.append(stats)
        if i < 5:
            print(f"Frame {i:2d}: mean={stats['mean']:6.2f}, std={stats['std']:5.2f}, "
                  f"hash={stats['hash']:6d}")
    
    # 檢查 hash 唯一性
    hashes_after = [s['hash'] for s in roi_stats_after]
    unique_hashes_after = len(set(hashes_after))
    
    print(f"\nROI (after resize) uniqueness:")
    print(f"  Total frames: {len(roi_after_resize)}")
    print(f"  Unique hashes: {unique_hashes_after}")
    
    if unique_hashes_after == 1:
        print("  ❌ CRITICAL: All ROIs are IDENTICAL after resize!")
        print("     Problem is in preprocessing or video content.")
    elif unique_hashes_after < len(roi_after_resize) * 0.5:
        print(f"  ⚠️  WARNING: Many duplicate ROIs after resize ({unique_hashes_after}/{len(roi_after_resize)})")
    else:
        print(f"  ✅ ROIs are diverse after resize")
    
    # 計算 ROI 之間的像素差異
    print("\nPixel-level differences between consecutive ROIs:")
    for i in range(min(5, len(roi_after_resize) - 1)):
        diff = np.abs(roi_after_resize[i].astype(float) - roi_after_resize[i+1].astype(float))
        mean_diff = diff.mean()
        max_diff = diff.max()
        print(f"  Frame {i} vs {i+1}: mean_diff={mean_diff:6.2f}, max_diff={max_diff:6.2f}")
    
    # 整體差異統計
    all_diffs = []
    for i in range(len(roi_after_resize) - 1):
        diff = np.abs(roi_after_resize[i].astype(float) - roi_after_resize[i+1].astype(float)).mean()
        all_diffs.append(diff)
    
    print(f"\nOverall consecutive differences:")
    print(f"  Mean: {np.mean(all_diffs):.2f}")
    print(f"  Std:  {np.std(all_diffs):.2f}")
    print(f"  Min:  {np.min(all_diffs):.2f}")
    print(f"  Max:  {np.max(all_diffs):.2f}")
    
    if np.mean(all_diffs) < 1.0:
        print("  ⚠️  WARNING: ROIs are extremely similar (mean diff < 1.0)")
        print("     This could explain why model outputs are identical.")
    
    # =========================================================================
    # 分析 5: 最終輸入的差異
    # =========================================================================
    print("\n" + "-"*80)
    print("[ANALYSIS 5] FINAL MODEL INPUT DIVERSITY")
    print("-"*80)
    
    for i in range(min(5, len(final_inputs))):
        x_in = final_inputs[i]
        print(f"Frame {i:2d}: dtype={x_in.dtype}, shape={x_in.shape}, "
              f"mean={x_in.mean():6.2f}, std={x_in.std():5.2f}, "
              f"range=[{x_in.min()}, {x_in.max()}]")
    
    # 檢查最終輸入是否相同
    print("\nChecking if final inputs are identical...")
    
    all_identical = True
    for i in range(1, len(final_inputs)):
        if not np.array_equal(final_inputs[0], final_inputs[i]):
            all_identical = False
            break
    
    if all_identical:
        print("  ❌ CRITICAL: All final model inputs are IDENTICAL!")
        print("     This is why model outputs are the same.")
    else:
        print("  ✅ Final model inputs are different.")
        
        # 計算輸入間的差異
        input_diffs = []
        for i in range(len(final_inputs) - 1):
            diff = np.abs(final_inputs[i].astype(float) - final_inputs[i+1].astype(float)).mean()
            input_diffs.append(diff)
        
        print(f"\n  Input differences:")
        print(f"    Mean: {np.mean(input_diffs):.4f}")
        print(f"    Std:  {np.std(input_diffs):.4f}")
        print(f"    Min:  {np.min(input_diffs):.4f}")
        print(f"    Max:  {np.max(input_diffs):.4f}")
        
        if np.mean(input_diffs) < 0.1:
            print("    ⚠️  Inputs are very similar (mean diff < 0.1)")
    
    print("\n" + "="*80)
    
    return final_inputs


def test_model_with_inputs(model, inputs):
    """
    測試模型對不同輸入的反應
    """
    print("\n" + "="*80)
    print("🧪 TESTING MODEL WITH COLLECTED INPUTS")
    print("="*80)
    
    outputs = []
    
    print("\nRunning model on all inputs...")
    for i, x_in in enumerate(inputs):
        out = model(x_in, training=False).numpy()[0].copy()
        outputs.append(out)
        
        if i < 10:
            print(f"Input {i:2d}: output = {out}")
    
    outputs = np.array(outputs)
    
    print(f"\n{'='*80}")
    print("MODEL OUTPUT ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nTotal outputs: {len(outputs)}")
    
    # 檢查所有輸出是否相同
    all_same = np.all(outputs == outputs[0])
    
    if all_same:
        print("❌ CRITICAL: All model outputs are IDENTICAL!")
        print(f"   Constant output: {outputs[0]}")
    else:
        print("✅ Model produces different outputs")
        
        # 統計分析
        print(f"\nOutput statistics (per class):")
        labels = ['Bored', 'Engaged', 'Confused', 'Frustrated']
        for i, label in enumerate(labels):
            values = outputs[:, i]
            print(f"  {label:12s}: mean={values.mean():.6f}, std={values.std():.6f}, "
                  f"range=[{values.min():.6f}, {values.max():.6f}]")
        
        # 整體變異性
        total_variance = outputs.var(axis=0).sum()
        print(f"\nTotal output variance: {total_variance:.8f}")
        
        if total_variance < 1e-6:
            print("  ⚠️  Extremely low variance - outputs are nearly identical")
        elif total_variance < 1e-4:
            print("  ⚠️  Very low variance - model may not be responding to inputs properly")
    
    # 檢查輸出間的差異
    print("\nOutput differences between consecutive frames:")
    for i in range(min(10, len(outputs) - 1)):
        diff = np.abs(outputs[i] - outputs[i+1])
        max_diff = diff.max()
        print(f"  Frame {i:2d} vs {i+1:2d}: max_diff = {max_diff:.8f}")
    
    print("\n" + "="*80)
    
    return outputs


def load_specific_model(checkpoint_dir: str, epoch: int):
    model_path = os.path.join(checkpoint_dir, f"Epoch_{epoch}_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"\nLoading model: {model_path}\n")
    return tf.keras.models.load_model(model_path, compile=False)


# =========================
# Main
# =========================
if __name__ == '__main__':
    checkpoint_dir = 'checkpoints/'
    use_pretrained = False
    data_augmentation = True
    pretrained_name = 'mobilenet'

    checkpoint_dir += (pretrained_name if use_pretrained else 'scratch')
    checkpoint_dir += '_aug/' if data_augmentation else '/'
    
    validation_root = "/Users/zhangyanyu/Downloads/Engagement-recognition-using-DAISEE-dataset-master/dataset/DAiSEE/DataSet/Validation"
    
    # 選擇一個影片
    face_cascade = load_face_cascade()
    video_dir, video_path = pick_random_video(validation_root)
    
    print("\n" + "🎬"*40)
    print(f"Selected video: {video_path}")
    print("🎬"*40)
    
    # =========================================================================
    # 步驟 1: 詳細分析輸入多樣性
    # =========================================================================
    inputs = analyze_input_diversity(
        video_path, 
        face_cascade, 
        num_frames=32, 
        frame_stride=10
    )
    
    if inputs is None or len(inputs) < 2:
        print("\n❌ Cannot proceed - not enough valid inputs!")
        exit(1)
    
    # =========================================================================
    # 步驟 2: 測試模型
    # =========================================================================
    model = load_specific_model(checkpoint_dir, epoch=450)
    outputs = test_model_with_inputs(model, inputs)
    
    # =========================================================================
    # 步驟 3: 綜合結論
    # =========================================================================
    print("\n" + "📋"*40)
    print("COMPREHENSIVE DIAGNOSIS CONCLUSION")
    print("📋"*40)
    
    # 檢查輸入多樣性
    input_hashes = [hash(inp.tobytes()) % 1000000 for inp in inputs]
    unique_inputs = len(set(input_hashes))
    
    # 檢查輸出多樣性
    output_unique = not np.all(outputs == outputs[0])
    output_variance = outputs.var(axis=0).sum()
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"  Inputs collected: {len(inputs)}")
    print(f"  Unique inputs: {unique_inputs}")
    print(f"  Outputs unique: {output_unique}")
    print(f"  Output variance: {output_variance:.8f}")
    print(f"{'='*80}\n")
    
    if unique_inputs == 1:
        print("❌ ROOT CAUSE: All inputs are IDENTICAL")
        print("\n   This is why outputs are the same.")
        print("\n   Possible reasons:")
        print("   1. Video shows person sitting completely still")
        print("   2. All frames are from the same second of video")
        print("   3. Face detection always detects the same region")
        print("\n   Solutions:")
        print("   ✓ Increase frame_stride (try 30 or 50)")
        print("   ✓ Try a different video with more movement")
        print("   ✓ Use multiple videos for testing")
        
    elif output_variance < 1e-6:
        print("❌ ROOT CAUSE: Model produces IDENTICAL outputs")
        print("\n   Inputs are different, but model doesn't respond.")
        print("\n   Possible reasons:")
        print("   1. Model weights are all zero or constant")
        print("   2. Model architecture has collapsed")
        print("   3. Training had critical bugs")
        print("\n   Solutions:")
        print("   ✓ Check model weights with deep_model_diagnosis()")
        print("   ✓ Try an earlier checkpoint (Epoch 383 or 400)")
        print("   ✓ Retrain model with fixed training script")
        
    else:
        print("✅ Inputs and outputs both show variation")
        print("\n   If CSV still shows identical values, check:")
        print("   1. Are you looking at the right CSV file?")
        print("   2. Is the CSV writing code correct?")
        print("   3. Try running the diagnosis again")
    
    print("\n" + "="*80 + "\n")

"""
```

---

## 🎯 這個診斷會告訴你:

### **1. 原始影片幀是否有差異**
```
Frame 0: mean=125.32, std=42.15
Frame 1: mean=125.34, std=42.18
Frame 2: mean=125.33, std=42.16
Mean variance: 0.0012  ← 如果很小,影片太靜態
```

### **2. 臉部位置是否相同**
```
Frame 0: face at (245, 120), size (180 x 180)
Frame 1: face at (245, 120), size (180 x 180)
X std: 0.5  ← 如果 < 5,臉幾乎不動
```

### **3. ROI 是否真的不同**
```
Unique hashes: 1/32  ← 如果是 1,所有 ROI 相同!
```

### **4. 模型輸入是否相同**
```
❌ CRITICAL: All final model inputs are IDENTICAL!
```

### **5. 模型輸出差異**
```
Total output variance: 0.000000  ← 如果是 0,模型有問題
"""