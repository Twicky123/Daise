#!/usr/bin/env python
# coding: utf-8
"""
Fixed training script for DAiSEE engagement recognition
修復後的訓練腳本
"""
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from daisee_data_preprocessing import DataPreprocessing
import datetime
import os
from tqdm import tqdm

# GPU 設定
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# =============================================================================
# 超參數設定
# =============================================================================
BATCH_SIZE = 64
ADAM_LR = 0.005
EPOCHS = 550
SHUFFLE_BUFFER = 1000  # ← 添加這個

use_pretrained = False
data_augmentation = True
pretrained_name = 'mobilenet'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = 'checkpoints/'
log_dir = 'logs/'

if use_pretrained:
    checkpoint_dir += pretrained_name
    log_dir += pretrained_name
else:
    checkpoint_dir += 'scratch'
    log_dir += 'scratch'

if data_augmentation:
    checkpoint_dir += '_aug'
    log_dir += '_aug'

train_summary_writer = tf.summary.create_file_writer(log_dir)

# =============================================================================
# 輔助函數
# =============================================================================
def create_log_dir(log_dir, checkpoint_dir):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


# =============================================================================
# ✅ 修復後的模型架構 - 添加激活函數!
# =============================================================================
def network():
    """
    修復後的模型架構
    關鍵修改: 為 Dense 層添加 activation='relu'
    """
    model = tf.keras.Sequential()
    model.add(kl.InputLayer(input_shape=(224, 224, 3), dtype=tf.uint8))
    
    # ⭐ 添加正規化層 (將 uint8 轉成 float32 並縮放到 0-1)
    model.add(kl.Rescaling(1./255))
    
    if use_pretrained:
        if pretrained_name == 'vgg':
            vgg = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
            vgg.trainable = False
            model.add(vgg)
        if pretrained_name == 'mobilenet':
            mobnet = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
            mobnet.trainable = False
            model.add(mobnet)
    else:
        # First conv block
        model.add(kl.Conv2D(filters=128, kernel_size=3, padding='same', strides=2))
        model.add(kl.ReLU())
        model.add(kl.MaxPooling2D(pool_size=(2, 2)))
        
        # Second conv block
        model.add(kl.Conv2D(filters=256, kernel_size=3, padding='same', strides=2))
        model.add(kl.ReLU())
        model.add(kl.MaxPooling2D(pool_size=(2, 2)))
        
        # Third conv block
        model.add(kl.Conv2D(filters=512, kernel_size=3, padding='same', strides=2))
        model.add(kl.ReLU())
        model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten
    model.add(kl.Flatten())
    
    # ⭐ 修復: 添加激活函數和 Dropout
    model.add(kl.Dense(1024, activation='relu'))  # ← 添加 activation='relu'
    model.add(kl.Dropout(0.5))
    
    model.add(kl.Dense(256, activation='relu'))   # ← 添加 activation='relu'
    model.add(kl.Dropout(0.3))
    
    # Output layer
    model.add(kl.Dense(4, activation='sigmoid', name='prediction'))
    
    return model


# =============================================================================
# Macro F1 metric
# =============================================================================
@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """
    Compute the macro F1-score on a batch of observations
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


# =============================================================================
# ✅ 修復後的訓練函數 - 移除多餘參數!
# =============================================================================
@tf.function
def train_step(x, y):  # ← 只接受 x, y 兩個參數
    """
    修復後的訓練步驟
    關鍵修改: 移除多餘的參數,添加 training=True
    """
    with tf.GradientTape() as tape:
        logits = model(x, training=True)  # ← 添加 training=True
        loss_value = loss_fn(y, logits)
    
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    # Track progress
    train_loss_avg.update_state(loss_value)
    train_accuracy.update_state(macro_f1(y, logits))
    
    return loss_value


# =============================================================================
# ✅ 修復後的驗證函數
# =============================================================================
@tf.function
def test_step(x, y, metric_obj):  # ← 簡化參數
    """
    修復後的驗證步驟
    """
    logits = model(x, training=False)
    metric_obj.update_state(macro_f1(y, logits))


# =============================================================================
# 主訓練流程
# =============================================================================
if __name__ == '__main__':
    preprocessing_class = DataPreprocessing()
    
    # =========================================================================
    # 載入數據集
    # =========================================================================
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80 + "\n")
    
    # Open train set
    tfrecord_path = 'tfrecords/train.tfrecords'
    train_set = tf.data.TFRecordDataset(tfrecord_path)
    train_set = train_set.map(preprocessing_class.decode)
    train_set = train_set.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    train_set = train_set.batch(BATCH_SIZE)
    train_set = train_set.prefetch(tf.data.AUTOTUNE)
    
    # Open test set
    tfrecord_path = 'tfrecords/test.tfrecords'
    test_set = tf.data.TFRecordDataset(tfrecord_path)
    test_set = test_set.map(preprocessing_class.decode)
    test_set = test_set.batch(BATCH_SIZE)
    test_set = test_set.prefetch(tf.data.AUTOTUNE)
    
    # Open val set
    tfrecord_path = 'tfrecords/val.tfrecords'
    val_set = tf.data.TFRecordDataset(tfrecord_path)
    val_set = val_set.map(preprocessing_class.decode)
    val_set = val_set.batch(BATCH_SIZE)
    val_set = val_set.prefetch(tf.data.AUTOTUNE)
    
    print("✅ Datasets loaded successfully\n")
    
    # =========================================================================
    # 創建或載入模型
    # =========================================================================
    print("="*80)
    print("MODEL SETUP")
    print("="*80 + "\n")
    
    # Create directories
    create_log_dir(log_dir, checkpoint_dir)
    
    # Check for existing checkpoints
    last_models = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')])
    
    if last_models:
        last_model_path = os.path.join(checkpoint_dir, last_models[-1])
        first_epoch = int(last_models[-1].split("_")[1]) + 1
        print(f"📂 Found existing checkpoint: {last_model_path}")
        print(f"📊 Resuming from epoch {first_epoch}")
        model = tf.keras.models.load_model(last_model_path, compile=False)
    else:
        first_epoch = 0
        print("🆕 Training from scratch")
        model = network()
    
    # Display model architecture
    print("\n" + "-"*80)
    model.summary()
    print("-"*80 + "\n")
    
    # ⭐ 驗證模型能對不同輸入產生不同輸出
    print("="*80)
    print("MODEL SANITY CHECK")
    print("="*80 + "\n")
    
    import numpy as np
    test_zeros = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    test_255s = np.ones((1, 224, 224, 3), dtype=np.uint8) * 255
    test_random = np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8)
    
    out_zeros = model(test_zeros, training=False).numpy()[0]
    out_255s = model(test_255s, training=False).numpy()[0]
    out_random = model(test_random, training=False).numpy()[0]
    
    print(f"Zeros input:  {out_zeros}")
    print(f"255s input:   {out_255s}")
    print(f"Random input: {out_random}")
    
    max_diff = max(
        np.abs(out_zeros - out_255s).max(),
        np.abs(out_zeros - out_random).max()
    )
    
    print(f"\nMax output difference: {max_diff:.6f}")
    
    if max_diff < 0.001:
        print("❌ WARNING: Model produces identical outputs!")
        print("   Check model architecture (activations)")
    else:
        print("✅ Model responds to different inputs")
    
    print("\n" + "="*80 + "\n")
    
    # =========================================================================
    # 優化器和指標
    # =========================================================================
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=ADAM_LR)
    
    train_loss_avg = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.Mean()
    
    # =========================================================================
    # 訓練循環
    # =========================================================================
    print("="*80)
    print(f"TRAINING: Epochs {first_epoch} to {EPOCHS}")
    print("="*80 + "\n")
    
    best_val_f1 = 0.0
    
    for epoch in tqdm(range(first_epoch, EPOCHS + 1), total=EPOCHS + 1 - first_epoch):
        try:
            # Reset metrics
            train_loss_avg.reset_state()
            train_accuracy.reset_state()
            val_accuracy.reset_state()
            
            # =====================================================================
            # Training loop
            # =====================================================================
            for x_batch_train, y_batch_train in train_set:
                loss_value = train_step(x_batch_train, y_batch_train)  # ← 修復後的調用
            
            # =====================================================================
            # Validation loop
            # =====================================================================
            for x_batch_val, y_batch_val in val_set:
                test_step(x_batch_val, y_batch_val, val_accuracy)  # ← 修復後的調用
            
            # =====================================================================
            # Logging
            # =====================================================================
            train_loss = train_loss_avg.result()
            train_f1 = train_accuracy.result()
            val_f1 = val_accuracy.result()
            
            # Write to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('Train Loss', train_loss, step=epoch)
                tf.summary.scalar('Train F1 Score', train_f1, step=epoch)
                tf.summary.scalar('Val F1 Score', val_f1, step=epoch)
            
            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch:4d}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Train F1:   {train_f1:.4f}")
                print(f"  Val F1:     {val_f1:.4f}")
                
                # Track best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    print(f"  🌟 New best Val F1: {best_val_f1:.4f}")
            
            # =====================================================================
            # Save checkpoint
            # =====================================================================
            if epoch % 50 == 0:
                save_path = f'{checkpoint_dir}/Epoch_{epoch}_model.h5'
                tf.keras.models.save_model(model, save_path)
                print(f"\n💾 Model saved: {save_path}")
                
                # ⭐ 驗證保存的模型
                print("   Verifying saved model...")
                test_model = tf.keras.models.load_model(save_path, compile=False)
                test_out_1 = test_model(test_zeros, training=False).numpy()[0]
                test_out_2 = test_model(test_255s, training=False).numpy()[0]
                diff = np.abs(test_out_1 - test_out_2).max()
                
                if diff < 0.001:
                    print(f"   ❌ WARNING: Saved model produces identical outputs!")
                else:
                    print(f"   ✅ Saved model works correctly (diff={diff:.6f})")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            save_path = f'{checkpoint_dir}/Epoch_{epoch}_interrupted.h5'
            tf.keras.models.save_model(model, save_path)
            print(f"💾 Model saved: {save_path}")
            break
    
    # =========================================================================
    # Final evaluation on test set
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80 + "\n")
    
    test_accuracy.reset_state()
    for x_batch_test, y_batch_test in test_set:
        test_step(x_batch_test, y_batch_test, test_accuracy)
    
    test_f1 = test_accuracy.result().numpy()
    
    print(f"Final Test F1 Score: {test_f1:.4f}")
    print(f"Best Val F1 Score:   {best_val_f1:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80 + "\n")