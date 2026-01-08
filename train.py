#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from daisee_data_preprocessing import DataPreprocessing
import datetime
import os
from tqdm import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 64
LR = 0.005
EPOCHS = 500
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

# The statements in this function throws unknows errors
# def create_log_dir(log_dir, checkpoint_dir):
#     if not os.path.exists(log_dir):
#         os.mkdir(log_dir)
#     if not os.path.exists(checkpoint_dir):
#         os.mkdir(checkpoint_dir)

def create_log_dir(log_dir, checkpoint_dir):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


def network():
    model = tf.keras.Sequential()
    model.add(kl.InputLayer(input_shape=(224, 224, 3)))
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
        model.add(tf.keras.layers.ReLU())
        model.add(kl.MaxPooling2D(pool_size=(2, 2)))
        # Second conv block
        model.add(kl.Conv2D(filters=256, kernel_size=3, padding='same', strides=2))
        model.add(tf.keras.layers.ReLU())
        model.add(kl.MaxPooling2D(pool_size=(2, 2)))
        # Third conv block
        model.add(kl.Conv2D(filters=512, kernel_size=3, padding='same', strides=2))
        model.add(tf.keras.layers.ReLU())
        model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    # Flatten
    model.add(kl.Flatten())
    # First FC
    model.add(kl.Dense(1024))
    # Second Fc
    model.add(kl.Dense(256))
    # Output FC with sigmoid at the end
    model.add(kl.Dense(4, activation='sigmoid', name='prediction'))
    return model
'''
https://keras.io/guides/writing_a_training_loop_from_scratch/
Compile into a static graph any function that take tensors as input to apply global performance optimizations.
'''

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Track progress
    train_loss_avg.update_state(loss_value)
    train_accuracy.update_state(macro_f1(y, logits))
    #acc_update(y, logits)
    return loss_value

@tf.function
def test_step(model, x, y, set_name):
    logits = model(x)
    if set_name == 'val':
        val_accuracy.update_state(y, logits)
    else:
        test_accuracy.update_state(y, logits)


if __name__ == '__main__':
    preprocessing_class = DataPreprocessing()

    # Open train set
    tfrecord_path = 'tfrecords/train.tfrecords'
    train_set = tf.data.TFRecordDataset(tfrecord_path)
    # Parse the record into tensors with map.
    train_set = train_set.map(preprocessing_class.decode)
    train_set = train_set.shuffle(1)
    train_set = train_set.batch(BATCH_SIZE)

    # Open test set
    tfrecord_path = 'tfrecords/test.tfrecords'
    test_set = tf.data.TFRecordDataset(tfrecord_path)
    # Parse the record into tensors with map.
    test_set = test_set.map(preprocessing_class.decode)
    test_set = test_set.shuffle(1)
    test_set = test_set.batch(BATCH_SIZE)

    # Open val set
    tfrecord_path = 'tfrecords/val.tfrecords'
    val_set = tf.data.TFRecordDataset(tfrecord_path)
    # Parse the record into tensors with map.
    val_set = val_set.map(preprocessing_class.decode)
    val_set = val_set.shuffle(1)
    val_set = val_set.batch(BATCH_SIZE)

    # Create the model
    model = network()

    # Optimizers and metrics
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    train_loss_avg = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.Mean()

    create_log_dir(log_dir, checkpoint_dir)

    last_models = sorted(os.listdir(checkpoint_dir))
    if last_models:
        last_model_path = checkpoint_dir + '/' + last_models[-1]
        first_epoch = int(last_models[-1].split("_")[1]) + 1
        print("First epoch is ", first_epoch)
        model = tf.keras.models.load_model(last_model_path)
    else:
        first_epoch = 0
        model = network()

    # Train
    epoch_pbar = tqdm(range(first_epoch, EPOCHS+1), total=EPOCHS+1-first_epoch, desc='Training Progress')
    for epoch in epoch_pbar:
        try:
            # 為每個 batch 創建進度條
            train_batches = tqdm(train_set, desc=f'Epoch {epoch} - Training', leave=False)
            
            # Training loop
            for x_batch_train, y_batch_train in train_batches:
                # Do step
                loss_value = train_step(x_batch_train, y_batch_train)
                
                # 即時更新進度條顯示當前 batch 的 loss 和 F1
                train_batches.set_postfix({
                    'loss': f'{train_loss_avg.result().numpy():.4f}',
                    'f1': f'{train_accuracy.result().numpy():.4f}'
                })

            # Test on validation set
            val_batches = tqdm(val_set, desc=f'Epoch {epoch} - Validation', leave=False)
            for x_batch_val, y_batch_val in val_batches:
                test_step(model, x_batch_val, y_batch_val, 'val')

            # 更新 epoch 級別的進度條，顯示完整的訓練和驗證指標
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss_avg.result().numpy():.4f}',
                'train_f1': f'{train_accuracy.result().numpy():.4f}',
                'val_f1': f'{val_accuracy.result().numpy():.4f}'
            })

            # Write in the summary
            with train_summary_writer.as_default():
                tf.summary.scalar('Train Loss', train_loss_avg.result(), step=epoch)
                tf.summary.scalar('Train F1 Score', train_accuracy.result(), step=epoch)
                tf.summary.scalar('Val F1 Score', val_accuracy.result(), step=epoch)

            # Reset training metrics at the end of each epoch
            train_accuracy.reset_state()
            val_accuracy.reset_state()

            if epoch % 50 == 0:
                tf.keras.models.save_model(model, '{}/Epoch_{}_model.h5'.format(checkpoint_dir, str(epoch)),
                                           save_format="h5")

        except KeyboardInterrupt:
            print("\nKeyboard Interruption...")
            # Save model
            tf.keras.models.save_model(model, '{}/Epoch_{}_model.hp5'.format(checkpoint_dir, str(epoch)),
                                       save_format="h5")
            break

    # Test on validation set
    test_batches = tqdm(test_set, desc='Testing')
    for x_batch_test, y_batch_test in test_batches:
        test_step(model, x_batch_test, y_batch_test, 'test')
    test_set_acc = test_accuracy.result().numpy()
    print("Accuracy on test set is", test_set_acc)
"""
```

## 主要修改說明：

1. **Epoch 級別的進度條**：使用 `epoch_pbar` 顯示整體訓練進度
2. **Batch 級別的進度條**：
   - 訓練時顯示即時的 loss 和 F1 分數
   - 驗證時顯示進度
3. **即時更新**：在每個 batch 完成後立即更新顯示
4. **顯示格式**：使用 `set_postfix()` 在進度條右側顯示指標

訓練時你會看到類似這樣的輸出：
```
Training Progress: 45%|████▌     | 225/500 [2:15:30<2:45:20, train_loss=0.3245, train_f1=0.6789, val_f1=0.6543]
"""