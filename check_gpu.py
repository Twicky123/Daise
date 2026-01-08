import tensorflow as tf

# 列出所有可用的 GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"🎉 成功！偵測到 GPU: {gpus}")
    print("詳細資訊:", tf.config.experimental.get_device_details(gpus[0]))
else:
    print("❌ 失敗... 沒看到 GPU，只有 CPU。")