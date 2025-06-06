import os
import json
import tensorflow as tf
import numpy as np
import cv2
import tensorflow_hub as hub

# ==== 配置路径 ====
ANNOTATION_FILE = './training/coco_annotations.json'
IMAGE_DIR = './training/images'
NUM_KEYPOINTS = 17
INPUT_SIZE = 256
BATCH_SIZE = 1
EPOCHS = 50

# ==== 加载 COCO 数据 ====
def load_data(annotation_path, image_dir):
    with open(annotation_path) as f:
        coco = json.load(f)

    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    samples = []

    for ann in coco['annotations']:
        image_id = ann['image_id']
        if image_id not in id_to_filename:
            continue
        filename = id_to_filename[image_id]
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue

        keypoints = np.array(ann['keypoints']).reshape((-1, 3))
        samples.append((img_path, keypoints))

    return samples

# ==== 数据增强与预处理 ====
def preprocess(img_path, keypoints):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"无法读取图像: {img_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    
    # 保持宽高比的缩放
    scale = INPUT_SIZE / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    image = cv2.resize(image, (new_w, new_h))
    
    # 填充到正方形
    pad_h = (INPUT_SIZE - new_h) // 2
    pad_w = (INPUT_SIZE - new_w) // 2
    image = cv2.copyMakeBorder(image, 
                              pad_h, INPUT_SIZE - new_h - pad_h,
                              pad_w, INPUT_SIZE - new_w - pad_w,
                              cv2.BORDER_CONSTANT, 
                              value=(0, 0, 0))
    image = image.astype(np.int32)

    # 关键点坐标转换 (原始坐标 -> 填充后坐标 -> 归一化)
    keypoints[:, 0] = keypoints[:, 0] * scale + pad_w
    keypoints[:, 1] = keypoints[:, 1] * scale + pad_h
    keypoints[:, :2] /= INPUT_SIZE  # 归一化到 [0, 1]
    
    return image, keypoints[:, :2]

# ==== tf.data.Dataset ====
def create_dataset(samples):
    def generator():
        for img_path, kp in samples:
            try:
                img, kps = preprocess(img_path, kp.copy())
                yield img.astype(np.int32), kps.astype(np.float32)
            except Exception as e:
                print(f"处理 {img_path} 时出错: {str(e)}")

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.int32),
            tf.TensorSpec(shape=(NUM_KEYPOINTS, 2), dtype=tf.float32)
        )
    )
    return dataset.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==== 加载并构建模型 ====
def build_keypoint_model():
    hub_layer = hub.KerasLayer(
        "https://tfhub.dev/google/movenet/singlepose/thunder/4",
        signature='serving_default',
        signature_outputs_as_dict=True,
        trainable=False  # 不改变参数
    )

    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), dtype=tf.int32)
    outputs = hub_layer(inputs)['output_0']  # [batch, 1, 17, 3]
    
    # 添加一个小网络来微调关键点输出
    x = tf.keras.layers.Reshape((17, 3))(outputs[:, 0])  # [batch, 17, 3]
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(2)(x)  # 输出 [batch, 17, 2]，即 (x, y)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

# ==== 训练 ====
def train():
    samples = load_data(ANNOTATION_FILE, IMAGE_DIR)
    print(f"加载 {len(samples)} 个样本")
    
    for i, (img_path, kp) in enumerate(samples[:3]):
        print(f"样本 {i}: {img_path}, 关键点形状: {kp.shape}")
    
    dataset = create_dataset(samples)
    model = build_keypoint_model()
    model.compile(optimizer='adam', loss='mse')

    # ✅ 移动到这里
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True)
    ]

    val_size = int(0.1 * len(samples))
    train_data = dataset.skip(val_size)
    val_data = dataset.take(val_size)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save('./models/movenet_finetuned.h5')


if __name__ == '__main__':
    train()
