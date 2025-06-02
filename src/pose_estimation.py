import tensorflow as tf
import tensorflow_hub as hub
from . import utils
from config import MODEL_URL, KEYPOINT_INDICES

class PoseEstimator:
    def __init__(self):
        self.model = hub.load(MODEL_URL)
        self.movenet = self.model.signatures['serving_default']
        self.keypoint_indices = KEYPOINT_INDICES
        
    def detect_keypoints(self, frame):
        """
        使用MoveNet检测人体关键点

        输入：帧

        输出：关键点
        """

        # 调整图像大小并转换为RGB
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
        input_img = tf.cast(img, dtype=tf.int32)
        
        # 运行模型推理
        outputs = self.movenet(input_img)
        keypoints = outputs['output_0'].numpy()[0][0]
        
        return keypoints
    
    def get_keypoint_coordinates(self, frame, keypoints, keypoint_name):
        """
        获取特定关键点的坐标
        
        输入：帧，关键点，关键点名称
        
        输出：关键点坐标
        """
        
        idx = self.keypoint_indices[keypoint_name]
        kp = keypoints[idx]
        y, x, _ = frame.shape
        return (kp[0] * y, kp[1] * x)