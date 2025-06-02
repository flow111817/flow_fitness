import time
import cv2
import numpy as np
from . import utils
from .pose_estimation import PoseEstimator
from config import ANALYSIS_PARAMS, KEYPOINT_INDICES

class PushupAnalyzer:
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.reset()
        
    def reset(self):
        """
        重置分析状态
        """
        
        self.frame_counter = 0
        self.pushup_count = 0
        self.is_down_position = False
        self.start_time = None
        self.analysis_data = {
            'timestamps': [],
            'elbow_angles': [],
            'pushup_counts': [],
            'frames': []
        }

    def analyze_frame(self, frame):
        """
        分析单帧图像，检测姿态并计数俯卧撑
        
        输入：帧
        
        输出：添加字后的帧
        """

        # 初始化时间
        if self.start_time is None:
            self.start_time = time.time()
        current_time = time.time() - self.start_time
        self.frame_counter += 1

        # 检测关键点
        keypoints = self.pose_estimator.detect_keypoints(frame)
        
        # 获取关键点坐标
        y, x, _ = frame.shape
        left_shoulder = self._get_keypoint_coord(frame, keypoints, 'LEFT_SHOULDER')
        right_shoulder = self._get_keypoint_coord(frame, keypoints, 'RIGHT_SHOULDER')
        left_elbow = self._get_keypoint_coord(frame, keypoints, 'LEFT_ELBOW')
        right_elbow = self._get_keypoint_coord(frame, keypoints, 'RIGHT_ELBOW')
        left_wrist = self._get_keypoint_coord(frame, keypoints, 'LEFT_WRIST')
        right_wrist = self._get_keypoint_coord(frame, keypoints, 'RIGHT_WRIST')
        
        # 计算肘部角度
        left_angle = utils.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = utils.calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_angle = (left_angle + right_angle) / 2
        
        # 检测俯卧撑动作
        if avg_angle < ANALYSIS_PARAMS['down_threshold']:
            if not self.is_down_position:
                self.is_down_position = True
        elif avg_angle > ANALYSIS_PARAMS['up_threshold']:
            if self.is_down_position:
                self.pushup_count += 1
                self.is_down_position = False
        
        # 存储数据用于报告和视频
        self.analysis_data['timestamps'].append(current_time)
        self.analysis_data['elbow_angles'].append(avg_angle)
        self.analysis_data['pushup_counts'].append(self.pushup_count)
        
        # 采样帧用于视频生成
        if self.frame_counter % 8 == 0 :
            self.analysis_data['frames'].append(frame.copy())
        
        return self._annotate_frame(frame, keypoints, avg_angle, current_time)
    
    def _get_keypoint_coord(self, frame, keypoints, keypoint_name):
        """
        获取关键点坐标
        
        输入：帧，关键点，关键点名称

        输出：帧
        """

        idx = KEYPOINT_INDICES[keypoint_name]
        kp = keypoints[idx]
        y, x, _ = frame.shape
        return (kp[0] * y, kp[1] * x)
    
    def _annotate_frame(self, frame, keypoints, avg_angle, current_time):
        """
        在帧上添加注释

        输入：帧，关键点，角度，当前时间
        
        输出：帧
        """

        # 绘制关键点和骨架
        frame = self._draw_keypoints(frame, keypoints)
        
        # 显示计数
        cv2.putText(frame, f"Pushups: {self.pushup_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示当前角度
        cv2.putText(frame, f"Angle: {avg_angle:.1f}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 显示状态
        status = "DOWN" if self.is_down_position else "UP"
        cv2.putText(frame, f"Status: {status}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        # 显示时间
        cv2.putText(frame, f"Time: {current_time:.1f}s", (20, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _draw_keypoints(self, frame, keypoints):
        """
        在图像上绘制关键点和骨架
        
        输入：帧，关键点
        
        输出：帧
        """

        y, x, _ = frame.shape
        confidence_threshold = ANALYSIS_PARAMS['confidence_threshold']
        
        # 绘制关键点
        for kp in keypoints:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx * x), int(ky * y)), 5, (0, 255, 0), -1)
        
        # 绘制骨架连接
        connections = [
            (KEYPOINT_INDICES['LEFT_SHOULDER'], KEYPOINT_INDICES['RIGHT_SHOULDER']),
            (KEYPOINT_INDICES['LEFT_SHOULDER'], KEYPOINT_INDICES['LEFT_ELBOW']),
            (KEYPOINT_INDICES['LEFT_ELBOW'], KEYPOINT_INDICES['LEFT_WRIST']),
            (KEYPOINT_INDICES['RIGHT_SHOULDER'], KEYPOINT_INDICES['RIGHT_ELBOW']),
            (KEYPOINT_INDICES['RIGHT_ELBOW'], KEYPOINT_INDICES['RIGHT_WRIST']),
            (KEYPOINT_INDICES['LEFT_SHOULDER'], KEYPOINT_INDICES['LEFT_HIP']),
            (KEYPOINT_INDICES['RIGHT_SHOULDER'], KEYPOINT_INDICES['RIGHT_HIP']),
            (KEYPOINT_INDICES['LEFT_HIP'], KEYPOINT_INDICES['RIGHT_HIP']),
            (KEYPOINT_INDICES['LEFT_HIP'], KEYPOINT_INDICES['LEFT_KNEE']),
            (KEYPOINT_INDICES['LEFT_KNEE'], KEYPOINT_INDICES['LEFT_ANKLE']),
            (KEYPOINT_INDICES['RIGHT_HIP'], KEYPOINT_INDICES['RIGHT_KNEE']),
            (KEYPOINT_INDICES['RIGHT_KNEE'], KEYPOINT_INDICES['RIGHT_ANKLE'])
        ]
        
        for start, end in connections:
            start_kp = keypoints[start]
            end_kp = keypoints[end]
            
            if start_kp[2] > confidence_threshold and end_kp[2] > confidence_threshold:
                start_point = (int(start_kp[1] * x), int(start_kp[0] * y))
                end_point = (int(end_kp[1] * x), int(end_kp[0] * y))
                cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
        
        return frame
    
    