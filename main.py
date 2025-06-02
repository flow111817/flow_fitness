import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import find_peaks
from datetime import datetime 
import matplotlib.pyplot as plt
import os
import time

# 关键点索引 (COCO格式)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# 加载MoveNet单人体态模型
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# 初始化视频捕获
cap = cv2.VideoCapture(1)  # 使用摄像头
# cap = cv2.VideoCapture('video.mp4')  # 使用视频文件


pushup_count = 0  # 数量记录
is_down_position = False  # 姿态记录
start_time = None  # 记录开始时间

# 用于报告生成的数据
analysis_data = {
    'timestamps': [],
    'elbow_angles': [],
    'pushup_counts': [],
    'frames': []
}


def calculate_angle(a, b, c):
    """计算三点之间的角度"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def detect_keypoints(frame):
    """使用MoveNet检测人体关键点"""
    # 调整图像大小并转换为RGB
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192,192)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # 运行模型推理
    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy()[0][0]
    
    return keypoints

def draw_keypoints(frame, keypoints, confidence_threshold=0.4):
    """在图像上绘制关键点和骨架"""
    y, x, _ = frame.shape
    
    # 绘制关键点
    for i, kp in enumerate(keypoints):
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx * x), int(ky * y)), 5, (0, 255, 0), -1)
    
    # 绘制骨架连接
    connections = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE)
    ]
    
    for start, end in connections:
        start_kp = keypoints[start]
        end_kp = keypoints[end]
        
        if start_kp[2] > confidence_threshold and end_kp[2] > confidence_threshold:
            start_point = (int(start_kp[1] * x), int(start_kp[0] * y))
            end_point = (int(end_kp[1] * x), int(end_kp[0] * y))
            cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
    
    return frame

def analyze_frame(frame):
    """分析单帧图像，检测姿态并计数俯卧撑"""
    global pushup_count, is_down_position, start_time, analysis_data
    
    # 如果是第一次调用，记录开始时间
    if start_time is None:
        start_time = time.time()
    
    # 当前时间戳（秒）
    current_time = time.time() - start_time
    
    # 检测关键点
    keypoints = detect_keypoints(frame)
    
    # 获取关键点坐标
    y, x, _ = frame.shape
    left_shoulder = (keypoints[LEFT_SHOULDER][0] * y, keypoints[LEFT_SHOULDER][1] * x)
    right_shoulder = (keypoints[RIGHT_SHOULDER][0] * y, keypoints[RIGHT_SHOULDER][1] * x)
    left_elbow = (keypoints[LEFT_ELBOW][0] * y, keypoints[LEFT_ELBOW][1] * x)
    right_elbow = (keypoints[RIGHT_ELBOW][0] * y, keypoints[RIGHT_ELBOW][1] * x)
    left_wrist = (keypoints[LEFT_WRIST][0] * y, keypoints[LEFT_WRIST][1] * x)
    right_wrist = (keypoints[RIGHT_WRIST][0] * y, keypoints[RIGHT_WRIST][1] * x)
    
    # 计算肘部角度
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    avg_angle = (left_angle + right_angle) / 2
    
    # 检测俯卧撑动作
    if avg_angle < 90:  # 手臂弯曲角度小于90度
        if not is_down_position:
            is_down_position = True
    elif avg_angle > 160:  # 手臂伸直角度大于160度
        if is_down_position:
            pushup_count += 1
            is_down_position = False
    
    # 存储数据用于报告和视频
    analysis_data['timestamps'].append(current_time)
    analysis_data['elbow_angles'].append(avg_angle)
    analysis_data['pushup_counts'].append(pushup_count)
    
    # 每隔0.5秒保存一帧用于视频生成
    if len(analysis_data['frames']) == 0 or current_time - analysis_data['timestamps'][-1] >= 0.5:
        analysis_data['frames'].append(frame.copy())
    
    # 在图像上绘制结果
    frame = draw_keypoints(frame, keypoints)
    
    # 显示计数
    cv2.putText(frame, f"Pushups: {pushup_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示当前角度
    cv2.putText(frame, f"Angle: {avg_angle:.1f}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # 显示状态
    status = "DOWN" if is_down_position else "UP"
    cv2.putText(frame, f"Status: {status}", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    
    # 显示时间
    cv2.putText(frame, f"Time: {current_time:.1f}s", (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def realtime_analysis():
    """实时视频分析"""
    global analysis_data
    
    # 重置分析数据
    analysis_data = {
        'timestamps': [],
        'elbow_angles': [],
        'pushup_counts': [],
        'frames': []
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 分析当前帧
        analyzed_frame = analyze_frame(frame)
        
        # 显示结果
        cv2.imshow('Pushup Analysis', analyzed_frame)
        
        # 退出条件
        if cv2.waitKey(10) & 0xFF == ord(' '):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def generate_analysis_report():
    """生成俯卧撑分析报告（含时间线计数）"""
    if not analysis_data['timestamps']:
        print("No data to generate report")
        return
    
    # 创建保存路径
    os.makedirs('./report', exist_ok=True)
    report_name = './report/report_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 计算俯卧撑下压时刻（找角度最小的峰值）
    peaks, _ = find_peaks([-x for x in analysis_data['elbow_angles']], height=-100, distance=10)
    
    # 创建图表
    plt.figure(figsize=(14, 10))
    
    # 子图1：肘部角度变化图（使用真实时间）
    plt.subplot(2, 1, 1)
    plt.plot(analysis_data['timestamps'], analysis_data['elbow_angles'], 'b-', label='Elbow Angle')
    plt.axhline(y=90, color='r', linestyle='--', label='Down Threshold (90°)')
    plt.axhline(y=160, color='g', linestyle='--', label='Up Threshold (160°)')
    
    # 标记俯卧撑下压点
    peak_times = [analysis_data['timestamps'][i] for i in peaks]
    peak_angles = [analysis_data['elbow_angles'][i] for i in peaks]
    plt.plot(peak_times, peak_angles, 'ro', label='Pushup Down')
    
    plt.title('Elbow Angle During Pushups')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)
    
    # 子图2：俯卧撑累积计数随时间变化
    plt.subplot(2, 1, 2)
    plt.plot(analysis_data['timestamps'], analysis_data['pushup_counts'], color='skyblue', linewidth=2)
    
    # 标记每个完成的俯卧撑
    completed_pushups = []
    for i in range(1, max(analysis_data['pushup_counts']) + 1):
        # 找到计数变为i的时间点
        idx = next(j for j, count in enumerate(analysis_data['pushup_counts']) if count >= i)
        completed_pushups.append(analysis_data['timestamps'][idx])
    
    plt.plot(completed_pushups, range(1, len(completed_pushups) + 1), 'go', markersize=8, 
             label='Completed Pushups')
    
    plt.title('Cumulative Pushup Count Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cumulative Pushups')
    plt.ylim(0, max(analysis_data['pushup_counts']) + 1)
    plt.grid(True)
    plt.legend()
    
    # 添加整体统计信息
    total_time = analysis_data['timestamps'][-1]
    total_pushups = analysis_data['pushup_counts'][-1]
    pushups_per_min = (total_pushups / total_time) * 60 if total_time > 0 else 0
    
    plt.figtext(0.5, 0.01, 
                f"Total Pushups: {total_pushups} | Total Time: {total_time:.1f}s | Pushups/min: {pushups_per_min:.1f}",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
    
    # 保存和展示
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f'{report_name}.png', dpi=300)
    plt.show()
    
    print(f"Analysis complete. Total pushups: {total_pushups}")
    print(f"Report saved as {report_name}.png")

def create_animation():
    """创建关键帧动画并保存为 MP4 文件"""
    if not analysis_data['frames']:
        print("No frames to create animation")
        return
    
    # 创建保存路径
    os.makedirs('./video', exist_ok=True)
    video_name = './video/video_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 获取视频尺寸
    height, width, _ = analysis_data['frames'][0].shape
    fps = 5  # 帧率
    
    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{video_name}.mp4', fourcc, fps, (width, height))
    
    # 写入所有保存的帧
    for frame in analysis_data['frames']:
        out.write(frame)
    
    out.release()
    
    print(f"Animation saved as {video_name}.mp4")
    print(f"Video duration: {len(analysis_data['frames']) / fps:.1f} seconds")


if __name__ == "__main__":
    # 运行实时分析
    realtime_analysis()
    
    # 创建关键帧动画
    create_animation()
    
    # 生成分析报告
    generate_analysis_report()
    