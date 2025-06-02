import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# 关键点历史记录
elbow_angles = []
shoulder_heights = []
hip_heights = []
frames = []
pushup_count = 0
is_down_position = False

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
    global pushup_count, is_down_position
    
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
    
    # 存储数据用于可视化
    elbow_angles.append(avg_angle)
    frames.append(frame.copy())
    
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
    
    return frame

def realtime_analysis():
    """实时视频分析"""
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
    """生成俯卧撑分析报告"""
    
    if not elbow_angles:
        print("No data to generate report")
        return
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 肘部角度变化图
    plt.subplot(2, 1, 1)
    plt.plot(elbow_angles, 'b-', label='Elbow Angle')
    plt.axhline(y=90, color='r', linestyle='--', label='Down Threshold (90°)')
    plt.axhline(y=160, color='g', linestyle='--', label='Up Threshold (160°)')
    
    # 标记俯卧撑动作
    peaks, _ = find_peaks([-x for x in elbow_angles], height=-100, distance=10)
    plt.plot(peaks, [elbow_angles[i] for i in peaks], 'ro', label='Pushup Down')
    
    plt.title('Elbow Angle During Pushups')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)
    
    # 俯卧撑计数展示
    plt.subplot(2, 1, 2)
    plt.bar(['Pushups'], [pushup_count], color='skyblue')
    plt.title(f'Total Pushups: {pushup_count}')
    plt.ylabel('Count')
    plt.ylim(0, max(pushup_count + 2, 10))
    
    plt.tight_layout()
    plt.savefig('pushup_analysis_report.png')
    plt.show()
    
    print(f"Analysis complete. Total pushups: {pushup_count}")

def create_animation():
    """创建关键帧动画"""
    if not frames:
        print("No frames to create animation")
        return
    
    fig, ax = plt.subplots()
    im = ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    ax.axis('off')
    plt.title('Pushup Motion Analysis')
    
    def update(frame):
        im.set_array(cv2.cvtColor(frames[frame], cv2.COLOR_BGR2RGB))
        return [im]
    
    # 每5帧取一帧制作动画
    step = max(1, len(frames) // 50)
    ani = FuncAnimation(fig, update, frames=range(0, len(frames), step), 
                       interval=100, blit=True)
    
    # 保存动画
    ani.save('pushup_animation.gif', writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    # 运行实时分析
    realtime_analysis()
    
    # 生成分析报告
    generate_analysis_report()
    
    # 创建关键帧动画
    create_animation()

