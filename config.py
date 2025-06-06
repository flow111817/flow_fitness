# 关键点索引 (COCO格式)
KEYPOINT_INDICES = {
    'NOSE': 0,
    'LEFT_EYE': 1,
    'RIGHT_EYE': 2,
    'LEFT_EAR': 3,
    'RIGHT_EAR': 4,
    'LEFT_SHOULDER': 5,
    'RIGHT_SHOULDER': 6,
    'LEFT_ELBOW': 7,
    'RIGHT_ELBOW': 8,
    'LEFT_WRIST': 9,
    'RIGHT_WRIST': 10,
    'LEFT_HIP': 11,
    'RIGHT_HIP': 12,
    'LEFT_KNEE': 13,
    'RIGHT_KNEE': 14,
    'LEFT_ANKLE': 15,
    'RIGHT_ANKLE': 16
}

# 分析参数
ANALYSIS_PARAMS = {
    'down_threshold': 90,    # 下压角度阈值
    'up_threshold': 160,     # 上举角度阈值
    'confidence_threshold': 0.4,  # 关键点置信度阈值
    'frame_sampling_interval': 0.5  # 采样间隔(秒)
}

# 模型配置
MODEL_URL = "./models/movenet_finetuned.h5"

# 输出路径
OUTPUT_DIRS = {
    'reports': './data/output/reports',
    'videos': './data/output/videos'
}
