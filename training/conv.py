import json
import xml.etree.ElementTree as ET

# 解析XML
tree = ET.parse('annotations.xml')
root = tree.getroot()

# COCO数据结构
coco = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 1,
        "name": "person",
        "keypoints": [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ],
        "skeleton": [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
    }]
}

# 关键点名称映射 (CVAT -> COCO索引)
keypoint_mapping = {
    "NOSE": 0, "LEFT_EYE": 1, "RIGHT_EYE": 2,
    "LEFT_EAR": 3, "RIGHT_EAR": 4, "LEFT_SHOULDER": 5,
    "RIGHT_SHOULDER": 6, "LEFT_ELBOW": 7, "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9, "RIGHT_WRIST": 10, "LEFT_HIP": 11,
    "RIGHT_HIP": 12, "LEFT_KNEE": 13, "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15, "RIGHT_ANKLE": 16
}

# 处理每张图片
for img_idx, image in enumerate(root.findall('image')):
    # 添加图片信息
    img_id = int(image.get('id'))
    img_name = image.get('name')
    img_width = int(image.get('width'))
    img_height = int(image.get('height'))
    
    coco["images"].append({
        "id": img_id,
        "file_name": img_name,
        "width": img_width,
        "height": img_height
    })
    
    # 初始化关键点数组 (17个点 x 3个值)
    keypoints = [0] * 51  # 17*3=51
    
    # 提取所有关键点
    for point in image.findall('points'):
        label = point.get('label')
        x, y = map(float, point.get('points').split(','))
        visible = 2  # COCO: 2=可见, 1=遮挡, 0=未标注
        
        # 将点放入对应位置
        if label in keypoint_mapping:
            idx = keypoint_mapping[label]
            keypoints[idx*3] = x
            keypoints[idx*3 + 1] = y
            keypoints[idx*3 + 2] = visible
    
    # 计算边界框 (包含所有关键点的最小矩形)
    all_x = [keypoints[i] for i in range(0, 51, 3) if keypoints[i] > 0]
    all_y = [keypoints[i+1] for i in range(0, 51, 3) if keypoints[i] > 0]
    
    if all_x and all_y:
        x_min = min(all_x)
        y_min = min(all_y)
        x_max = max(all_x)
        y_max = max(all_y)
        width = x_max - x_min
        height = y_max - y_min
        bbox = [x_min, y_min, width, height]
        area = width * height
    else:
        bbox = [0, 0, 0, 0]
        area = 0
    
    # 添加标注信息
    coco["annotations"].append({
        "id": img_idx,
        "image_id": img_id,
        "category_id": 1,
        "keypoints": keypoints,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0
    })

# 保存为JSON文件
with open('coco_annotations.json', 'w') as f:
    json.dump(coco, f, indent=2)

print("转换完成! 已保存为 coco_annotations.json")