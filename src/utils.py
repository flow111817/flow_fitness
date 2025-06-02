import os
import numpy as np
from datetime import datetime


def calculate_angle(a, b, c):
    """
    计算三点之间的角度
    
    输入：边点，角点，边点
    
    输出：角度值
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def create_directories(dirs):
    """
    创建必要的输出目录
    
    输入：输出的字典，格式：文件种类:输出路径
    """

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

def get_timestamp():
    """
    获取当前时间戳字符串
    
    返回当前时间：年月日_时分秒
    """
    
    return datetime.now().strftime('%Y%m%d_%H%M%S')