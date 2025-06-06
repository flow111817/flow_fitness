import cv2
from src.pushup_analyzer import PushupAnalyzer
from src.analy_generator import ReportGenerator
from config import OUTPUT_DIRS
from src.utils import create_directories

def realtime_analysis(camera_index=0):
    """
    实时视频分析
    """
    
    # 初始化组件
    analyzer = PushupAnalyzer()
    report_generator = ReportGenerator()
    
    # 创建输出目录
    create_directories(OUTPUT_DIRS)
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 分析当前帧
            analyzed_frame = analyzer.analyze_frame(frame)
            
            # 显示结果
            cv2.imshow('Pushup Analysis', analyzed_frame)
            
            # 退出条件
            if cv2.waitKey(10) & 0xFF == ord(' '):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # 生成报告
    report_generator.generate_analysis_report(analyzer.analysis_data)
    report_generator.create_animation(analyzer.analysis_data)

if __name__ == "__main__":
    # 使用摄像头1
    realtime_analysis(camera_index=0)
    
    # 或者使用视频文件
    # realtime_analysis(camera_index='video.mp4')
    