import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from config import OUTPUT_DIRS
from . import utils
import cv2


class ReportGenerator:
    def __init__(self):
        self.output_dirs = OUTPUT_DIRS
        utils.create_directories(self.output_dirs)
    
    def generate_analysis_report(self, analysis_data):
        """
        生成俯卧撑分析报告
        
        输入：分析数据-->字典类型
        
        输出：报告文件路径
        """

        if not analysis_data['timestamps']:
            print("No data to generate report")
            return
        
        report_name = f"report_{utils.get_timestamp()}"
        report_path = os.path.join(self.output_dirs['reports'], report_name)
        
        # 计算俯卧撑下压时刻
        peaks, _ = find_peaks([-x for x in analysis_data['elbow_angles']], height=-100, distance=10)
        
        # 创建图表
        plt.figure(figsize=(14, 10))
        
        # 肘部角度变化图
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
        
        # 俯卧撑累积计数
        plt.subplot(2, 1, 2)
        plt.plot(analysis_data['timestamps'], analysis_data['pushup_counts'], color='skyblue', linewidth=2)
        
        # 标记每个完成的俯卧撑
        completed_pushups = []
        for i in range(1, max(analysis_data['pushup_counts']) + 1):
            try:
                idx = next(j for j, count in enumerate(analysis_data['pushup_counts']) if count >= i)
                completed_pushups.append(analysis_data['timestamps'][idx])
            except StopIteration:
                break
        
        plt.plot(completed_pushups, range(1, len(completed_pushups) + 1), 'go', markersize=8, 
                 label='Completed Pushups')
        
        plt.title('Cumulative Pushup Count Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Cumulative Pushups')
        plt.ylim(0, max(analysis_data['pushup_counts']) + 1)
        plt.grid(True)
        plt.legend()
        
        # 添加统计信息
        total_time = analysis_data['timestamps'][-1]
        total_pushups = analysis_data['pushup_counts'][-1]
        pushups_per_min = (total_pushups / total_time) * 60 if total_time > 0 else 0
        
        plt.figtext(0.5, 0.01, 
                    f"Total Pushups: {total_pushups} | Total Time: {total_time:.1f}s | Pushups/min: {pushups_per_min:.1f}",
                    ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
        
        # 保存和展示
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(f'{report_path}.png', dpi=300)
        
        print(f"Analysis complete. Total pushups: {total_pushups}")
        print(f"Report saved as {report_path}.png")
        
        return report_path
    
    def create_animation(self, analysis_data):
        """
        创建关键帧动画
        
        输入：分析数据-->字典类型
        
        输出：视频文件路径
        """

        if not analysis_data['frames']:
            print("No frames to create animation")
            return None
        
        video_name = f"video_{utils.get_timestamp()}"
        video_path = os.path.join(self.output_dirs['videos'], f"{video_name}.mp4")
        
        # 获取视频尺寸
        height, width, _ = analysis_data['frames'][0].shape
        fps = 20  # 帧率
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # 写入所有保存的帧
        for frame in analysis_data['frames']:
            out.write(frame)
        
        out.release()
        
        print(f"Animation saved as {video_path}")
        print(f"Video duration: {len(analysis_data['frames']) / fps:.1f} seconds")
        
        return video_path