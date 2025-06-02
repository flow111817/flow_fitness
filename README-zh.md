# Fitness_optimization

### **项目简介**

基于movenet的轻量俯卧撑动作识别，支持实时计数，关键帧捕捉，动作可视化分析，适合健身的辅助工具



### **项目特点**

- ✅ 实时动作识别与计数
- 🎥 关键帧动画保存为 MP4
- 📊 计数结果可视化（横轴为真实时间）



### **安装**

```bash
git clone https://github.com/yourusername/AwesomePushupCounter.git

cd AwesomePushupCounter

pip install -r requirements.txt

python main.py
```



### **使用方法**

main.py里面可以设置使用摄像头还是一段录像进行识别，config.py设置参数包括movenet-singlepose的参数，输出目录，分析参数



### **项目结构**

```
├── data/					
	├── output/				#输出文件
		├── reports/		#图表报告
		├── videos/			#视频输出
	├── sameple_video/		#示例视频输入	
├── src/					
	├── code				#源码模块
├── main.py					#主程序
└── requirements.txt		依赖部分
```



### **后续计划**

- 添加更多健身计划
- 可部署至前端进行录像识别
- 引入姿态评分
- 接入语言模型进行动作指导
- 对模型进行微调加强准确率



### **贡献指南**

欢迎提交 Pull Request！
