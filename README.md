# Fitness_optimization
### Project Introduction

Lightweight push up motion recognition based on movenet, supporting real-time counting, keyframe capture, motion visualization analysis, suitable as an auxiliary tool for fitness

### Project Features

-  ✅  Real time action recognition and counting
-  🎥  Save keyframe animation as MP4
-  📊  Visualization of counting results (with real time on the horizontal axis)
###Installation

```bash
git clone  https://github.com/yourusername/AwesomePushupCounter.git
cd AwesomePushupCounter
pip install -r requirements.txt
python main.py
```
### Instructions for use


In main. py, you can set whether to use a camera or a recorded video for recognition. In config. py, you can set parameters such as moving object, output directory, and analysis parameters


### Project Structure

```
├── data/					
	├── output/				
		├── reports/		#report output
		├── videos/			#video output
	├── sameple_video/		#sample
├── src/					
	├── code				#sorces
├── main.py					#main
└── requirements.txt		#dependences
```
### Follow up plan

- Add more fitness plans
- Can be deployed to the front-end for video recognition
- Introduce posture scoring
- Integrating language models for action guidance
- Fine tune the model to enhance accuracy

### Contribution Guide

Welcome to submit a Pull Request!