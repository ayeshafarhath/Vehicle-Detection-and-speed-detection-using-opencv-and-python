# Vehicle Detection and Speed Detection using OpenCV and Python

A simple computer vision project to detect vehicles and estimate their speed and distance using OpenCV Haar Cascade.

### How it works
- Uses `cars.xml` Haar Cascade to detect cars from video
- Calculates distance from camera based on bounding box width
- Estimates speed using time difference between two reference lines

### Tech Stack
- Python
- OpenCV (cv2)
- NumPy

### Files in this repo
- `speedDetect.py` - Main code for vehicle detection and speed estimation
- `cars.xml` - Pre-trained Haar Cascade model for car detection
- `requirements.txt` - Dependencies

### Installation
```bash
pip install -r requirements.txt
```
### Usage
1. Add your own traffic video as `car.mp4` in the same folder
2. Run:
```bash
python speedDetect.py
```
