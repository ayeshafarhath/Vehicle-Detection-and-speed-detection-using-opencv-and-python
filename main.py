
import cv2
import numpy as np
from ultralytics import YOLO
import time

"""
Vehicle Detection and Speed Detection using OpenCV and YOLOv8
Author: Ayesha Farhath - AIML 2026
Detects vehicles and estimates speed from video/webcam
"""

# Load YOLOv8 model (auto-downloads on first run)
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # nano model - fast

# Vehicle classes in COCO dataset: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO ids
CLASS_NAMES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Speed estimation setup
prev_positions = {}  # track vehicle positions
PIXELS_PER_METER = 10  # calibration - adjust based on camera
FPS = 30

def estimate_speed(prev_pos, curr_pos, fps):
    """Calculate speed in km/h using pixel displacement"""
    distance_pixels = np.sqrt((curr_pos[0]-prev_pos[0])**2 + (curr_pos[1]-prev_pos[1])**2)
    distance_meters = distance_pixels / PIXELS_PER_METER
    speed_mps = distance_meters * fps  # m/s
    speed_kmh = speed_mps * 3.6
    return speed_kmh

def main():
    # Try to open video file if exists, else webcam
    video_path = "traffic.mp4"  # put your video here or leave for webcam
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video {video_path} not found, using webcam...")
        cap = cv2.VideoCapture(0)
    
    print("Starting detection... Press 'q' to quit")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Run YOLO detection every 3 frames for speed
        if frame_count % 3 == 0:
            results = model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in VEHICLE_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Center point for tracking
                        center = ((x1+x2)//2, (y1+y2)//2)
                        track_id = f"{cls_id}_{x1//50}"  # simple tracking
                        
                        speed = 0
                        if track_id in prev_positions:
                            speed = estimate_speed(prev_positions[track_id], center, FPS)
                        
                        prev_positions[track_id] = center
                        
                        # Draw box and label
                        label = f"{CLASS_NAMES[cls_id]} {speed:.1f} km/h {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        cv2.putText(frame, "Vehicle Detection | Press Q to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("Vehicle Detection and Speed Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
