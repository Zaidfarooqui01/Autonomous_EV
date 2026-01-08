#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
# Force TensorFlow to use CPU only 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np


# In[3]:


print("TensorFlow version:", tf.__version__)
print("Available Devices:", tf.config.list_physical_devices())  # Changed to show all devices


# In[4]:


# Use a lighter weight model for better CPU performance
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"  # Simpler version for CPU
print("Loading optimized model for CPU...")
detector = hub.load(MODEL_URL)
print("Model loaded successfully on CPU.")

# COCO vehicle classes of interest
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car",
    4: "motorcycle", 6: "bus", 7: "truck"   
}

def detect_vehicles_single(frame):
    """Detect vehicles in a single video frame (optimized for CPU)."""
    # Reduce image size further for faster CPU processing
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    resized_img = cv2.resize(rgb_img, (300, 300))  # Reduced from 320x320 to 300x300
    input_tensor = tf.convert_to_tensor(resized_img, dtype=tf.uint8)[None, ...]

    detections = detector(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()  
    h, w, _ = frame.shape 

    results = []
    for i in range(len(scores)):
        if scores[i] >= 0.5 and class_ids[i] in [2, 3, 1, 6, 7]:
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            results.append({
                'class': COCO_LABELS.get(class_ids[i], 'unknown'),
                'score': scores[i],
                'box': (x1, y1, x2, y2)
            })
    return results

# Load the video file
video_path = r"D:\AEV_Data\Highway_clip_10s.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video.")
    exit()

frame_index = 0
print("Starting frame-by-frame CPU processing...")
print("Press 'q' to quit, 'p' to pause")

# Reduce frame processing rate for better CPU performance
process_every_n_frames = 2  # Process every 2nd frame to reduce load
frame_skip_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break
    
    frame_skip_counter += 1
    if frame_skip_counter % process_every_n_frames != 0:
        continue  # Skip processing this frame

    frame = cv2.resize(frame, (640, 480))

    detections = detect_vehicles_single(frame)

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = f"{det['class']}:{det['score']:.2f}" 
        color = (0, 255, 128) if det['class'] == 'car' else (
                (255, 0, 0) if det['class'] == 'bicycle' else ((0, 0 , 255) if det['class'] == 'truck' 
                                                               else ((155, 155, 0) if det['class'] == 'person' else ((120,120,120)) )))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add performance info to frame
    cv2.putText(frame, f"CPU Mode - Frame: {frame_index}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Vehicle Detection (CPU Mode)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Interrupted by user.")
        break
    elif key == ord('p'):
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)

    frame_index += 1
    if frame_index % 10 == 0:
        print(f"Processed frame {frame_index}")

cap.release()
cv2.destroyAllWindows()
print("Finished processing video.")

