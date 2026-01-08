#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')


def process_frame(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel_size = 7
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


    
    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    mask = np.zeros_like(edges)
    mask_color = 255

    imshape = image.shape
    vertices = np.array([
        [ (0, imshape[0]) , (200,int(imshape[0] / 1.7)) ,
         (int(imshape[1]-200), int(imshape[0] / 1.7))  ,  (imshape[1], imshape[0])] ], dtype=np.int32)

    mask=cv2.fillPoly(mask, vertices, mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
   

    rho = 5
    theta = np.pi / 180
    threshold = 100
    min_line_length = 80
    max_line_gap = 200
    line_image = np.zeros_like(image)

    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    COLOR=[45,100,255]
    if lines is not None:
        for line in lines:
            
            for x1, y1, x2, y2 in line:
                # Calculate angle in degrees
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if 30<= angle <= 40:
                    cv2.line(line_image, (x1, y1), (x2, y2), COLOR, 2)

    # Overlay lines on the original image
    combo = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return combo

# Open the video file


# In[2]:


cap = cv2.VideoCapture("D:\AEV_Data\Highway_clip_10s.mp4")
out = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("video breaked: or ended:")
        break
 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = process_frame(frame_rgb)
     
    
    processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID', 'MJPG', etc.
    
    
    if out is None:
        height, width = processed_frame_bgr.shape[:2]
        out = cv2.VideoWriter('new_video.mp4', fourcc, 25.0, (width, height))

    
    out.write(processed_frame_bgr)
    fps = 60.0  # Target FPS just a float 
    wait_time = int(1000/ fps)   
    cv2.imshow('Lane Detection', processed_frame_bgr)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




