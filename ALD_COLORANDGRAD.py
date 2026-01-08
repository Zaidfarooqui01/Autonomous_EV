#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2 
import numpy as np 



def Color_and_Grad(undistorted,RH,RL,SH,SL,HIGH_THRESHOLD, LOW_THRESHOLD):
    
    # Step 1: HSL → S channel (use index 1 for SATURATION)
    hls = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_binary = cv2.inRange(s_channel, SL, SH)
    
    # Step 2: RGB → R channel (use index 2 for RED)
    r_channel = undistorted[:,:,2]
    r_binary = cv2.inRange(r_channel, RL, RH)
    
    # Step 3: Gray + Canny
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, LOW_THRESHOLD, HIGH_THRESHOLD)

    #step 4 r_Binary and s_binary \
    sr_combined = cv2.bitwise_or(r_binary, s_binary)
    # sr_combined = cv2.bitwise_and(r_channel, s_channel)
    # sr_binary = cv2.inRange(sr_combined, 80, 200)
    combined = cv2.bitwise_or(canny, sr_combined)
    return combined; 
    
cap = cv2.VideoCapture("D:\\AEV_Data\\Highway_clip_10s.mp4") #video  path
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    BEVIMG=Color_and_Grad(frame,250,220,250,220,140,200)
    cv2.imshow("Final Combined result", BEVIMG)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()

   


# In[ ]:





# In[ ]:




