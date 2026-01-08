#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def Bev(Combinedimg):
    shape_of_img = Combinedimg.shape
    
    # # Draw vertices
    # vertices = [(600,450),(750,450),(1100,700),(250,700)]
    # for v in vertices:
    #     cv2.circle(Combinedimg, v, radius=5, color=255, thickness=-1)  
    
    # plt.imshow(Combinedimg, cmap="gray")
    # plt.title("First Frame Processed")
    # plt.axis("on")
    # plt.show()
    
    # Define source and destination points
    src_pts = np.float32([[600,450],[750,450],[1100,700],[250,700]])
    dst_pts = np.float32([[200,0], [1200,0], [900,720],[200,720]])
    
    # Perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warpedd = cv2.warpPerspective(Combinedimg, M, (shape_of_img[1], shape_of_img[0]))
    
    return warpedd



    


# In[ ]:





# In[ ]:




