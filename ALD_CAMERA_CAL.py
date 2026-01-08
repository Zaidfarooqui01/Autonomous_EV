#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np
import glob

def get_new_camera_matrix(calib_images_path="D:\\AEV_Data\\chessboard_images\\calibration*.jpg",
                          board_size=(9,6)):
    """
    Calibrate the camera using chessboard images.
    Returns newcameramtx, mtx, dist
    """
    # Prepare object points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    realpoints = []  # 3D points
    imagepoints = [] # 2D points

    images = glob.glob(calib_images_path)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            realpoints.append(objp)
            imagepoints.append(corners)

    if realpoints and imagepoints:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            realpoints, imagepoints, gray.shape[::-1], None, None
        )

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        return newcameramtx, mtx, dist
    else:
        raise ValueError("No chessboard corners found in images. Check path or board size.")



# In[ ]:




