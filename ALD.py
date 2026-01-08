#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np


#you can download features files from jupyter notebook as script
# === Import your custom modules ===
from ALD_CAMERA_CAL import get_new_camera_matrix
from ALD_COLORANDGRAD import Color_and_Grad
from ALD_BEV import Bev
from ALD_HISTORGRAM import fit_polynomial


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # ---- Parameters ----
        self.declare_parameter('video_file', r"D:\AEV_Data\Clip_27s.mp4")
        video_file = self.get_parameter('video_file').get_parameter_value().string_value

        # ---- Open video ----
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
                self.get_logger().error(f"Cannot open video file: {video_file}")
                rclpy.shutdown()
                exit(1)

        # ---- Camera Calibration ----
        # self.New_camera_matrix, self.mtx, self.dist = get_new_camera_matrix()

        # ---- ROS Publishers ----
        self.bridge = CvBridge()
        self.pub_img = self.create_publisher(Image, 'lane_detected_image', 10)
        self.pub_info = self.create_publisher(String, 'lane_status', 10)

        # ---- Timer for processing frames ----
        self.timer = self.create_timer(0.05, self.process_frame)  # 20 Hz approx

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video or cannot fetch frame.")
            self.cap.release()
            rclpy.shutdown()  
            return

        # Step 1: Undistort
        undistorted = cv2.undistort(frame, self.mtx, self.dist, None, self.New_camera_matrix)

        # Step 2: Color & Gradient
        combined_img = Color_and_Grad(undistorted, 250, 220, 250, 220, 140, 200)

        # Step 3: Bird’s Eye View
        bev_img = Bev(combined_img)

        # Step 4: Normalize
        normalized_img = bev_img / 255.0

        # Step 5: Histogram + Polynomial Fit
        fit_img = fit_polynomial(normalized_img)

        # ---- Format for ROS publishing ----
        if len(fit_img.shape) == 2 or fit_img.shape[2] == 1:
            fit_img_bgr = cv2.cvtColor(fit_img, cv2.COLOR_GRAY2BGR)
        else:
            fit_img_bgr = fit_img

        # Ensure uint8
        if fit_img_bgr.dtype != np.uint8:
            fit_img_bgr = np.uint8(fit_img_bgr * 255) if fit_img_bgr.max() <= 1.0 else np.uint8(fit_img_bgr)

        # ---- Publish Image ----
        msg = self.bridge.cv2_to_imgmsg(fit_img_bgr, encoding='bgr8')
        self.pub_img.publish(msg)

        # ---- Publish Lane Info (for now just placeholder) ----
        self.pub_info.publish(String(data="Lane detected"))

        # ---- (Optional) Debug preview ----
        cv2.imshow("Lane Detection", fit_img_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap.isOpened():
            node.cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
