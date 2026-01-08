## 🧠 Project Overview

This project demonstrates the **core perception stack of an Autonomous Electric Vehicle (AEV)** using **ROS 2**. The system is developed on **Ubuntu installed on Windows** and focuses on real-time computer vision pipelines commonly used in self-driving vehicles.

### The implementation includes:
- Lane detection using classical computer vision  
- Bird’s Eye View (BEV) transformation  
- Vehicle detection using deep learning  
- Traffic sign classification using CNNs  
- ROS 2 nodes for modular and scalable architecture  

---

## ⚙️ Technology Stack

### 🖥️ Platform
- Ubuntu (installed on Windows)
- ROS 2

### 🧩 Core Libraries
- Python
- OpenCV
- NumPy
- TensorFlow & TensorFlow Hub
- Matplotlib

### 🤖 ROS 2 Components
- ROS 2 Nodes
- Topics & Publishers
- Sensor message handling
- Image transport via `cv_bridge`

---

## 🚦 Major Functional Modules

### 🛣️ Lane Detection
- Camera calibration using chessboard images  
- Color and gradient thresholding  
- Perspective transformation (Bird’s Eye View)  
- Histogram-based lane pixel detection  
- Polynomial curve fitting for lane estimation  

### 🦅 Bird’s Eye View (BEV)
- Perspective warping for top-down road view  
- Improves lane geometry understanding  

### 🚘 Vehicle Detection
- SSD MobileNet (CPU-optimized)  
- Detects cars, buses, trucks, bicycles, and pedestrians  
- Frame skipping and resizing for performance optimization  

### 🚸 Traffic Sign Recognition
- CNN-based classifier trained on traffic sign datasets  
- Multi-class classification (43 classes)  
- Image normalization and evaluation pipeline  

### 🧩 ROS 2 Lane Detection Node
- Reads video input  
- Processes frames in real time  
- Publishes:
  - Lane-detected images  
  - Lane status messages  

---

## 📡 ROS 2 Architecture
- **Publisher Nodes:** Publish processed images and detection results  
- **Topics:**
  - `/lane_detected_image`
  - `/lane_status`
- **Timers:** Maintain near real-time frame processing  

---

## 🛠️ Setup & Execution (High-Level)
1. Install Ubuntu on Windows  
2. Install ROS 2 (recommended: Foxy / Humble)  
3. Install dependencies (OpenCV, TensorFlow, `cv_bridge`)  
4. Clone the repository into ROS 2 workspace  
5. Build the workspace using `colcon`  
6. Run the ROS 2 nodes  

---

## 🎯 Learning Outcomes
- Practical understanding of ROS 2 architecture  
- Real-time perception pipeline design  
- Computer vision for autonomous driving  
- Integration of classical CV with deep learning  
- Performance optimization for CPU-based systems  

---

## 🔮 Future Enhancements
- LiDAR integration  
- Sensor fusion (camera + LiDAR)  
- Path planning & control nodes  
- Real-time camera feed instead of video input  
- Deployment on embedded hardware  

---

## 👤 Author
**Mohammad Zaid**  
B.Tech (AI & ML) | Autonomous Systems & ROS 2 Enthusiast  

---

## ⚠️ Disclaimer
This project is intended for **academic, research, and learning purposes only** and does not represent a production-ready autonomous driving system.
