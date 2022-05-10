REALTIME OPENPOSE + POINT CLOUD
---

Requirements:

1. Install OpenPose: `https://github.com/CMU-Perceptual-Computing-Lab/openpose`
2. Install RealSense ROS packages: `https://github.com/IntelRealSense/realsense-ros`

To run:

1. `$ roslaunch realsense2_camera rs_rgbd.launch camera:=cam_0`
2. `$ roslaunch realtime_openpose realtime_openpose.py`
