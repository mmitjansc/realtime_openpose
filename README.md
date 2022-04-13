REALTIME OPENPOSE + POINT CLOUD
---

Requirements:

1. Install OpenPose
2. Install RealSense ROS packages

To run:

1. `$ roslaunch realsense2_camera rs_rgbd.launch camera:=cam_0`
2. `$ roslaunch realtime_openpose realtime_openpose.py`