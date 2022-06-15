#!/usr/bin/env python3

import rospy
import os
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo
import sys
import struct
import numpy as np
from termcolor import cprint
import math
import signal

home_dir = os.environ["HOME"]

# Load OpenPose
sys.path.insert(0,home_dir+'/openpose/build/python')
from openpose import pyopenpose as op

# Load depth filtering networks
USE_TORCH = True
try:
    import torch
    import torch.nn as nn
    sys.path.insert(0,home_dir+'/catkin_ws/src/movement_assessment/activity_recognition/scripts')
    from trajectory_filter import TrajectoryPredictor
except Exception as e:
    USE_TORCH = False
    cprint("Can't open torch module, not filtering depth values.","yellow")

class OpenPoseFilter(object):

    _id = 1

    def __init__(self):
        global USE_TORCH

        # List of parts that we want to print
        self.list_of_parts = [1, 9, 10, 11, 12, 13, 14]
        self.arm_parts = [2,3,4,5,6,7]

        self.op_filter = None
        if USE_TORCH:
            try:
                nan_model   = torch.load(home_dir+'/catkin_ws/src/movement_assessment/activity_recognition/scripts/trained_nets/depth_nn_filter.pt',map_location='cpu')
                noise_model = torch.load(home_dir+'/catkin_ws/src/movement_assessment/activity_recognition/scripts/trained_nets/noise_removal_network.pt',map_location='cpu')
                self.op_filter = TrajectoryPredictor(nan_model,noise_model)
                torch.cuda.empty_cache()
                cprint("DNN models for filtering loaded.","green")
            except:
                USE_TORCH = False
                cprint("Can't load the models, not filtering depth values.","yellow")
                
        # OpenPose initialization
        self.params = dict()
        self.params["model_folder"] = home_dir+"/openpose/models/"
        self.params["keypoint_scale"] = 3
        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()
        self.datum = op.Datum()

        self.previous_time = rospy.Time.now()
        self.cloud_msg = None
        self.OP_DURATION = 1./5 # In seconds

        # Camera calibration matrix
        self.K = None

        # Openpose publisher 
        self.openpose_pub = rospy.Publisher(f"/camera/openpose",Image,queue_size=1)
        # markers publisher
        self.dnn_markers_pub = rospy.Publisher('/dnn_markers',Marker,queue_size=1)
        self.dnn_right_pub = rospy.Publisher('/dnn_right',Marker,queue_size=1)
        self.dnn_left_pub = rospy.Publisher('/dnn_left',Marker,queue_size=1)
        self.markers_pub = rospy.Publisher('/op_markers',Marker,queue_size=1)
        self.right_pub = rospy.Publisher('/op_right',Marker,queue_size=1)
        self.left_pub = rospy.Publisher('/op_left',Marker,queue_size=1)
        # Camera info subscriber
        self.cam_info_sub = rospy.Subscriber("/camera/color/camera_info",CameraInfo,self.camera_info_callback)
        # Image subscriber
        self.sub = rospy.Subscriber(f"/camera/color/image_raw",Image,self.openpose_callback,queue_size=1)
        # depth subscriber
        self.depth_sub = rospy.Subscriber(f"/camera/depth_registered/points",PointCloud2,self.depth_callback,queue_size=1)

    def camera_info_callback(self,msg):
        """Camera info callback to get the K"""
        self.K = np.array(msg.K).reshape((3,3))
        # Unsubscribe from the topic
        self.cam_info_sub.unregister()

    def shutdown(self,sig,frame):
        rospy.loginfo("Shutting down ROS...")
        rospy.signal_shutdown("Shutting down ROS")

    def depth_callback(self,depth_msg):
        # Store the cloud msg 
        self.cloud_msg = depth_msg

    def openpose_callback(self, img_msg):

        new_time = rospy.Time.now()
        d3_coords = list()
        arm_coords = list()

        if new_time - self.previous_time > rospy.Duration(self.OP_DURATION):

            image = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)

            self.datum.cvInputData = image
            try:
                self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
            except:
                self.opWrapper.emplaceAndPop([self.datum])

            ros_img = img_msg
            ros_img.data = self.datum.cvOutputData.tobytes()

            # Project the openpose joints to the cloud:
            if self.datum.poseKeypoints.shape:

                positions = self.datum.poseKeypoints[0]

                # We need to keep track of the part IDs so that we get only the ones we want
                part_id = 0

                for pos in positions:

                    if self.K and part_id in self.arm_parts and pos[0] != 0.0 and pos[1] != 0.0:
                        # It's an arm joint, store it differently IF self.K is already stored
                        img_x = int(pos[0] * self.cloud_msg.width + 0.5)
                        img_y = int(pos[1] * self.cloud_msg.height + 0.5)
                        arm_coords.append([img_x,img_y])

                    if part_id not in self.list_of_parts:
                        part_id += 1
                        continue

                    if pos[0] != 0.0 and pos[1] != 0.0 and self.cloud_msg is not None:

                        img_x = int(pos[0] * self.cloud_msg.width + 0.5)
                        img_y = int(pos[1] * self.cloud_msg.height + 0.5)

                        try:
                            [x, y, z] = self.pixelTo3DPoint(self.cloud_msg, img_x, img_y)
                            d3_coords.append([x,y,z])
                        except TypeError as e:
                            raise e
                        except Exception as e:
                            print(e)
                            cprint("[WARN] No Cloud data for this pixel? We keep the previous XYZ values","yellow")
                            d3_coords.append([float('nan') for _ in range(3)])
                    else:
                        d3_coords.append([float('nan') for _ in range(3)])

                    part_id += 1

            coords = np.array(d3_coords)
            if coords.size and np.isnan(coords[:,0]).sum() < 4:
                # Only publish coordinates if we have at lest 4 joints visible
                assert coords.shape == (7,3)

                # # If we have 3D coordianates, first filter them!
                if self.op_filter is not None:
                    self.op_filter.add_trajectory(coords)
                    new_coords = self.op_filter.predict_trajectory().squeeze().T
                    assert new_coords.shape == (7,3)

                    # Get the XYZ coordinates for the arms
                    if arm_coords:
                        # If we do have arms, append the markers to new_coords. 
                        # This should NOT affect the marker_links function, and should work with skeleton_markers
                        # Compute the plane first:
                        A,B,C,D = self.compute_plane(new_coords)
                        arm_joints = np.array(arm_coords)
                        unit_xyz = np.linalg.inv(self.K).dot(np.concatenate((arm_joints,np.ones((arm_joints.shape[0],1))),axis=1).T).T
                        z = -D/(A*unit_xyz[:,[0]] + B*unit_xyz[:,[1]] + C)
                        x = unit_xyz[:,[0]]*z
                        y = unit_xyz[:,[1]]*z
                        point_coord = np.concatenate((x,y,z),axis=1)
                        new_coords = np.concatenate((new_coords,point_coord),axis=0)

                    # Create markers and publish
                    markers_msg = self.skeleton_markers(-OpenPoseFilter._id,new_time,new_coords,[0,0,1])
                    right_links_msg = self.marker_links(OpenPoseFilter._id,new_time,new_coords,[0,1,1,2,2,3],[0,0,1])
                    left_links_msg = self.marker_links(OpenPoseFilter._id,new_time,new_coords,[0,4,4,5,5,6],[0,0,1])
                    OpenPoseFilter._id += 1

                    self.dnn_markers_pub.publish(markers_msg)
                    self.dnn_right_pub.publish(right_links_msg)
                    self.dnn_left_pub.publish(left_links_msg)

                # Create markers and publish raw OpenPose measurements
                markers_msg = self.skeleton_markers(-OpenPoseFilter._id,new_time,coords,[1,0,0])
                right_links_msg = self.marker_links(OpenPoseFilter._id,new_time,coords,[0,1,1,2,2,3],[1,0,0])
                left_links_msg = self.marker_links(OpenPoseFilter._id,new_time,coords,[0,4,4,5,5,6],[1,0,0])
                OpenPoseFilter._id += 1

                self.markers_pub.publish(markers_msg)
                self.right_pub.publish(right_links_msg)
                self.left_pub.publish(left_links_msg)

            self.openpose_pub.publish(ros_img)
            self.previous_time = new_time

    def compute_plane(self,coords):
        """Use the already filtered coordinates to compute the plane for the arms"""
        X = coords.copy() # For easier manipulation
        mean = X.mean(axis=0,keepdims=True)
        Cov = (X-mean).T.dot(X-mean) / (X.shape[0]-1)
        # Perform SVD
        U,S,Vt = np.linalg.svd(Cov)
        V = Vt.T
        # Get the plane
        A,B,C = V[:,2]
        D = -(A*X[:,0] + B*X[:,1] + C*X[:,2]).mean()
        plane = np.array([A,B,C,D])
        return plane


    def pixelTo3DPoint(self, cloud, x_img, y_img):

        width = cloud.width
        height = cloud.height
        point_step = cloud.point_step
        row_step = cloud.row_step

        array_pos = x_img * point_step + y_img * row_step

        bytesX = [x for x in cloud.data[array_pos:array_pos + 4]]
        bytesY = [x for x in cloud.data[array_pos + 4: array_pos + 8]]

        byte_format = struct.pack('4B', *bytesX)
        X = struct.unpack('f', byte_format)[0]

        byte_format = struct.pack('4B', *bytesY)
        Y = struct.unpack('f', byte_format)[0]

        bytesZ = [x for x in cloud.data[array_pos + 8:array_pos + 12]]
        byte_format = struct.pack('4B', *bytesZ)
        Z = struct.unpack('f', byte_format)[0]

        return [X, Y, Z]

    def skeleton_markers(self,id_, time, positions,rgb=[0,1,0]):
        ''' Prepares Rviz visualization data for OpenPose skeleton '''
        markerLines = Marker()
        markerLines.header.frame_id = "camera_color_optical_frame"
        markerLines.header.stamp = time
        markerLines.ns = "bones"
        markerLines.id = id_
        markerLines.action = Marker.ADD
        markerLines.type = Marker.POINTS
        markerLines.scale.x = 0.1
        markerLines.scale.y = 0.1
        markerLines.scale.z = 0.1
        markerLines.color.r = rgb[0]
        markerLines.color.g = rgb[1]
        markerLines.color.b = rgb[2]
        markerLines.color.a = 1.0
        markerLines.points = []
        markerLines.lifetime = rospy.Duration(self.OP_DURATION)
        for i in range(len(positions)):
            if not math.isnan(positions[i][0]) and not math.isnan(positions[i][1]) and not math.isnan(positions[i][2]):
                markerLines.points.append(Point(positions[i][0], positions[i][1], positions[i][2]))

        return markerLines

    def marker_links(self,id, time, positions,backpointers=[0,1,1,2,2,3],rgb=[0,1,0]):
        ''' Prepares Rviz visualization data for OpenPose skeleton '''
        markerLines = Marker()
        markerLines.header.frame_id = "camera_color_optical_frame"
        markerLines.header.stamp = time
        markerLines.ns = "bones"
        markerLines.id = id
        markerLines.type = Marker.LINE_STRIP
        markerLines.scale.x = 0.04
        markerLines.scale.y = 0.04
        markerLines.scale.z = 0.04
        markerLines.color.r = rgb[0]
        markerLines.color.g = rgb[1]
        markerLines.color.b = rgb[2]
        markerLines.color.a = 1.0
        markerLines.points = []
        markerLines.lifetime = rospy.Duration(self.OP_DURATION)
        if not len(positions):
            return markerLines
        for i in backpointers:
            if not math.isnan(positions[i][0]) and not math.isnan(positions[i][1]) and not math.isnan(positions[i][2]):
                markerLines.points.append(Point(positions[i][0], positions[i][1], positions[i][2]))

        return markerLines

    def run(self):
        print("Running OpenPose filter at {:.1f} Hz, waiting for images...".format(1./self.OP_DURATION))
        rospy.spin()

if __name__ == "__main__":

    rospy.init_node("realtime_openpose_filter")

    openpose_filter = OpenPoseFilter()
    signal.signal(signal.SIGINT,openpose_filter.shutdown)

    openpose_filter.run()