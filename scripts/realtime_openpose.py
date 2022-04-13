#!/usr/bin/env python3.7

mode = 'caffe'

import rospy
import sys
import numpy as np
import signal
import os
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from termcolor import cprint
import copy
import struct
from visualization_msgs.msg import Marker
import math
from geometry_msgs.msg import Point


if mode is 'caffe':
    try:
        root_dir = os.environ["HOME"]
        sys.path.append(root_dir + '/openpose/build/python')

        from openpose import pyopenpose as op
    except ImportError as e:
        print(e)
        print("Caffe OpenPose is not installed! Switching back to tf_pose instead.")
        mode = 'tf'


if mode is 'tf':
    from tf_pose import common
    from tf_pose.estimator import TfPoseEstimator
    from tf_pose.networks import model_wh


def shutdown(signum,stack):
    rospy.loginfo("Shutting down ROS...")
    rospy.signal_shutdown("Shutting down ROS")

class OpenPoseHandler(object):

    id_ = 1

    def __init__(self,cam_id):

        self.cam_id = cam_id

        # Neural Network Model parameters
        home_dir = os.environ["HOME"]

        if mode is 'tf':
            self.model = home_dir + '/tf-pose-estimation/models/graph/cmu/graph_opt.pb'

            self.w, self.h = model_wh('432x368')
            self.e = TfPoseEstimator(self.model, target_size=(432, 368))

        if mode is 'caffe':

            # Custom Params (refer to include/openpose/flags.hpp for more parameters)
            self.params = dict()
            self.params["model_folder"] = home_dir + "/openpose/models/"
            self.params["keypoint_scale"] = 3

            # Starting OpenPose
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(self.params)
            self.opWrapper.start()

            self.datum = op.Datum()

        self.previous_time = rospy.Time.now()
        self.cloud_msg = None
        self.current_xyz = None
        self.OP_DURATION = 1./5 # In seconds

        # Openpose publisher 
        self.openpose_pub = rospy.Publisher(f"/cam_{cam_id}/openpose",Image,queue_size=1)
        # markers publisher
        self.markers_pub = rospy.Publisher('/op_markers',Marker,queue_size=1)
        self.right_pub = rospy.Publisher('/op_right',Marker,queue_size=1)
        self.left_pub = rospy.Publisher('/op_left',Marker,queue_size=1)
        # Image subscriber
        self.sub = rospy.Subscriber(f"/cam_{cam_id}/color/image_raw",Image,self.openpose_callback,queue_size=1)
        # depth subscriber
        self.depth_sub = rospy.Subscriber(f"/cam_{cam_id}/depth_registered/points",PointCloud2,self.depth_callback,queue_size=1)


    def draw_humans(self, image, humans):
        npimg = np.copy(image)
        image_h, image_w = npimg.shape[:2]

        human_centers = []

        for human in humans:

            centers = {}
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))

                centers[i] = center
                cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
            human_centers.append(centers)

            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

        return npimg, human_centers

    def depth_callback(self,depth_msg):
        # Store the cloud msg 
        self.cloud_msg = depth_msg

    def openpose_callback(self, img_msg):

        new_time = rospy.Time.now()
        d3_coords = list()

        if new_time - self.previous_time > rospy.Duration(self.OP_DURATION):

            image = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)

            if mode is 'tf':

                # Run Open Pose on the images
                humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=4.0)
                op_img, _ = self.draw_humans(image, humans)

                ros_img = img_msg
                ros_img.data = op_img.tobytes()

            if mode is 'caffe':

                self.datum.cvInputData = image
                self.opWrapper.emplaceAndPop([self.datum])

                ros_img = img_msg
                ros_img.data = self.datum.cvOutputData.tobytes()

                # Project the openpose joints to the cloud:
                if self.datum.poseKeypoints.shape:

                    positions = self.datum.poseKeypoints[0]

                    for pos in positions:
                        if pos[0] != 0.0 and pos[1] != 0.0 and self.cloud_msg is not None:

                            img_x = int(pos[0] * self.cloud_msg.width + 0.5)
                            img_y = int(pos[1] * self.cloud_msg.height + 0.5)

                            try:
                                [x, y, z] = self.pixelTo3DPoint(self.cloud_msg, img_x, img_y)
                                self.current_xyz = copy.copy([x,y,z])
                            except TypeError as e:
                                print(e)
                            except Exception as e:
                                print(e)
                                cprint("[WARN] No Cloud data for this pixel? We keep the previous XYZ values","yellow")
                                # [x,y,z] = copy.copy(self.current_xyz)
                            
                            d3_coords.append(self.current_xyz)

            if d3_coords:
                # If we have 3d coordinates, create markers and publish
                markers_msg = self.pubmarkerSkeleton(-OpenPoseHandler.id_,new_time,d3_coords,[0,0,1],self.cam_id)
                # right_links_msg = self.RightLinks(OpenPoseHandler.id_,new_time,d3_coords,[0,0,1],self.cam_id)
                # left_links_msg = self.LeftLinks(OpenPoseHandler.id_,new_time,d3_coords,[0,0,1],self.cam_id)
                OpenPoseHandler.id_ += 1

                self.markers_pub.publish(markers_msg)
                # self.right_pub.publish(right_links_msg)
                # self.left_pub.publish(left_links_msg)

            self.openpose_pub.publish(ros_img)

            self.previous_time = new_time

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

    def pubmarkerSkeleton(self,id_, time, positions,rgb=[0,1,0],camera_num=1):
        ''' Prepares Rviz visualization data for OpenPose skeleton '''
        markerLines = Marker()
        markerLines.header.frame_id = "cam_{}_color_optical_frame".format(camera_num)
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
        for i in range(0, len(positions)):
            if not math.isnan(positions[i][0]) and not math.isnan(positions[i][1]) and not math.isnan(positions[i][2]):
                markerLines.points.append(Point(positions[i][0], positions[i][1], positions[i][2]))

        return markerLines

    def RightLinks(self,id, time, positions,rgb=[0,1,0],camera_num=1):
        ''' Prepares Rviz visualization data for OpenPose skeleton '''
        markerLines = Marker()
        markerLines.header.frame_id = "cam_{}_color_optical_frame".format(camera_num)
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

        if positions == list():
            return markerLines

        idx_1 = [0, 1, 1, 2, 2, 3]

        for i in idx_1:

            if not math.isnan(positions[i][0]) and not math.isnan(positions[i][1]) and not math.isnan(positions[i][2]):
                markerLines.points.append(Point(positions[i][0], positions[i][1], positions[i][2]))

        return markerLines

    def LeftLinks(self,id, time, positions,rgb=[0,1,0],camera_num=1):
        ''' Prepares Rviz visualization data for OpenPose skeleton '''
        markerLines = Marker()
        markerLines.header.frame_id = "cam_{}_color_optical_frame".format(camera_num)
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

        if positions == list():
            return markerLines

        idx_2 = [0, 4, 4, 5, 5, 6]

        for i in idx_2:

            if not math.isnan(positions[i][0]) and not math.isnan(positions[i][1]) and not math.isnan(positions[i][2]):
                markerLines.points.append(Point(positions[i][0], positions[i][1], positions[i][2]))

        return markerLines


if __name__ == "__main__":

    rospy.init_node("openpose_caller_ros",anonymous=True,disable_signals=True)
    signal.signal(signal.SIGINT,shutdown)

    OpenPose_cam1 = OpenPoseHandler(0)
    # OpenPose_cam2 = OpenPoseHandler(1) 

    rate = rospy.Rate(0.1) #Every 10 seconds

    rospy.loginfo("Running OpenPose real time...")
    rospy.spin()
