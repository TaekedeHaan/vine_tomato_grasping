#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:30:31 2020

@author: jelle
"""

import rospy
import math

from cv_bridge import CvBridge, CvBridgeError

# msg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from flex_grasp.msg import Tomato
from flex_grasp.msg import Truss
from flex_grasp.msg import Peduncle

# custom func
from detect_crop.ProcessImage import ProcessImage
from func.conversions import point_to_pose_stamped

from func.utils import camera_info2intrinsics

import pyrealsense2 as rs

# import pathlib
import os # os.sep

class ObjectDetection(object):
    """ObjectDetection"""
    def __init__(self):

        self.event = None

        self.color_image = None
        self.depth_image = None
        self.depth_info = None
        self.color_info = None
        self.trans = None
        self.init = None

        self.color_frame = None
        self.depth_frame = None

        self.camera_sim = rospy.get_param("camera_sim")
        self.use_truss = rospy.get_param("use_truss")
        self.debug_mode = rospy.get_param("object_detection/debug")

        self.bridge = CvBridge()

        pathCurrent = os.path.dirname(__file__) # path to THIS file
        self.pwdProcess = os.path.join(pathCurrent, '..', '..', 'results')

        rospy.loginfo("Storing visiual results in: ", self.pwdProcess)



        if self.debug_mode:
            log_level = rospy.DEBUG
            rospy.loginfo("[OBJECT DETECTION] Luanching object detection node in debug mode")
        else:
            log_level = rospy.INFO

        rospy.init_node("object_detection", anonymous=True, log_level=log_level)

        # Publish
        self.pub_e_out = rospy.Publisher("~e_out",
                                         String, queue_size=10, latch=True)

        self.pub_object_features = rospy.Publisher("object_features",
                                        Truss, queue_size=5, latch=True)

        self.pub_segment_image = rospy.Publisher("segment_image",
                                        Image, queue_size=5, latch=True)

        self.pub_tomato_image = rospy.Publisher("tomato_image",
                                Image, queue_size=5, latch=True)

        self.pub_color_hue = rospy.Publisher("color_hue",
                        Image, queue_size=5, latch=True)

        self.pub_color_saturation = rospy.Publisher("color_saturation",
                                Image, queue_size=5, latch=True)

        self.pub_color_A = rospy.Publisher("color_A",
                                Image, queue_size=5, latch=True)

        # Subscribe
        rospy.Subscriber("~e_in", String, self.e_in_cb)

        if not self.debug_mode:
            rospy.Subscriber("camera/color/image_raw", Image, self.color_image_cb)
            rospy.Subscriber("camera/depth/image_rect_raw", Image, self.depth_image_cb)
            rospy.Subscriber("camera/color/camera_info", CameraInfo, self.color_info_cb)
            rospy.Subscriber("camera/color/camera_info", CameraInfo, self.depth_info_cb)

    def e_in_cb(self, msg):
        if self.event is None:
            self.event = msg.data
            rospy.logdebug("[OBJECT DETECTION] Received object detection event message: %s", self.event)

            msg = String()
            msg.data = ""
            self.pub_e_out.publish(msg)

    def color_image_cb(self, msg):
        if (self.color_image is None) and (self.event == "e_start"):
            rospy.logdebug("[OBJECT DETECTION] Received color image message")
            try:
                self.color_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                self.color_frame = "camera_color_optical_frame" # msg.header.frame_id
            except CvBridgeError as e:
                print(e)

    def depth_image_cb(self, msg):
        if (self.depth_image is None) and (self.event == "e_start"):
            rospy.logdebug("[OBJECT DETECTION] Received depth image message")
            try:
                if self.camera_sim:
                    self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                else:
                    self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")/1000.0
                self.depth_frame = "camera_depth_optical_frame" # msg.header.frame_id
            except CvBridgeError as e:
                print(e)

    def color_info_cb(self, msg):
        if (self.color_info is None) and (self.event == "e_start"):
            rospy.logdebug("[OBJECT DETECTION] Received color info message")
            self.color_info = msg

    def depth_info_cb(self, msg):
        if (self.depth_info is None) and (self.event == "e_start"):
            rospy.logdebug("[OBJECT DETECTION] Received depth info message")
            self.depth_info = msg

    def received_all_data(self):
        return (self.color_image is not None) and (self.depth_image is not None) and (self.depth_info is not None) and (self.color_info is not None)

    def clear_all_data(self):
        self.color_image = None
        self.depth_image = None
        self.depth_info = None
        self.color_info = None

    def wait_for_data(self, time_out):
        while not self.received_all_data():
            rospy.sleep(0.1)
        return True


    def detect_object(self):


        if self.wait_for_data(5):
            pwd = os.path.dirname(__file__)


            image = ProcessImage(self.color_image,
                                 camera_sim = self.camera_sim,
                                 use_truss = self.use_truss,
                                 tomatoName = 'ros_tomato',
                                 pwdProcess = pwd,
                                 saveIntermediate = False)

            # rospy.logdebug("Image dimensions: %s", image.DIM)

            image.process_image()
            object_feature = image.get_object_features()

            # get results
            img_hue, img_saturation, img_A  = image.get_color_components()
            img_segment = image.get_segmented_image()
            img_tomato = image.get_tomato_visualization()
            frame = self.color_frame # "camera_color_optical_frame"

            #%%##################
            ### Cage location ###
            #####################

            row = object_feature['grasp']['row']
            col = object_feature['grasp']['col']
            angle = -object_feature['grasp']['angle'] # minus since camera frame is upside down...
            rpy = [0, 0, angle]

            intrin = camera_info2intrinsics(self.depth_info)
            xyz = self.deproject(row, col, intrin)
            cage_pose =  point_to_pose_stamped(xyz, rpy, frame, rospy.Time.now())

            #%%#############
            ### tomatoes ###
            ################
            tomatoes = []

            # rospy.logdebug("cols: %s [px]", col)
            for i in range(0, len(object_feature['tomato']['col'])):

                # Load from struct
                col = object_feature['tomato']['col'][i]
                row = object_feature['tomato']['row'][i]
                radius = object_feature['tomato']['radii'][i]

                point = self.deproject(row, col, intrin)

                depth = self.depth_image[(row, col)]
                point1 = rs.rs2_deproject_pixel_to_point(intrin, [0,0], depth)
                point2 = rs.rs2_deproject_pixel_to_point(intrin, [0,radius], depth)
                radius_m = euclidean(point1, point2)

                tomatoes.append(point_to_tomato(point, radius_m, frame))

            #%%#############
            ### Peduncle ###
            ################
            peduncle = Peduncle()
            peduncle.pose = cage_pose
            peduncle.radius = 0.01
            peduncle.length = 0.15

            truss = self.create_truss(tomatoes, cage_pose, peduncle)

            # publish results tomato_img
            imgmsg_segment = self.bridge.cv2_to_imgmsg(img_segment, encoding="rgb8")
            imgmsg_tomato = self.bridge.cv2_to_imgmsg(img_tomato, encoding="rgb8")
            imgmsg_hue = self.bridge.cv2_to_imgmsg(img_hue)
            imgmsg_saturation = self.bridge.cv2_to_imgmsg(img_saturation)
            imgmsg_A = self.bridge.cv2_to_imgmsg(img_A)

            rospy.loginfo("Publishing results")
            self.pub_segment_image.publish(imgmsg_segment)
            self.pub_tomato_image.publish(imgmsg_tomato)
            self.pub_color_hue.publish(imgmsg_hue)
            self.pub_color_saturation.publish(imgmsg_saturation)
            self.pub_color_A.publish(imgmsg_A)
            self.pub_object_features.publish(truss)

            return True
        else:
            rospy.logwarn("Did not receive all data")
            return False

    def generate_object(self):

        #%%##################
        ### Cage location ###
        #####################
        table_height = 0.23
        frame = "world"
        object_x = rospy.get_param("object_x")
        object_y = rospy.get_param("object_y")
        angle = rospy.get_param("object_angle")
        xyz = [object_x, object_y, 0.05 + table_height]
        rpy = [3.1415, 0, angle] #3.1415/2.0

        cage_pose =  point_to_pose_stamped(xyz, rpy, frame, rospy.Time.now())

        #%%#############
        ### Peduncle ###
        ################
        L = 0.15
        peduncle = Peduncle()
        peduncle.pose = cage_pose
        peduncle.radius = 0.005
        peduncle.length = L

        #%%#############
        ### tomatoes ###
        ################
        radii = [0.05, 0.05]
        t1x = xyz[0] + (L/2 + radii[0])*math.cos(angle)
        t1y = xyz[1] - (L/2 + radii[0])*math.sin(angle)
        t2x = xyz[0] - (L/2 + radii[1])*math.cos(angle)
        t2y = xyz[1] + (L/2 + radii[1])*math.sin(angle)
        point1 = [t1x, t1y, table_height]
        point2 = [t2x, t2y, table_height]
        points = [point1, point2]

        tomatoes = []
        for point, radius in zip(points, radii):
            # tomatoes.append(point_to_tomato(point, radius, frame))
            pass

        truss = self.create_truss(tomatoes, cage_pose, peduncle)
        self.pub_object_features.publish(truss)
        return True

    def create_truss(self, tomatoes, cage_pose, peduncle):
        #%%##########
        ### Truss ###
        #############
        truss = Truss()
        truss.tomatoes = tomatoes
        truss.cage_location = cage_pose
        truss.peduncle = peduncle

        return truss

    def deproject(self, row, col, intrin):
        # Deproject
        index = (row, col)
        depth = self.depth_image[index]
        # rospy.logdebug("Corresponding depth: %s", self.depth_image[index])
        # https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0

        pixel = [float(col), float(row)]
        depth = float(depth)

        point = rs.rs2_deproject_pixel_to_point(intrin, pixel, depth)
        return point

    def take_action(self):
        success = None
        msg = String()

        if (self.event == "e_start"):
            if not self.debug_mode:
                success = self.detect_object()
            if self.debug_mode:
                success = self.generate_object()

        elif (self.event == "e_init"):
            self.init = True
            success = True

        # publish success
        if success is not None:
            if success == True:
                self.clear_all_data()
                msg.data = "e_success"
                self.event = None

            elif success == False:
                msg.data = "e_failure"
                rospy.logwarn("Object detection failed to execute %s", self.event)
                self.event = None

            self.pub_e_out.publish(msg)

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5


def point_to_tomato(point, radius, frame):

    tomato = Tomato()
    tomato.header.frame_id = frame
    tomato.header.stamp = rospy.Time.now()

    tomato.position.x = point[0]
    tomato.position.y = point[1]
    tomato.position.z = point[2] + radius

    tomato.radius = radius
    return tomato



def main():
    try:
        object_detection = ObjectDetection()
        rospy.loginfo("[OBJECT DETECTION] Initialized")
        rate = rospy.Rate(10)

        while not rospy.core.is_shutdown():
            object_detection.take_action()
            rate.sleep()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == '__main__':
    main()
