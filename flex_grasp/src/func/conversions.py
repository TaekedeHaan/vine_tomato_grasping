#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:09:14 2020

@author: taeke
"""

from moveit_commander import MoveItCommanderException
from geometry_msgs.msg import Pose, PoseStamped, Transform, Point
from moveit_commander.conversions import pose_to_list

from tf.transformations import euler_from_quaternion, quaternion_from_euler

def pose_to_lists(pose_msg, orientation_type):
    print(orientation_type)
    pose = pose_to_list(pose_msg)
    position = pose[0:3]
    quaternion = pose[3:7]

    if orientation_type == "euler":
        euler = euler_from_quaternion(quaternion)
        orientation = euler
    elif orientation_type == "quaternion":
        orientation = quaternion
    else:
        raise MoveItCommanderException("Unknown type, accepts type 'euler' or 'quaternion'")

    return position, orientation


def position_to_list(position_msg):
    position = []

    position.append(position_msg.x)
    position.append(position_msg.y)
    position.append(position_msg.z)

    return position

def list_to_position(position_list):
    position_msg = Point()

    if len(pose_list) == 3:
        position_msg.x = position_list[0]
        position_msg.y = position_list[1]
        position_msg.z = position_list[2]
    else:
        raise MoveItCommanderException("Expected 3 elements in list: (x,y,z)")

    return position_msg


def orientation_to_list(orientation_msg):
    orientation = []

    orientation.append(orientation_msg.x)
    orientation.append(orientation_msg.y)
    orientation.append(orientation_msg.z)
    orientation.append(orientation_msg.w)

    return orientation