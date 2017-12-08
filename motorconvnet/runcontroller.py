#!/usr/bin/env python

import sys
import rospy
import os
import time
import numpy
import tf
import math

from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion, Twist, Wrench
from std_srvs.srv import Empty
from scipy.signal import savgol_filter
from matplotlib import pyplot


rospy.wait_for_service('gazebo/apply_joint_effort')
rospy.wait_for_service('gazebo/get_link_state')
get_link_state = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
set_link_state = rospy.ServiceProxy('gazebo/set_link_state', SetLinkState)
apply_joint_effort = rospy.ServiceProxy('gazebo/apply_joint_effort', ApplyJointEffort)
reset_sim = rospy.ServiceProxy('gazebo/reset_simulation', Empty)


def euler_to_quaternion(roll, pitch, yaw):
    tf_quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    return Quaternion(tf_quat[0], tf_quat[1], tf_quat[2], tf_quat[3])


def quaternion_to_euler(orientation):
    return tf.transformations.euler_from_quaternion(
        (orientation.x, orientation.y, orientation.z, orientation.w))


prev_delta = {'upper_link': 0, 'lower_link': 0, 'lower_link_clone': 0}


def set_joint_alpha(parent, link, joint, target, limits, pgain=50, dgain=500):
    parent_state = get_link_state(parent, None).link_state
    parent_angle = quaternion_to_euler(parent_state.pose.orientation)[0]
    state = get_link_state(link, None).link_state
    current_angle = quaternion_to_euler(state.pose.orientation)[0] - parent_angle
    if current_angle < limits[0] - 0.1 * (limits[1] - limits[0]):
        current_angle += 2 * math.pi
    if current_angle > limits[1] + 0.1 * (limits[1] - limits[0]):
        current_angle -= 2 * math.pi
    current_alpha = (current_angle - limits[0]) / (limits[1] - limits[0])
    delta = target - current_alpha
    effort = pgain * delta + dgain * (delta - prev_delta[link])
    prev_delta[link] = delta
    apply_effort(joint, effort)
    print('{} {}'.format(current_alpha, effort))


def get_hand_position():
    return get_link_state('lower_link_clone', None).link_state.pose.position


def apply_effort(joint, effort):
    start = None
    duration = genpy.Duration()
    duration.secs = 0
    duration.nsecs = 2e7
    resp = apply_joint_effort(joint, effort, start, duration)


if __name__ == '__main__':
    try:
        time.sleep(1)
        rospy.init_node('armcontroller')
        rate = rospy.Rate(50)
        for run in range(1000, 5000):
            rospy.loginfo('Run {}'.format(run))
            reset_sim()
            joint_efforts = 100 * (numpy.random.rand(250, 3) - 0.5)
            joint_efforts = savgol_filter(joint_efforts, 31, 3, axis=0)
            hand_positions = numpy.zeros((250, 2))
            row = 0
            while not rospy.is_shutdown() and row < 250:
                apply_effort('ShoulderJoint', joint_efforts[row,0])
                apply_effort('ElbowJoint', joint_efforts[row,1])
                apply_effort('WristJoint', joint_efforts[row,2])
                pos = get_hand_position()
                hand_positions[row,0] = pos.x
                hand_positions[row,1] = pos.z
                row += 1
                rate.sleep()
            numpy.save('/home/tim/data/joints{}'.format(run), joint_efforts)
            numpy.save('/home/tim/data/hand{}'.format(run), hand_positions)
    except rospy.ROSInterruptException:
        pass
