#!/usr/bin/env python3

from __future__ import print_function
import argparse
import cv2
import numpy as np
import os
import rospy
import yaml
import psutil
import time
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import WheelsCmdStamped
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from threading import Thread
from typing import Tuple, cast
from collections import namedtuple
from dt_class_utils import DTReminder

class StopForDuckie(DTROS):
    def __init__(self, node_name):
        super(StopForDuckie,self).__init__(node_name=node_name,node_type=NodeType.PERCEPTION)
        self.bridge = CvBridge()

        self.sub_img= rospy.Subscriber(
            "/goose/camera_node/image/compressed", CompressedImage, self.color_filter, buff_size=10000000, queue_size=1
        )

        self.pub_img = rospy.Publisher(
            "~debug/color_filter/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        self.pub_wheels_cmd = rospy.Publisher(
            "/goose/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1
        )

        self.veh_name = rospy.get_namespace().strip("/")

        # Set parameters using a robot-specific yaml file if such exists
        self.readParamFromFile()

        # Get static parameters
        self._baseline = rospy.get_param('~baseline')
        self._radius = rospy.get_param('~radius')
        self._k = rospy.get_param('~k')
        # Get editable parameters
        self._gain = DTParam(
            '~gain',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=3.0
        )
        self._trim = DTParam(
            '~trim',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=3.0
        )
        self._limit = DTParam(
            '~limit',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0
        )

        # Wait for the automatic gain control
        # of the camera to settle, before we stop it
        rospy.sleep(2.0)
        rospy.set_param(f"/{self.veh_name}/camera_node/exposure_mode" , 'off')

        self.log("Initialized")
        
    def color_filter(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return

        # Capture the video frame
        # by frame
        image = img

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # define the list of boundaries
        lower = np.array([80, 100, 100])
        upper = np.array([100, 255, 255])
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(hsv, lower, upper)

        ylwcnts = cv2.findContours(mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(ylwcnts)>0:
                yellow_area = max(ylwcnts, key=cv2.contourArea)
                (xg,yg,wg,hg) = cv2.boundingRect(yellow_area)
                cv2.rectangle(img,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)

        output = cv2.bitwise_and(img,img, mask = mask)
        # show the images

        image2 = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

        # Display the resulting frame
        frame_out = self.bridge.cv2_to_compressed_imgmsg(image)
        self.pub_img.publish(frame_out)

        if len(ylwcnts) == 0:
            left,right = self.speedToCmd(10,10)
            self.publish_motor(left,right)

        else:
            left,right = self.speedToCmd(0,0)
            self.publish_motor(left,right)
        

    def speedToCmd(self, speed_l, speed_r):
        """Applies the robot-specific gain and trim to the
        output velocities

        Applies the motor constant k to convert the deisred wheel speeds
        to wheel commands. Additionally, applies the gain and trim from
        the robot-specific kinematics configuration.

        Args:
            speed_l (:obj:`float`): Desired speed for the left
                wheel (e.g between 0 and 1)
            speed_r (:obj:`float`): Desired speed for the right
                wheel (e.g between 0 and 1)

        Returns:
            The respective left and right wheel commands that need to be
                packed in a `WheelsCmdStamped` message

        """

        # assuming same motor constants k for both motors
        k_r = self._k
        k_l = self._k

        # adjusting k by gain and trim
        k_r_inv = (self._gain.value + self._trim.value) / k_r
        k_l_inv = (self._gain.value - self._trim.value) / k_l

        # conversion from motor rotation rate to duty cycle
        u_r = speed_r * k_r_inv
        u_l = speed_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = self.trim(u_r,
                                -self._limit.value,
                                self._limit.value)
        u_l_limited = self.trim(u_l,
                                -self._limit.value,
                                self._limit.value)

        return u_l_limited, u_r_limited
     
    def readParamFromFile(self):
        """
        Reads the saved parameters from
        `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml` or
        uses the default values if the file doesn't exist. Adjsuts
        the ROS paramaters for the node with the new values.

        """
        # Check file existence
        fname = self.getFilePath(self.veh_name)
        # Use the default values from the config folder if a
        # robot-specific file does not exist.
        if not os.path.isfile(fname):
            self.log("Kinematics calibration file %s does not "
                     "exist! Using the default file." % fname, type='warn')
            fname = self.getFilePath('default')

        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

        # Set parameters using value in yaml file
        if yaml_dict is None:
            # Empty yaml file
            return
        for param_name in ["gain", "trim", "baseline", "k", "radius", "limit"]:
            param_value = yaml_dict.get(param_name)
            if param_name is not None:
                rospy.set_param("~"+param_name, param_value)
            else:
                # Skip if not defined, use default value instead.
                pass

    def trim(self, value, low, high):
        """
        Trims a value to be between some bounds.

        Args:
            value: the value to be trimmed
            low: the minimum bound
            high: the maximum bound

        Returns:
            the trimmed value
        """

        return max(min(value, high), low)

    def publish_motor(self, left, right):
        wheels_cmd_msg = WheelsCmdStamped()
        wheels_cmd_msg.header.stamp = rospy.Time.now()

        wheels_cmd_msg.vel_left = left
        wheels_cmd_msg.vel_right = right
       
        self.pub_wheels_cmd.publish(wheels_cmd_msg)

    def getFilePath(self, name):
        """
        Returns the path to the robot-specific configuration file,
        i.e. `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`.

        Args:
            name (:obj:`str`): the Duckiebot name

        Returns:
            :obj:`str`: the full path to the robot-specific
                calibration file

        """
        cali_file_folder = '/data/config/calibrations/kinematics/'
        cali_file = cali_file_folder + name + ".yaml"
        return cali_file

if __name__ == '__main__':
    node = StopForDuckie('StopForDuckie')
    rospy.spin()
