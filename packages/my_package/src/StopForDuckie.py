#!/usr/bin/env python3

from __future__ import print_function
import argparse
import cv2
import numpy as np
import os
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
import psutil
import time
from threading import Thread
from typing import Tuple, cast
from collections import namedtuple
from dt_class_utils import DTReminder

class StopForDuckie(DTROS):
    def __init__(self, node_name):
        super(StopForDuckie,self).__init__(node_name=node_name,node_type=NodeType.PERCEPTION)
        self.bridge = CvBridge()

        self.sub_img= rospy.Subscriber(
            "/goose/camera_node/image/compressed", CompressedImage, self.callback, buff_size=10000000, queue_size=1
        )

        self.pub_img = rospy.Publisher(
            "~debug/videoFilter/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        self.pub_img2 = rospy.Publisher(
            "~debug/videoFilter2/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        
    def callback(self, msg):
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

if __name__ == '__main__':
    node = StopForDuckie('StopForDuckie')
    rospy.spin()
