#!/usr/bin/env python3

import rospy
import sys
import argparse as ap
import cv2 
import numpy as np
import rospkg
import os
from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import roslogger, LogType
from pyaarapsi.core.ros_tools import Heartbeat, NodeState, img2msg

'''
ROS Image Spinner Tool

Load in an image and publish it at a fixed rate, slowly moving 
it horizontally one pixel at a time.
'''

class mrc:
    def __init__(self, node_name, namespace, anon, log_level, rate, img_path):
        self.node_name  = node_name
        self.namespace  = namespace
        self.anon       = anon
        self.log_level  = log_level
        self.rate_num   = rate
        self.img_path   = img_path
    
        rospy.init_node(self.node_name, anonymous=self.anon, log_level=self.log_level)
        roslogger('Starting %s node.' % (self.node_name), LogType.INFO, ros=True)
        self.rate_obj   = rospy.Rate(self.rate_num)
        self.heartbeat  = Heartbeat(self.node_name, self.namespace, NodeState.INIT, self.rate_num)

        self.img        = cv2.imread(self.img_path)
        self.bridge     = CvBridge()
        self.pub        = rospy.Publisher(self.namespace + '/image_spin/compressed', CompressedImage, queue_size=1)

    def main(self):
        self.heartbeat.set_state(NodeState.MAIN)
        self.seq                = 0
        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            self.img            = np.roll(self.img, 1, 1)
            msg                 = img2msg(self.img[:,0:self.img.shape[0],:], 'CompressedImage', bridge=self.bridge)
            msg.header.stamp    = rospy.Time.now()
            msg.header.seq      = self.seq
            msg.header.frame_id = "image"
            self.seq            = self.seq + 1
            self.pub.publish(msg)
            del msg

if __name__ == '__main__':

    img_path_default = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/media/pano1.jpg"

    parser = ap.ArgumentParser(prog="image_spin.py", 
                            description="ROS Image Spinner Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--node-name', '-N', type=check_string,              default="image_spin",     help="Specify node name (default: %(default)s).")
    parser.add_argument('--namespace', '-n', type=check_string,              default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--anon',      '-a', type=check_bool,                default=False,            help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--rate',      '-r', type=check_positive_float,      default=20.0,             help='Set node rate (default: %(default)s).')
    parser.add_argument('--img-path',  '-p', type=check_string,              default=img_path_default, help="Specify ROS namespace (default: %(default)s).")
    
    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])

    node_name   = args['node_name']
    namespace   = args['namespace']
    anon        = args['anon']
    log_level   = args['log_level']
    rate        = args['rate']
    img_path    = args['img_path']

    try:
        nmrc = mrc(node_name, namespace, anon, log_level, rate, img_path)
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO, ros=True)
        sys.exit()
    except:
        roslogger("Error state reached, system exit triggered.", LogType.INFO, ros=True)
