#!/usr/bin/env python3

import rospy
import rosbag
import rospkg
import math
import cv2
import sys
import os
import time
import struct

import argparse as ap
import numpy as np

from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry

from tqdm.auto import tqdm
from cv_bridge import CvBridge

from aarapsi_robot_pack.msg import ImageLabelStamped, CompressedImageLabelStamped, ImageOdom, CompressedImageOdom # Our custom msg structures

from pyaarapsi.vpr_simple import VPRImageProcessor, Tolerance_Mode, FeatureType, labelImage, makeImage, grey2dToColourMap, \
                                            doMtrxFig, updateMtrxFig, doDVecFig, updateDVecFig, doOdomFig, updateOdomFig
from pyaarapsi.core.enum_tools import enum_value_options, enum_get, enum_name
from pyaarapsi.core.argparse_tools import check_bounded_float, check_positive_float, check_positive_two_int_tuple, check_positive_int, check_bool, check_str_list, check_enum, check_string, check_string_list
from pyaarapsi.core.ros_tools import ROS_Param, roslogger, LogType, yaw_from_q
from pyaarapsi.core.helper_tools import vis_dict

class mrc:
    def __init__(self, odom_topic, img_topics, bag_path, rate, compress, save_path, save_name):
        self.odom_topic     = odom_topic
        if compress:
            self.img_topics = [i + "/compressed" if i.endswith("/compressed") else i for i in img_topics]
        else:
            self.img_topics = img_topics
        self.img_types      = [CompressedImage if i.endswith("/compressed") else Image for i in img_topics]
        self.topic_list     = [odom_topic] + img_topics

        self.bag_path       = bag_path
        self.rate           = rate
        self.compress       = compress
        self.save_path      = save_path
        self.save_name      = save_name

    def main(self):    
        odom_msg = Odometry()
        img_msgs = {topic: type() for type, topic in zip(self.img_types, self.img_topics)}
        odom_arr = []
        imgs_arr = []  
        logged_t = -1

        # Read rosbag
        roslogger("Ripping through rosbag ...", LogType.INFO)
        with rosbag.Bag(self.bag_path, 'r') as ros_bag:
            for topic, msg, timestamp in tqdm(ros_bag.read_messages(topics=self.topic_list)):
                if logged_t == -1:
                    logged_t = timestamp.to_sec()
                elif timestamp.to_sec() - logged_t > 1/self.rate:
                    odom_arr.append(odom_msg)
                    imgs_arr.append(img_msgs)
                    logged_t = timestamp.to_sec()
                
                if topic in self.img_topics:
                    img_msgs[topic] = msg
                elif topic == self.odom_topic:
                    odom_msg = msg
                else:
                    raise Exception("Type not in img_topics or odom_topic; how did we hit this? topic: %s" % (str(topic)))
        
        roslogger("Done! Converting stored messages (%s)" % (str(len(imgs_arr))), LogType.INFO)
        bridge = CvBridge()
        odom_dict = {'position': {'x': [], 'y': [], 'yaw': []}, 'velocity': {'x': [], 'y': [], 'yaw': []}}
        imgs_dict = {key: [] for key in list(new_imgs.keys())}

        compress_func = lambda msg: bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        raw_img_func = lambda msg: bridge.imgmsg_to_cv2(msg, "passthrough")

        for (new_odom, new_imgs) in tqdm(zip(odom_arr, imgs_arr)):
            odom_dict['position']['x'].append(new_odom.pose.pose.position.x)
            odom_dict['position']['y'].append(new_odom.pose.pose.position.y)
            odom_dict['position']['yaw'].append(yaw_from_q(new_odom.pose.pose.orientation))

            odom_dict['velocity']['x'].append(new_odom.twist.twist.linear.x)
            odom_dict['velocity']['y'].append(new_odom.twist.twist.linear.y)
            odom_dict['velocity']['yaw'].append(new_odom.twist.twist.angular.z)

            for topic in list(new_imgs.keys()):
                if "/compressed" in topic:
                    imgs_dict[topic].append(compress_func(new_imgs[topic]))
                else:
                    imgs_dict[topic].append(raw_img_func(new_imgs[topic]))

        master_dict = {'odom': odom_dict, 'imgs': imgs_dict}
        vis_dict(odom_dict)
        
if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="Bag Ripper", 
                            description="ROS Bag Ripper Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    save_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/data"
    # Positional Arguments:
    parser.add_argument('odom-topic',           type=check_string,                              help='Set odometry topic.')
    parser.add_argument('img-topics',           type=check_string_list,                         help='Set image topics (/compressed can be added to each to mix compressed and uncompressed topics).')
    parser.add_argument('bag-path',             type=check_string,                              help='Set system file path (including file name) for bag to read.')

    # Optional Arguments:
    parser.add_argument('--rate', '-r',         type=check_positive_float, default=10.0,        help='Set sample rate (default: %(default)s).')
    parser.add_argument('--compress', '-C',     type=check_bool,           default=False,       help='Force use of only compressed image topics (default: %(default)s).')
    parser.add_argument('--save-path', '-Sp',   type=check_string,         default=save_path,   help='Set output .npz file path (excluding file name) (default: %(default)s).')
    parser.add_argument('--save-name', '-Sn',   type=check_string,         default=None,        help='Set output .npz file name (default: %(default)s).')
    
    raw_args    = parser.parse_known_args()
    args        = vars(raw_args[0])

    odom_topic  = args['odom-topic']
    img_topics  = args['img-topics']
    bag_path    = args['bag-path']

    rate        = args['rate']
    compress    = args['compress']
    save_path   = args['save_path']
    save_name   = args['save_name']

    nmrc = mrc(odom_topic, img_topics, bag_path, rate, compress, save_path, save_name)
    try:
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO)
        sys.exit()
    except:
        roslogger("Error state reached, system exit triggered.", LogType.INFO)
