#!/usr/bin/env python3

import rosbag
import rospkg
import sys
import os

import argparse as ap
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from pathlib import Path

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string, check_string_list
from pyaarapsi.core.ros_tools import roslogger, LogType, process_bag
from pyaarapsi.core.helper_tools import vis_dict, formatException

'''

ROS Bag Ripper

Open up a bag, grab out odom_topic and img_topics, and save to a compressed npz library

'''        

class mrc:
    def __init__(self, odom_topic, img_topics, bag_path, bag_name, rate, compress, save_path, save_name):
        self.odom_topic     = odom_topic
        if compress:
            self.img_topics = [i + "/compressed" if not i.endswith("/compressed") else i for i in img_topics]
        else:
            self.img_topics = img_topics
        self.img_types      = [CompressedImage if i.endswith("/compressed") else Image for i in self.img_topics]

        self.bag_path       = bag_path
        self.bag_name       = bag_name
        self.rate           = rate
        self.compress       = compress
        self.save_path      = save_path
        self.save_name      = save_name

        Path(self.bag_path).mkdir(parents=False, exist_ok=True)
        Path(self.save_path).mkdir(parents=False, exist_ok=True)

    def main(self):
        new_dict = process_bag(self.bag_path + '/' + self.bag_name, self.rate, self.odom_topic, self.img_topics, printer=lambda x: roslogger(x, LogType.INFO))
        vis_dict(new_dict)
        np.savez(self.save_path + "/" + self.save_name, **new_dict)
        
if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="Bag Ripper", 
                            description="ROS Bag Ripper Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    sp_default = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/data/ripped_bags"
    try:
        bp_default = os.environ['AARAPSI_ONEDRIVE_ROOT'] + '/3.0 Data/3.1 rosbags/improved'
    except:
        roslogger("Environment variable AARAPSI_ONEDRIVE_ROOT not set, defaulting to HOME as root instead.", LogType.WARN, ros=False)
        roslogger("AARAPSI_ONEDRIVE_ROOT should specify directory to (and including) aarapsi_home, i.e. /path/example/aarapsi_home.", LogType.WARN, ros=False)
        bp_default = os.environ['HOME'] + '/OneDrive/aarapsi_home/3.0 Data/3.1 rosbags/improved'
    ot_default = '/odom/filtered'
    it_default = '/ros_indigosdk_occam/image0'

    parser.add_argument('bag-name',             type=check_string,                             help='Set input .bag file name.')
    parser.add_argument('--odom-topic', '-Ot',  type=check_string,         default=ot_default, help='Set odometry topic.')
    parser.add_argument('--img-topics', '-It',  type=check_string_list,    default=it_default, help='Set image topics (/compressed can be added to each to mix compressed and uncompressed topics).')
    parser.add_argument('--bag-path',   '-Bp',  type=check_string,         default=bp_default, help='Set system directory (excluding file name) for bag to read.')
    parser.add_argument('--save-path',  '-Sp',  type=check_string,         default=sp_default, help='Set output .npz file path (excluding file name) (default: %(default)s).')
    parser.add_argument('--save-name',  '-Sn',  type=check_string,         default='rip.npz',  help='Set output .npz file name (default: %(default)s).')
    parser.add_argument('--rate',       '-r',   type=check_positive_float, default=10.0,       help='Set sample rate (default: %(default)s).')
    parser.add_argument('--compress',   '-C',   type=check_bool,           default=True,       help='Force use of only compressed image topics (default: %(default)s).')
    
    raw_args    = parser.parse_known_args()
    args        = vars(raw_args[0])

    bag_name    = args['bag-name']
    odom_topic  = args['odom_topic']
    img_topics  = args['img_topics']
    bag_path    = args['bag_path']
    save_path   = args['save_path']
    save_name   = args['save_name']
    rate        = args['rate']
    compress    = args['compress']

    try:
        nmrc = mrc(odom_topic, img_topics, bag_path, bag_name, rate, compress, save_path, save_name)
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO, ros=False) # False as rosnode likely terminated
        sys.exit()
    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)
