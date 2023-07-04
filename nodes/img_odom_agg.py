#!/usr/bin/env python3

import rospy
import sys
import argparse as ap

import cv2
from cv_bridge import CvBridge

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from aarapsi_robot_pack.msg import ImageOdom

from pyaarapsi.core.argparse_tools  import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools       import NodeState, roslogger, LogType, set_rospy_log_lvl
from pyaarapsi.core.helper_tools    import formatException, np_ndarray_to_uint8_list
from pyaarapsi.core.enum_tools      import enum_value_options
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

'''
Image+Odometry Aggregator

This node exists to fuse an image stream with an odometry stream 
into a single synchronised message structure that keeps images and
odometry aligned in time across networks. Without the use of this 
node, if either the odometry or image stream fails, nodes further 
down the pipeline may still act, likely to a poor result, or get 
into locked or bad states. This node ensures down-pipeline nodes can
only progress if both odometry and images arrive across the network.

'''

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_params(self, rate_num, log_level, reset):
        super().init_params(rate_num, log_level, reset)
        
    def init_vars(self):
        super().init_vars()

        self.img        = None
        self.odom       = None
        self.new_img    = False
        self.new_odom   = False
        self.time       = rospy.Time.now().to_sec()

        self.bridge     = CvBridge()

        if self.IMG_TOPIC.get().endswith('compressed'):
            self.img_type       = CompressedImage
            self.img_convert    = lambda img: np_ndarray_to_uint8_list(self.bridge.compressed_imgmsg_to_cv2(img, "bgr8"))
        else:
            self.img_type       = Image
            self.img_convert    = lambda img: np_ndarray_to_uint8_list(self.bridge.imgmsg_to_cv2(img, "passthrough"))

    def init_rospy(self):
        super().init_rospy()
        
        self.img_sub    = rospy.Subscriber(self.IMG_TOPIC.get(), self.img_type, self.img_cb, queue_size=1)
        self.odom_sub   = rospy.Subscriber(self.SLAM_ODOM_TOPIC.get(), Odometry, self.odom_cb, queue_size=1)
        self.pub        = self.add_pub(self.namespace + '/img_odom', ImageOdom, queue_size=1)

    def odom_cb(self, msg):
        self.odom       = msg
        self.new_odom   = True

    def img_cb(self, msg):
        self.img        = msg
        self.new_img    = True

    def main(self):
        self.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.loop_contents()

    def loop_contents(self):

        if rospy.Time.now().to_sec() - self.time > 1:
            self.print("Long message delay experienced.", LogType.WARN, throttle=5)

        if not (self.new_img and self.new_odom):
            rospy.sleep(0.005)
            return # denest
        self.rate_obj.sleep()
        self.time       = rospy.Time.now().to_sec()

        self.new_img                = False
        self.new_odom               = False

        msg_to_pub                  = ImageOdom()
        msg_to_pub.header.stamp     = rospy.Time.now()
        msg_to_pub.header.frame_id  = self.odom.header.frame_id
        msg_to_pub.odom             = self.odom
        msg_to_pub.image            = self.img_convert(self.img)

        self.pub.publish(msg_to_pub)
        del msg_to_pub

def do_args():
    parser = ap.ArgumentParser(prog="img_odom_agg.py", 
                            description="ROS Image+Odom Aggregator Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='img_odom_agg')

    # Parse args...
    return vars(parser.parse_known_args()[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = Main_ROS_Class(**args)
        nmrc.print("Initialisation complete.", LogType.INFO)
        nmrc.main()
        nmrc.print("Operation complete.", LogType.INFO, ros=False) # False as rosnode likely terminated
        sys.exit()
    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)