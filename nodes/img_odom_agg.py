#!/usr/bin/env python3

import rospy
import sys
import argparse as ap

from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from aarapsi_robot_pack.msg import ImageOdom, CompressedImageOdom

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, ROS_Home
from pyaarapsi.core.helper_tools import formatException

'''
Image+Odometry Aggregator

This node exists to fuse an image stream with an odometry stream 
into a single synchronised message structure that keeps images and
odometry aligned in time across networks. Without this node, if 
either the odometry or image stream fails, parts of down-pipeline
nodes may still act, likely to a poor result, or get into locked
or bad states. This node ensures down-pipeline nodes can only
progress if both odometry and images arrive across the network.

'''

class mrc:
    def __init__(self, node_name, rate, namespace, anon, img_topic, odom_topic, pub_topic, compressed, log_level, reset=True):

        self.node_name      = node_name
        self.namespace      = namespace
        self.nodespace      = self.namespace + "/" + self.node_name
        self.log_level      = log_level

        rospy.init_node(self.node_name, anonymous=anon, log_level=log_level)
        self.ROS_HOME       = ROS_Home(self.node_name, self.namespace, rate)
        self.print('Starting %s node.' % (self.node_name))

        self.init_params(rate, reset)
        self.init_vars(img_topic, odom_topic, pub_topic, compressed)
        self.init_rospy()

    def init_params(self, rate, reset):
        self.rate_num       = self.ROS_HOME.params.add(self.nodespace + "/rate",      rate,      check_positive_float,   force=reset)

    def init_vars(self, img_topic, odom_topic, pub_topic, compressed):
        self.img        = None
        self.odom       = None
        self.new_img    = False
        self.new_odom   = False

        self.odom_topic = odom_topic

        self.compressed = compressed
        if self.compressed:
            self.img_topic = img_topic + '/compressed'
            self.pub_topic = pub_topic + '/compressed'
            self.img_type  = CompressedImage
            self.pub_type  = CompressedImageOdom
        else:
            self.img_topic = img_topic
            self.pub_topic = pub_topic
            self.img_type  = Image
            self.pub_type  = ImageOdom

    def init_rospy(self):
        self.rate_obj   = rospy.Rate(self.rate_num.get())
        self.img_sub    = rospy.Subscriber(self.img_topic, self.img_type, self.img_cb, queue_size=1)
        self.odom_sub   = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.pub        = rospy.Publisher(self.namespace + self.pub_topic, self.pub_type, queue_size=1)

    def odom_cb(self, msg):
        self.odom       = msg
        self.new_odom   = True

    def img_cb(self, msg):
        self.img        = msg
        self.new_img    = True

    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            if not (self.new_img and self.new_odom):
                continue # denest
            self.new_img                = False
            self.new_odom               = False

            msg_to_pub                  = self.pub_type()
            msg_to_pub.header.stamp     = rospy.Time.now()
            msg_to_pub.header.frame_id  = self.odom.header.frame_id
            msg_to_pub.odom             = self.odom
            msg_to_pub.image            = self.img

            self.pub.publish(msg_to_pub)
            del msg_to_pub

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="image_odom_aggregator.py", 
                            description="ROS Image+Odom Aggregator Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--node-name', '-N', type=check_string,              default="img_odom_agg",    help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate', '-r',      type=check_positive_float,      default=10.0,              help='Set node rate (default: %(default)s).')
    parser.add_argument('--anon', '-a',      type=check_bool,                default=False,             help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace', '-n', type=check_string,              default="/vpr_nodes",      help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--img-topic',       type=check_string,              default='/occam/image0',   help="Specify input image topic (exclude /compressed) (default: %(default)s).")
    parser.add_argument('--odom-topic',      type=check_string,              default='/odom/filtered',  help="Specify input odometry topic (exclude /compressed) (default: %(default)s).")
    parser.add_argument('--pub-topic',       type=check_string,              default='/img_odom',       help="Specify output topic (exclude /compressed) (default: %(default)s).")
    parser.add_argument('--compress', '-C',  type=check_bool,                default=True,              help='Enable image compression (default: %(default)s)')
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2,                 help="Specify ROS log level (default: %(default)s).")
    
    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])

    node_name   = args['node_name']
    rate        = args['rate']
    namespace   = args['namespace']
    anon        = args['anon']
    img_topic   = args['img_topic']
    odom_topic  = args['odom_topic']
    pub_topic   = args['pub_topic']
    compressed  = args['compress']
    log_level   = args['log_level']

    try:
        nmrc = mrc(node_name, rate, namespace, anon, img_topic, odom_topic, pub_topic, compressed, log_level)
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