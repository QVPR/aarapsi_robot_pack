#!/usr/bin/env python3

import rospy
import sys
import argparse as ap

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from aarapsi_robot_pack.msg import ImageOdom, CompressedImageOdom

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string, check_positive_int
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, init_node, set_rospy_log_lvl
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
    def __init__(self, node_name, rate_num, namespace, anon, img_topic, odom_topic, pub_topic, compressed, log_level, reset=True, order_id=0):

        init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30)

        self.init_params(rate_num, log_level, reset)
        self.init_vars(img_topic, odom_topic, pub_topic, compressed)
        self.init_rospy()

        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_params(self, rate, log_level, reset):
        self.rate_num       = self.ROS_HOME.params.add(self.nodespace + "/rate",      rate,      check_positive_float,   force=reset)
        self.log_level      = self.ROS_HOME.params.add(self.nodespace + "/log_level", log_level, check_positive_int,     force=reset)

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
        self.param_sub  = rospy.Subscriber(self.namespace + "/params_update", String, self.param_cb, queue_size=100)
        self.img_sub    = rospy.Subscriber(self.img_topic, self.img_type, self.img_cb, queue_size=1)
        self.odom_sub   = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.pub        = self.ROS_HOME.add_pub(self.pub_topic, self.pub_type, queue_size=1)

    def odom_cb(self, msg):
        self.odom       = msg
        self.new_odom   = True

    def img_cb(self, msg):
        self.img        = msg
        self.new_img    = True

    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            if not (self.new_img and self.new_odom):
                rospy.sleep(0.005)
                continue # denest
            self.rate_obj.sleep()
            self.new_img                = False
            self.new_odom               = False

            msg_to_pub                  = self.pub_type()
            msg_to_pub.header.stamp     = rospy.Time.now()
            msg_to_pub.header.frame_id  = self.odom.header.frame_id
            msg_to_pub.odom             = self.odom
            msg_to_pub.image            = self.img

            self.pub.publish(msg_to_pub)
            del msg_to_pub

    def param_cb(self, msg):
        if self.ROS_HOME.params.exists(msg.data):
            self.print("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG)
            self.ROS_HOME.params.update(msg.data)

            if msg.data == self.log_level.name:
                set_rospy_log_lvl(self.log_level.get())
            elif msg.data == self.rate_num.name:
                self.rate_obj = rospy.Rate(self.rate_num.get())
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

def do_args():
    parser = ap.ArgumentParser(prog="image_odom_aggregator.py", 
                            description="ROS Image+Odom Aggregator Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="img_odom_agg",   help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=10.0,             help='Set node rate (default: %(default)s).')
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=False,            help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--img-topic',               type=check_string,                 default='/occam/image0',  help="Specify input image topic (exclude /compressed) (default: %(default)s).")
    parser.add_argument('--odom-topic',              type=check_string,                 default='/odom/filtered', help="Specify input odometry topic (exclude /compressed) (default: %(default)s).")
    parser.add_argument('--pub-topic',               type=check_string,                 default='/img_odom',      help="Specify output topic (exclude /compressed) (default: %(default)s).")
    parser.add_argument('--compress',         '-C',  type=check_bool,                   default=True,             help='Enable image compression (default: %(default)s)')
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')
    
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = mrc(args['node_name'], args['rate'], args['namespace'], args['anon'], args['img_topic'], \
                   args['odom_topic'], args['pub_topic'], args['compress'], args['log_level'], order_id=args['order_id'])
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