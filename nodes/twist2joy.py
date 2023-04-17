#!/usr/bin/env python3

import rospy
import sys
import copy

import argparse as ap

from std_msgs.msg import Header, Int16
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

from pyaarapsi.core.argparse_tools import check_positive_float, check_string, check_bool
from pyaarapsi.core.ros_tools import roslogger, LogType, Heartbeat, NodeState

'''
ROS Twist->Joy Node

Convert a twist topic to a Joy message and publish
Used to interface with bluetooth controller topics
Configured to keep safety lock-outs from the ps4 controller
'''

class mrc:
    def __init__(self, twist_topic, joy_sub_topic, joy_pub_topic, node_name, anon, namespace, rate, log_level):

        self.node_name      = node_name
        self.namespace      = namespace
        self.anon           = anon
        self.log_level      = log_level
        self.rate_num       = rate
    
        rospy.init_node(self.node_name, anonymous=self.anon, log_level=self.log_level)
        roslogger('Starting %s node.' % (self.node_name), LogType.INFO, ros=True)
        self.rate_obj   = rospy.Rate(self.rate_num)
        self.heartbeat  = Heartbeat(self.node_name, self.namespace, NodeState.INIT, self.rate_num)

        self.joy_sub        = rospy.Subscriber(joy_sub_topic, Joy, self.joy_cb, queue_size=1)
        self.joy_pub        = rospy.Publisher(joy_pub_topic, Joy, queue_size=1)
        self.mode_pub       = rospy.Publisher(node_name + "/mode", Int16, queue_size=1)
        self.twist_sub      = rospy.Subscriber(twist_topic, Twist, self.twist_cb, queue_size=1)

        self.mode           = 0

        self.buttons        = [1] + [0] * 14
        self.axes           = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        self.slow_safety_i  = 4
        self.fast_safety_i  = 5
        self.linI           = 1
        self.angI           = 3

        self.slow_max       = 0.5
        self.fast_max       = 2.0

        self.empty_joy      = Joy(header=Header(frame_id="/dev/input/ps4"), axes=self.axes, buttons=self.buttons)

    def main(self):
        while not rospy.is_shutdown():
            self.rate_obj.sleep()

    def joy_cb(self, msg):
        if msg.buttons[0] > 0:
            return
        if msg.buttons[self.fast_safety_i] > 0:
            self.mode = 2
        elif msg.buttons[self.slow_safety_i] > 0:
            self.mode = 1
        else:
            self.mode = 0
        self.mode_pub.publish(Int16(data=self.mode))

    def twist_cb(self, msg):
        if self.mode == 0:
            return
        elif self.mode == 1:
            max = self.slow_max
            ind = self.slow_safety_i
        elif self.mode == 2:
            max = self.fast_max
            ind = self.fast_safety_i
        else:
            raise Exception("Unknown safety mode.")
        
        joy                 = copy.deepcopy(self.empty_joy)
        joy.header.stamp    = rospy.Time.now()

        joy.axes[self.linI] = msg.linear.x * max
        joy.axes[self.angI] = msg.angular.z
        self.buttons[ind]   = 1

        roslogger("Publishing ... ", logtype=LogType.DEBUG, ros=True, throttle=5)

        self.joy_pub.publish(joy)
        
if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="twist2joy.py", 
                            description="ROS Twist to Joy Republisher Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Positional Arguments:
    parser.add_argument('twist-topic',          type=check_string,                                    help='Set input geometry_msgs/Twist topic.')
    parser.add_argument('joy-sub-topic',        type=check_string,                                    help='Set input sensor_msgs/Joy topic.')
    parser.add_argument('joy-pub-topic',        type=check_string,                                    help='Set output sensor_msgs/Joy topic.')

    # Optional Arguments:
    parser.add_argument('--node-name',  '-N',   type=check_string,              default="twist2joy",  help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon',       '-a',   type=check_bool,                default=False,        help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',  '-n',   type=check_string,              default="/vpr_nodes", help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--rate',       '-r',   type=check_positive_float,      default=50.0,         help='Set publish rate (default: %(default)s).')
    parser.add_argument('--log-level',  '-V',   type=int, choices=[1,2,4,8,16], default=2,            help="Specify ROS log level (default: %(default)s).")
    
    raw_args    = parser.parse_known_args()
    args        = vars(raw_args[0])

    twist_topic = args['twist-topic']
    joy_sub     = args['joy-sub-topic']
    joy_pub     = args['joy-pub-topic']

    node_name   = args['node_name']
    anon        = args['anon']
    namespace   = args['namespace']
    rate        = args['rate']
    log_level   = args['log_level']

    try:
        nmrc = mrc(twist_topic, joy_sub, joy_pub, node_name, anon, namespace, rate, log_level)
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO, ros=True)
        sys.exit()
    except:
        roslogger("Error state reached, system exit triggered.", LogType.INFO, ros=True)
