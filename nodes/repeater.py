#!/usr/bin/env python3

import rospy
import sys
import argparse as ap
from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import roslogger, LogType
from geometry_msgs.msg import Twist
from pyaarapsi.core.ros_tools import Heartbeat, NodeState

'''
ROS Repeater Tool

Repeat messages from bluetooth controller
Smooths over bluetooth trigger dead zones
'''

class mrc:
    def __init__(self, node_name, namespace, anon, log_level, rate):
        self.node_name      = node_name
        self.namespace      = namespace
        self.anon           = anon
        self.log_level      = log_level
        self.rate_num       = rate
    
        rospy.init_node(self.node_name, anonymous=self.anon, log_level=self.log_level)
        roslogger('Starting %s node.' % (self.node_name), LogType.INFO, ros=True)
        self.rate_obj   = rospy.Rate(self.rate_num)
        self.heartbeat  = Heartbeat(self.node_name, self.namespace, NodeState.INIT, self.rate_num)

        self.msg_in = Twist
        self.msg_time = 0
        self.msg_time_old = 0
        self.msg_received = False

        self.topic_in = "/jackal_velocity_controller/cmd_vel"
        self.topic_out = "/jackal_velocity_controller/cmd_vel"

        self.sub = rospy.Subscriber(self.topic_in, Twist, self.cmdvel_callback, queue_size=1)
        self.pub = rospy.Publisher(self.topic_out, Twist, queue_size=1)

    def main(self):
        self.heartbeat.set_state(NodeState.MAIN)
        
        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            self.msg_time = rospy.Time.now().to_sec()
            if (not self.msg_received): continue
            if (self.msg_time - self.msg_time_old) > (1/self.rate_num):
                self.pub.publish(self.msg_in)
            roslogger("Diff: %0.2f" % (self.msg_time - self.msg_time_old), LogType.INFO, ros=True)

    def cmdvel_callback(self, msg):
        if msg.angular.x < 0.01:
            self.msg_in = msg
            self.msg_in.angular.x = 0.05
            self.msg_time_old = self.msg_time
            self.msg_received = True

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="repeater.py", 
                            description="ROS Bluetooth Signal Repeater",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--node-name', '-N', type=check_string,              default="repeater",       help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate', '-r',      type=check_positive_float,      default=100.0,            help='Set node rate (default: %(default)s).')
    parser.add_argument('--anon', '-a',      type=check_bool,                default=False,            help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace', '-n', type=check_string,              default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2,                help="Specify ROS log level (default: %(default)s).")
    
    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])

    node_name   = args['node_name']
    rate        = args['rate']
    namespace   = args['namespace']
    anon        = args['anon']
    log_level   = args['log_level']

    try:
        nmrc = mrc(node_name, namespace, anon, log_level, rate)
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO, ros=True)
        sys.exit()
    except:
        roslogger("Error state reached, system exit triggered.", LogType.INFO, ros=True)
