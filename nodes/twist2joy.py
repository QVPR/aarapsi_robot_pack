#!/usr/bin/env python3

import rospy
import sys
import copy

import numpy as np
import argparse as ap

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

from pyaarapsi.core.argparse_tools import check_positive_float, check_string, check_bool
from pyaarapsi.core.ros_tools import roslogger, init_node, LogType, Heartbeat, NodeState
from pyaarapsi.core.enum_tools import enum_name
from pyaarapsi.vpr_simple.vpr_helpers import FeatureType

from aarapsi_robot_pack.srv import GetSafetyStates, GetSafetyStatesResponse

'''
ROS Twist->Joy Node

Convert a twist topic to a Joy message and publish
Used to interface with bluetooth controller topics
Configured to keep safety lock-outs from the ps4 controller
'''

class mrc:
    def __init__(self, twist_sub_topic, twist_pub_topic, joy_sub_topic, node_name, anon, namespace, rate_num, log_level, order_id=0):

        init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30)

        self.init_params()
        self.init_vars()
        self.init_rospy(rate_num, joy_sub_topic, twist_sub_topic, twist_pub_topic)

        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_params(self):
        pass

    def init_vars(self):
        self.mode           = 0
        # Button order:
        # X, O, Square, Triangle, LeftB, RightB, Share, Options, PS, LStickIn, RStickIn, LeftAr, RightAr, UpAr, DownAr
        # 0  1  2       3         4      5       6      7        8   9         10        11      12       13    14
        self.buttons        = [1] + [0] * 14
        # Axes order:
        # LStickX{L=1,R=-1}, LStickY{U=1,D=-1}, LTrigger(Released=1, Press=-1) RStickX{L=1,R=-1}, RStickY{U=1,D=-1}, RTrigger(Released=1, Press=-1)
        # 0                  1                  2                              3                  4                  5                  
        self.axes           = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

        # Button assignment:
        self.enable         = 0
        self.disable        = 1
        self.slow_safety_i  = 4
        self.fast_safety_i  = 5

        self.netvlad_ind    = 11
        self.hybridnet_ind  = 12
        self.raw_ind        = 13
        self.patchnorm_ind  = 14

        # Axes assignment:
        self.linI           = 1
        self.angI           = 3

        self.slow_max_lin   = 0.2
        self.fast_max_lin   = 0.4

        self.slow_max_ang   = 0.2
        self.fast_max_ang   = 0.6

        self.enabled        = False
        self.feature_mode   = FeatureType.RAW

    def init_rospy(self, rate_num, joy_sub_topic, twist_sub_topic, twist_pub_topic):
        self.rate_obj       = rospy.Rate(rate_num)

        self.joy_sub        = rospy.Subscriber(joy_sub_topic, Joy, self.joy_cb, queue_size=1)
        self.twist_sub      = rospy.Subscriber(twist_sub_topic, Twist, self.twist_cb, queue_size=1)
        self.twist_pub      = rospy.Publisher(twist_pub_topic, Twist, queue_size=1)
        self.srv_safety     = rospy.Service(self.namespace + '/safety', GetSafetyStates, self.handle_GetSafetyStates)

    def handle_GetSafetyStates(self, requ):
        ans = GetSafetyStatesResponse()
        ans.states.autonomous = self.enable
        ans.states.fast_mode = (self.mode==2)
        return ans

    def main(self):
        while not rospy.is_shutdown():
            self.rate_obj.sleep()

    def joy_cb(self, msg):
        # Toggle enable:
        if msg.buttons[self.enable] > 0:
            if not self.enabled == True:
                self.enabled = True
                rospy.logwarn("Autonomous mode: Enabled")
        elif msg.buttons[self.disable] > 0:
            if not self.enabled == False:
                self.enabled = False
                rospy.loginfo("Autonomous mode: Disabled")

        # Toggle mode:
        try:
            if msg.buttons[self.raw_ind]:
                if not self.feature_mode == FeatureType.RAW:
                    self.feature_mode = FeatureType.RAW
                    rospy.set_param(self.namespace + '/feature_type', enum_name(self.feature_mode))
                    rospy.loginfo("Switched to %s." % enum_name(self.feature_mode))
            elif msg.buttons[self.patchnorm_ind]:
                if not self.feature_mode == FeatureType.PATCHNORM:
                    self.feature_mode = FeatureType.PATCHNORM
                    rospy.set_param(self.namespace + '/feature_type', enum_name(self.feature_mode))
                    rospy.loginfo("Switched to %s." % enum_name(self.feature_mode))
            elif msg.buttons[self.netvlad_ind]:
                if not self.feature_mode == FeatureType.NETVLAD:
                    self.feature_mode = FeatureType.NETVLAD
                    rospy.set_param(self.namespace + '/feature_type', enum_name(self.feature_mode))
                    rospy.loginfo("Switched to %s." % enum_name(self.feature_mode))
            elif msg.buttons[self.hybridnet_ind]:
                if not self.feature_mode == FeatureType.HYBRIDNET:
                    self.feature_mode = FeatureType.HYBRIDNET
                    rospy.set_param(self.namespace + '/feature_type', enum_name(self.feature_mode))
                    rospy.loginfo("Switched to %s." % enum_name(self.feature_mode))
        except:
            rospy.logdebug_throttle(60, "Param switching is disabled for rosbags :-(")

        # Toggle safety:
        if msg.buttons[self.fast_safety_i] > 0:
            if not self.mode == 2:
                self.mode = 2
                rospy.logerr('Fast mode enabled.')
        elif msg.buttons[self.slow_safety_i] > 0:
            if not self.mode == 1:
                self.mode = 1
                rospy.logwarn('Slow mode enabled.')
        else:
            if not self.mode == 0:
                self.mode = 0
                rospy.loginfo('Safety released.')

    def twist_cb(self, msg):
        if self.mode == 0 or not self.enabled:
            return
        elif self.mode == 1:
            lin_max = self.slow_max_lin
            ang_max = self.slow_max_ang
        elif self.mode == 2:
            lin_max = self.fast_max_lin
            ang_max = self.fast_max_ang
        else:
            raise Exception("Unknown safety mode.")

        new_vx              = np.sign(msg.linear.x) * np.min([abs(msg.linear.x), lin_max])
        new_vw              = -1 * np.sign(msg.angular.z)* np.min([abs(msg.angular.z), ang_max])
        new_twist           = Twist()
        new_twist.linear.x  = new_vx
        new_twist.angular.z = new_vw

        roslogger("Publishing ... ", logtype=LogType.DEBUG, ros=True, throttle=5)
        roslogger("vx: %s, vw: %s" % (str(new_vx), str(new_vw)), LogType.INFO, ros=True)
        self.twist_pub.publish(new_twist)
        
def do_args():
    parser = ap.ArgumentParser(prog="twist2joy.py", 
                            description="ROS Twist to Joy Republisher Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Positional Arguments:
    parser.add_argument('twist-sub-topic',           type=check_string,                                           help='Set input geometry_msgs/Twist topic.')
    parser.add_argument('twist-pub-topic',           type=check_string,                                           help='Set output geometry_msgs/Twist topic.')
    parser.add_argument('joy-sub-topic',             type=check_string,                                           help='Set input sensor_msgs/Joy topic.')

    # Optional Arguments:
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="twist2joy",      help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=False,            help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=50.0,             help='Set publish rate (default: %(default)s).')
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')
    
    raw_args    = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = mrc(args['twist-sub-topic'], args['twist-pub-topic'], args['joy-sub-topic'], args['node_name'], \
                   args['anon'], args['namespace'], args['rate'], args['log_level'], order_id=args['order_id'])
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO, ros=True)
        sys.exit()
    except:
        roslogger("Error state reached, system exit triggered.", LogType.INFO, ros=True)
