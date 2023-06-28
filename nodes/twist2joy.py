#!/usr/bin/env python3

import rospy
import sys

import numpy as np
import argparse as ap

from enum import Enum

from sensor_msgs.msg import Joy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, TwistStamped

from pyaarapsi.core.argparse_tools import check_positive_float, check_string, check_bool, check_positive_int
from pyaarapsi.core.ros_tools import Base_ROS_Class, roslogger, LogType, NodeState, set_rospy_log_lvl
from pyaarapsi.core.enum_tools import enum_name, enum_value
from pyaarapsi.core.helper_tools import formatException
from pyaarapsi.vpr_simple.vpr_helpers import FeatureType

from aarapsi_robot_pack.srv import GetSafetyStates, GetSafetyStatesResponse

'''
ROS Twist->Joy Node

Convert a twist topic to a Joy message and publish
Used to interface with bluetooth controller topics
Configured to keep safety lock-outs from the ps4 controller
'''

class PS4_Buttons(Enum):
    X               = 0
    O               = 1
    Square          = 2
    Triangle        = 3
    LeftBumper      = 4
    RightBumper     = 5
    Share           = 6
    Options         = 7
    PS              = 8
    LeftStickIn     = 9
    RightStickIn    = 10
    LeftArrow       = 11
    RightArrow      = 12
    UpArrow         = 13
    DownArrow       = 14

class PS4_Triggers(Enum):
    LeftStickXAxis  = 0
    LeftStickYAxis  = 1
    LeftTrigger     = 2 # Released = 1, Pressed = -1
    RightStickXAxis = 3
    RightStickYAxis = 4
    RightTrigger    = 5 # Released = 1, Pressed = -1

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, twist_sub_topic, twist_pub_topic, joy_sub_topic, node_name, anon, namespace, rate_num, log_level, reset=True, order_id=0):
        super().__init__(node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30)

        self.init_params(rate_num, log_level, reset)
        self.init_vars()
        self.init_rospy(rate_num, joy_sub_topic, twist_sub_topic, twist_pub_topic)

        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_vars(self):
        super().init_vars()

        self.mode               = 0
        self.buttons            = [1] + [0] * 14                
        self.axes               = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

        self.enable             = enum_value(PS4_Buttons.X)
        self.disable            = enum_value(PS4_Buttons.O)
        self.slow_safety_i      = enum_value(PS4_Buttons.LeftBumper)
        self.fast_safety_i      = enum_value(PS4_Buttons.RightBumper)
        self.netvlad_ind        = enum_value(PS4_Buttons.LeftArrow)
        self.hybridnet_ind      = enum_value(PS4_Buttons.RightArrow)
        self.raw_ind            = enum_value(PS4_Buttons.UpArrow)
        self.patchnorm_ind      = enum_value(PS4_Buttons.DownArrow)
        self.linI               = enum_value(PS4_Triggers.LeftStickYAxis)
        self.angI               = enum_value(PS4_Triggers.RightStickXAxis)

        self.feat_arr           = {self.raw_ind: FeatureType.RAW,           self.patchnorm_ind: FeatureType.PATCHNORM, 
                                   self.netvlad_ind: FeatureType.NETVLAD,   self.hybridnet_ind: FeatureType.HYBRIDNET}

        self.slow_max_lin       = 0.2 # m/s
        self.fast_max_lin       = 0.4 # m/s

        self.slow_max_ang       = 0.2 # rad/s
        self.fast_max_ang       = 0.6 # rad/s

        self.enabled            = False
        self.parameters_ready   = True

        self.twist_msg          = Twist()

    def init_rospy(self, rate_num, joy_sub_topic, twist_sub_topic, twist_pub_topic):
        super().init_rospy()
        
        self.last_msg_time      = rospy.Time.now().to_sec()

        self.joy_sub            = rospy.Subscriber(joy_sub_topic, Joy, self.joy_cb, queue_size=1)
        self.twist_sub          = rospy.Subscriber(twist_sub_topic, TwistStamped, self.twist_cb, queue_size=1)
        self.twist_pub          = self.add_pub(twist_pub_topic, Twist, queue_size=1)
        self.srv_safety         = rospy.Service(self.namespace + '/safety', GetSafetyStates, self.handle_GetSafetyStates)

    def handle_GetSafetyStates(self, requ):
        ans                     = GetSafetyStatesResponse()
        ans.states.autonomous   = self.enabled
        ans.states.fast_mode    = self.mode == 2
        return ans

    def main(self):
        self.set_state(NodeState.MAIN)
        
        while not rospy.is_shutdown():
            self.rate_obj.sleep()

            self.twist_pub.publish(self.twist_msg)

    def joy_cb(self, msg):
        if abs(rospy.Time.now().to_sec() - msg.header.stamp.to_sec()) > 0.5: # if joy message was generated longer ago than half a second:
            self.mode = 0 
            self.twist_msg = Twist()
            self.print("Bad joy data.", LogType.WARN, throttle=5)
            return # bad data.

        # Toggle enable:
        if msg.buttons[self.enable] > 0:
            if not self.enabled == True:
                self.enabled = True
                rospy.logwarn("Autonomous mode: Enabled")
        elif msg.buttons[self.disable] > 0:
            if not self.enabled == False:
                self.enabled = False
                self.twist_msg = Twist()
                rospy.loginfo("Autonomous mode: Disabled")

        # Toggle mode:
        try:
            for i in self.feat_arr.keys():
                if msg.buttons[i] and (not self.FEAT_TYPE.get() == self.feat_arr[i]):
                    rospy.set_param(self.namespace + '/feature_type', enum_name(FeatureType.RAW))
                    rospy.loginfo("Switched to %s." % enum_name(self.FEAT_TYPE.get()))
                    break
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
                self.twist_msg = Twist()
                rospy.loginfo('Safety released.')

    def twist_cb(self, msg):
        if abs(rospy.Time.now().to_sec() - msg.header.stamp.to_sec()) > 0.5: # if twist message was generated longer ago than half a second:
            self.print("Bad twist data.", LogType.WARN, throttle=5)
            self.twist_msg = Twist()
            return # bad data.
        
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

        new_vx                      = np.sign(msg.twist.linear.x) * np.min([abs(msg.twist.linear.x), lin_max])
        new_vw                      = -1 * np.sign(msg.twist.angular.z)* np.min([abs(msg.twist.angular.z), ang_max])
        self.twist_msg.linear.x     = new_vx
        self.twist_msg.angular.z    = new_vw

        roslogger("vx: %s, vw: %s" % (str(new_vx), str(new_vw)), LogType.DEBUG, ros=True)
        
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
    parser.add_argument('--reset',            '-R',  type=check_bool,                   default=False,            help='Force reset of parameters to specified ones (default: %(default)s).')
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')

    raw_args    = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = Main_ROS_Class(args['twist-sub-topic'], args['twist-pub-topic'], args['joy-sub-topic'], args['node_name'], \
                   args['anon'], args['namespace'], args['rate'], args['log_level'], reset=args['reset'], order_id=args['order_id'])
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
