#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from std_msgs.msg import String
from pyaarapsi.core.argparse_tools import check_positive_float, check_positive_int, check_bool, check_string
from pyaarapsi.core.ros_tools import Heartbeat, NodeState, roslogger, LogType, ROS_Param_Server, set_rospy_log_lvl
from pyaarapsi.core.helper_tools import formatException

'''
Node Name Goes Here

Node description goes here.

'''

class mrc():
    def __init__(self, node_name, rate, namespace, anon, log_level, reset):

        self.node_name      = node_name
        self.namespace      = namespace
        self.nodespace      = self.namespace + "/" + self.node_name
        self.anon           = anon

        self.params         = ROS_Param_Server()
        self.log_level      = self.params.add(self.nodespace + "/log_level", log_level, check_positive_int,     force=reset)
        self.rate_num       = self.params.add(self.nodespace + "/rate",      rate,      check_positive_float,   force=reset)
    
        rospy.init_node(self.node_name, anonymous=self.anon, log_level=self.log_level.get())
        roslogger('Starting [%s] %s node.' % (self.namespace, self.node_name), LogType.INFO, ros=True)
        self.rate_obj       = rospy.Rate(self.rate_num.get())
        self.heartbeat      = Heartbeat(self.node_name, self.namespace, NodeState.INIT, self.rate_num)
        self.param_sub      = rospy.Subscriber(self.namespace + "/params_update", String, self.param_cb, queue_size=100)

    def param_cb(self, msg):
        if self.params.exists(msg.data):
            roslogger("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG, ros=True)
            self.params.update(msg.data)

            if msg.data == self.log_level.name:
                set_rospy_log_lvl(self.log_level.get())
            elif msg.data == self.rate_num.name:
                self.rate_obj = rospy.Rate(self.rate_num.get())
        else:
            roslogger("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG, ros=True)

    def main(self):
        self.heartbeat.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.rate_obj.sleep()


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="file_name_goes_here.py", 
                            description="Node Name Goes Here",
                            epilog="Maintainer: Your Name (your.email@email.com)")
    parser.add_argument('--node-name', '-N', type=check_string,              default="file_name_goes_here", help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate',      '-r', type=check_positive_float,      default=2.0,                   help='Specify node rate (default: %(default)s).')
    parser.add_argument('--anon',      '-a', type=check_bool,                default=False,                 help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace', '-n', type=check_string,              default="/vpr_nodes",          help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2,                     help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset',     '-R', type=check_bool,                default=False,                 help='Force reset of parameters to specified ones (default: %(default)s)')
    
    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])

    node_name   = args['node_name']
    rate        = args['rate']
    namespace   = args['namespace']
    anon        = args['anon']
    log_level   = args['log_level']
    reset       = args['reset']

    try:
        nmrc = mrc(node_name, rate, namespace, anon, log_level, reset)
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