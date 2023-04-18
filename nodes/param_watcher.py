#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from std_msgs.msg import String
from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import Heartbeat, NodeState, roslogger, LogType

'''
ROS Parameter Server Watcher

Our replacement for the mainstream dynamic parameter updates node/s.
This node observes changes to parameters in the /namespace, and then 
reports changes to each parameter on the /namespace/params_update 
topic. 

'''

class mrc():
    def __init__(self, node_name, rate, namespace, anon, log_level):

        self.node_name      = node_name
        self.namespace      = namespace
        self.anon           = anon
        self.log_level      = log_level
        self.rate_num       = rate
    
        rospy.init_node(self.node_name, anonymous=self.anon, log_level=self.log_level)
        roslogger('Starting %s node.' % (self.node_name), LogType.INFO, ros=True)
        self.rate_obj   = rospy.Rate(self.rate_num)
        self.heartbeat  = Heartbeat(self.node_name, self.namespace, NodeState.INIT, self.rate_num)

        self.watch_params   = [i for i in rospy.get_param_names() if i.startswith(namespace)]
        print(self.watch_params)
        self.params_dict    = dict.fromkeys(self.watch_params)

        self.watch_pub      = rospy.Publisher(self.namespace + "/params_update", String, queue_size=100)

    def watch(self):
        for i in list(self.params_dict.keys()):
            check_param = rospy.get_param(i)
            if not self.params_dict[i] == check_param:
                rospy.loginfo("Update detected for: %s (%s->%s)" % (i, self.params_dict[i], check_param))
                self.params_dict[i] = check_param
                self.watch_pub.publish(String(i))

    def main(self):
        self.heartbeat.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.watch()
            self.rate_obj.sleep()


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="param_watcher.py", 
                            description="ROS Parameter Server Watcher",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--node-name', '-N', type=check_string,              default="param_watcher",  help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate', '-r',      type=check_positive_float,      default=2.0,              help='Set node rate (default: %(default)s).')
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
        nmrc = mrc(node_name, rate, namespace, anon, log_level)
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO, ros=True)
        sys.exit()
    except:
        roslogger("Error state reached, system exit triggered.", LogType.INFO, ros=True)