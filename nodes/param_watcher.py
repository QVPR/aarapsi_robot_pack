#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from std_msgs.msg import String
from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, ROS_Home
from pyaarapsi.core.helper_tools import formatException

'''
ROS Parameter Server Watcher

Our replacement for the mainstream dynamic parameter updates node/s.
This node observes changes to parameters in the /namespace, and then 
reports changes to each parameter on the /namespace/params_update 
topic. 

'''

class mrc():
    def __init__(self, node_name, rate, namespace, anon, log_level, reset=True):

        self.node_name      = node_name
        self.namespace      = namespace
        self.nodespace      = self.namespace + "/" + self.node_name

        rospy.init_node(self.node_name, anonymous=anon, log_level=log_level)
        self.ROS_HOME       = ROS_Home(self.node_name, self.namespace, rate)
        self.print('Starting %s node.' % (self.node_name))
        
        self.init_params(rate, reset)
        self.init_vars()
        self.init_rospy()

    def init_params(self, rate, reset):
        self.rate_num       = self.ROS_HOME.params.add(self.nodespace + "/rate",      rate,      check_positive_float,   force=reset)

    def init_vars(self):
        self.watch_params   = [i for i in rospy.get_param_names() if i.startswith(self.namespace)]
        self.params_dict    = dict.fromkeys(self.watch_params)
        self.print("Watching params: %s" % str(self.watch_params), LogType.DEBUG)

    def init_rospy(self):
        self.rate_obj       = rospy.Rate(self.rate_num.get())
        self.watch_pub      = self.ROS_HOME.add_pub(self.namespace + "/params_update", String, queue_size=100)
        self.watch_timer    = rospy.Timer(rospy.Duration(secs=5), self.watch_cb)

    def watch_cb(self, event):
        new_keys = []
        new_params_list = [i for i in rospy.get_param_names() if i.startswith(namespace)]
        for key in new_params_list:
            if not key in self.watch_params:
                new_keys.append(key)
                self.watch_params.append(key)
                self.params_dict[key] = rospy.get_param(key)
        if len(new_keys):
            self.print('New params: %s' % str(new_keys))

    def watch(self):
        for i in list(self.params_dict.keys()):
            check_param = rospy.get_param(i)
            if not self.params_dict[i] == check_param:
                self.print("Update detected for: %s (%s->%s)" % (i, str(self.params_dict[i]), str(check_param)))
                self.params_dict[i] = check_param
                self.watch_pub.publish(String(i))

    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.watch()
            self.rate_obj.sleep()

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)


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
        roslogger("Operation complete.", LogType.INFO, ros=False) # False as rosnode likely terminated
        sys.exit()
    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)