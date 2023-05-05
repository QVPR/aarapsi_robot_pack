#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from std_msgs.msg import String
from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string, check_positive_int
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, init_node, set_rospy_log_lvl
from pyaarapsi.core.helper_tools import formatException

'''
ROS Parameter Server Watcher

Our replacement for the mainstream dynamic parameter updates node/s.
This node observes changes to parameters in the /namespace, and then 
reports changes to each parameter on the /namespace/params_update 
topic. 

'''

class mrc():
    def __init__(self, node_name, rate_num, namespace, anon, log_level, reset=True, order_id=0):

        init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30)
        
        self.init_params(rate_num, log_level, reset)
        self.init_vars()
        self.init_rospy()

        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_params(self, rate, log_level, reset):
        self.rate_num       = self.ROS_HOME.params.add(self.nodespace + "/rate",      rate,      check_positive_float,   force=reset)
        self.log_level      = self.ROS_HOME.params.add(self.nodespace + "/log_level", log_level, check_positive_int,     force=reset)

    def init_vars(self):
        self.watch_params   = [i for i in rospy.get_param_names() if i.startswith(self.namespace)]
        self.params_dict    = dict.fromkeys(self.watch_params)
        self.print("Watching params: %s" % str(self.watch_params), LogType.DEBUG)

    def init_rospy(self):
        self.rate_obj       = rospy.Rate(self.rate_num.get())
        self.param_sub      = rospy.Subscriber(self.namespace + "/params_update", String, self.param_cb, queue_size=100)
        self.watch_pub      = self.ROS_HOME.add_pub(self.namespace + "/params_update", String, queue_size=100)
        self.watch_timer    = rospy.Timer(rospy.Duration(secs=5), self.watch_cb)

    def watch_cb(self, event):
        new_keys = []
        new_params_list = [i for i in rospy.get_param_names() if i.startswith(self.namespace)]
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
                self.print("Update detected for: %s (%s->%s)" % (i, str(self.params_dict[i]), str(check_param)), LogType.DEBUG)
                self.params_dict[i] = check_param
                self.watch_pub.publish(String(i))

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

def do_args():
    parser = ap.ArgumentParser(prog="param_watcher.py", 
                            description="ROS Parameter Server Watcher",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="param_watcher",  help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=2.0,              help='Set node rate (default: %(default)s).')
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=False,            help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')
    
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = mrc(args['node_name'], args['rate'], args['namespace'], args['anon'], args['log_level'], order_id=args['order_id'])
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