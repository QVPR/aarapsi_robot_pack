#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from std_msgs.msg import String
from pyaarapsi.core.argparse_tools import check_positive_float, check_positive_int, check_bool, check_string
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, set_rospy_log_lvl, init_node
from pyaarapsi.core.helper_tools import formatException

'''
Node Name

Node description.

'''

class mrc():
    def __init__(self, node_name, rate_num, namespace, anon, log_level, reset, order_id=0):

        init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30)

        self.init_params(rate_num, log_level, reset)
        self.init_vars()
        self.init_rospy()

        self.main_ready      = True
        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_params(self, rate_num, log_level, reset):
        self.LOG_LEVEL       = self.ROS_HOME.params.add(self.nodespace + "/log_level",           log_level,              check_positive_int,                             force=reset)
        self.RATE_NUM        = self.ROS_HOME.params.add(self.nodespace + "/rate",                rate_num,               check_positive_float,                           force=reset)

    def init_vars(self):
        pass

    def init_rospy(self):
        self.rate_obj        = rospy.Rate(self.RATE_NUM.get())
        self.param_sub       = rospy.Subscriber(self.namespace + "/params_update", String, self.param_callback, queue_size=100)

    def param_callback(self, msg):
        self.parameters_ready = False
        if self.ROS_HOME.params.exists(msg.data):
            if not self.ROS_HOME.params.update(msg.data):
                self.print("Change to parameter [%s]; bad value." % msg.data, LogType.DEBUG)
        
            else:
                self.print("Change to parameter [%s]; updated." % msg.data, LogType.DEBUG)

                if msg.data == self.LOG_LEVEL.name:
                    set_rospy_log_lvl(self.LOG_LEVEL.get())
                elif msg.data == self.RATE_NUM.name:
                    self.rate_obj = rospy.Rate(self.RATE_NUM.get())
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)
        self.parameters_ready = True

    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.rate_obj.sleep()

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

    def exit(self):
        self.print("Quit received.")
        sys.exit()

def do_args():
    parser = ap.ArgumentParser(prog="file_name.py", 
                            description="Node Name",
                            epilog="Maintainer: Your Name (your.email@email.com)")
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="file_name",      help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=2.0,              help='Specify node rate (default: %(default)s).')
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=False,            help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset',            '-R',  type=check_bool,                   default=False,            help='Force reset of parameters to specified ones (default: %(default)s)')
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')

    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = mrc(args['node_name'], args['rate'], args['namespace'], args['anon'], args['log_level'], args['reset'], order_id=args['order_id'])
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