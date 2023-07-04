#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from std_msgs.msg import String

from pyaarapsi.core.argparse_tools  import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools       import NodeState, roslogger, LogType, set_rospy_log_lvl
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.enum_tools      import enum_value_options
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

'''
ROS Parameter Server Watcher

Our replacement for the mainstream dynamic parameter updates node/s.
This node observes changes to parameters in the /namespace, and then 
reports changes to each parameter on the /namespace/params_update 
topic. 

'''

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_vars(self):
        super().init_vars()
        self.watch_params   = [i for i in rospy.get_param_names() if i.startswith(self.namespace)]
        self.params_dict    = dict.fromkeys(self.watch_params)
        self.print("Watching params: %s" % str(self.watch_params), LogType.DEBUG)

    def init_rospy(self):
        super().init_rospy()
        
        self.watch_pub      = self.add_pub(self.namespace + "/params_update", String, queue_size=100)
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

    def main(self):
        self.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.watch()
            self.rate_obj.sleep()

def do_args():
    parser = ap.ArgumentParser(prog="param_watcher.py", 
                            description="ROS Parameter Server Watcher",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='param_watcher')
    
    # Parse args...
    return vars(parser.parse_known_args()[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = Main_ROS_Class(**args)
        nmrc.print("Initialisation complete.", LogType.INFO)
        nmrc.main()
        nmrc.print("Operation complete.", LogType.INFO, ros=False) # False as rosnode likely terminated
        sys.exit()
    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)