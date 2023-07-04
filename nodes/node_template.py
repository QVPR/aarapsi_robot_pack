#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from pyaarapsi.core.ros_tools       import NodeState, roslogger, LogType
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

'''
Node Name

Node description.

'''

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_params(self, rate_num, log_level, reset):
        super().init_params(rate_num, log_level, reset)

    def init_vars(self):
        super().init_vars()

    def init_rospy(self):
        super().init_rospy()

    def main(self):
        self.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.loop_contents()

    def loop_contents(self):
        self.rate_obj.sleep()

def do_args():
    parser = ap.ArgumentParser(prog="file_name.py", 
                            description="Node Name",
                            epilog="Maintainer: Your Name (your.email@email.com)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='file_name', rate=10.0)
    
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