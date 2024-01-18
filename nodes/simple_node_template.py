#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from pyaarapsi.core.ros_tools       import NodeState, roslogger, LogType
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.argparse_tools  import check_enum
from pyaarapsi.vpr_classes.base     import Super_ROS_Class

NODE_NAME = 'node'
NAMESPACE = ''
RATE      = 10.0 # Hz
ANON      = True

'''
Node Name

Node description.

'''

class Main_ROS_Class(Super_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready()

    def main(self):
        # Main loop process
        self.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            try:
                self.loop_contents()
            except rospy.exceptions.ROSInterruptException as e:
                pass
            except Exception as e:
                if self.parameters_ready:
                    raise Exception('Critical failure. ' + formatException()) from e
                else:
                    self.print('Main loop exception, attempting to handle; waiting for parameters to update. Details:\n' + formatException(), LogType.DEBUG, throttle=5)
                    rospy.sleep(0.5)

    def loop_contents(self):
        self.rate_obj.sleep()

def do_args():
    parser = ap.ArgumentParser(prog="file_name.py", 
                            description="Node Name",
                            epilog="Maintainer: Your Name (your.email@email.com)")
    
    # Optional Arguments:
    parser.add_argument('--log-level',   '-V',  type=lambda x: check_enum(x, LogType), default=2,          help="Specify ROS log level (default: %(default)s).")
    
    # Parse args...
    args = vars(parser.parse_known_args()[0])
    args.update({'node_name': NODE_NAME, 'namespace': NAMESPACE, 'rate_num': RATE, 'anon': True, 'reset': True})
    return args

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