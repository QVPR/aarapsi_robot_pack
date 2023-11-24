#!/usr/bin/env python3

import argparse as ap
import sys

from pyaarapsi.core.ros_tools                   import roslogger, LogType
from pyaarapsi.core.helper_tools                import formatException
from pyaarapsi.core.argparse_tools              import check_bool
from pyaarapsi.vpr_classes.base                 import base_optional_args

from pyaarapsi.pathing.simple_follower_base     import Simple_Follower_Class
from pyaarapsi.pathing.extended_follower_base   import Extended_Follower_Class

'''
Path Follower

Uses VPR or SLAM to perform actions such as path following or goal alignment.

'''

def do_args():
    parser = ap.ArgumentParser(prog="path_follower.py", 
                                description="Path Follower",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='path_follower')
    parser.add_argument('--simple', '-s', type=check_bool, default=True, help='Enable simple mode. (default: %(default)s)')

    # Parse args...
    return vars(parser.parse_known_args()[0])

if __name__ == '__main__':
    try:
        args = do_args()
        if args['simple']:
            nmrc = Simple_Follower_Class(**args)
            nmrc.print('Simple mode enabled.', LogType.INFO)
        else:
            nmrc = Extended_Follower_Class(**args)
            nmrc.print('Extended mode enabled.', LogType.INFO)
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