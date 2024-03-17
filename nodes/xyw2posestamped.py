#!/usr/bin/env python3

import rospy
import argparse as ap
import sys

from geometry_msgs.msg import PoseStamped
from aarapsi_robot_pack.msg import xyw

from pyaarapsi.core.ros_tools       import NodeState, roslogger, LogType, q_from_yaw, import_rosmsg_from_string
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.argparse_tools  import check_enum, check_string
from pyaarapsi.vpr_classes.base     import Super_ROS_Class

NODE_NAME = 'xyw2posestamped'
NAMESPACE = ''
RATE      = 10.0 # Hz
ANON      = True

'''
xyw2posestamped

Republish a aarapsi_robot_pack xyw message as a PoseStamped

'''

def retrieve_level(_object, _steps):
    if not len(_steps):
        return _object
    return retrieve_level(getattr(_object, _steps[0]), _steps[1:])

class Main_ROS_Class(Super_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars(frame_id=kwargs['frame_id'], topic_path=kwargs['topic_path'])
        self.init_rospy(topic_in=kwargs['topic_in'], topic_in_message=kwargs['topic_msg'], topic_out=kwargs['topic_out'])

        self.node_ready()

    def init_vars(self, frame_id: str, topic_path: str):
        super().init_vars()

        self.msg      = None
        self.new_msg  = False
        self.frame_id = frame_id
        self.path     = [] if not len(topic_path) else topic_path.split('.')

    def init_rospy(self, topic_in: str, topic_in_message: str, topic_out: str):
        super().init_rospy()

        self.class_type =  import_rosmsg_from_string(msg_str=topic_in_message)
        if self.path == [] and (self.class_type != xyw): raise Exception('Path is empty, but topic is not message type xyw!')


        self.sub   = rospy.Subscriber(topic_in, self.class_type, self.msg_cb, queue_size=1)
        self.pub   = self.add_pub(topic_out, PoseStamped, queue_size=1)

    def msg_cb(self, msg):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.frame_id

        assert isinstance(msg, self.class_type)
        xyw_msg = retrieve_level(msg, self.path)
        assert isinstance(xyw_msg, xyw)

        pose_msg.pose.position.x = xyw_msg.x
        pose_msg.pose.position.y = xyw_msg.y
        pose_msg.pose.orientation = q_from_yaw(xyw_msg.w)
        self.pub.publish(pose_msg)

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
    parser = ap.ArgumentParser(prog="xyw2posestamped.py", 
                            description="xyw to PoseStamped",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser.add_argument('--log-level',  '-V', type=lambda x: check_enum(x, LogType), default=2,                            help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--topic-in',   '-i', type=check_string,                     default='/xyw',                       help='Specify input topic (default: %(default)s).')
    parser.add_argument('--topic-msg',  '-m', type=check_string,                     default='aarapsi_robot_pack.msg.xyw', help='Specify input topic message (default: %(default)s).')
    parser.add_argument('--topic-path', '-p', type=check_string,                     default='',                           help='Specify input topic path (if message structure is not xyw) (default: %(default)s).')
    parser.add_argument('--topic-out',  '-o', type=check_string,                     default='/pose/out',                  help='Specify output PoseStamped topic (default: %(default)s).')
    parser.add_argument('--frame-id',   '-f', type=check_string,                     default='map',                        help='Specify output PoseStamped frame_id (default: %(default)s).')

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