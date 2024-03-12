#!/usr/bin/env python3

import rospy
import argparse as ap
import sys

from geometry_msgs.msg import PoseStamped
from aarapsi_robot_pack.msg import Label

from pyaarapsi.core.ros_tools       import NodeState, roslogger, LogType, q_from_yaw
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.argparse_tools  import check_enum, check_string
from pyaarapsi.vpr_classes.base     import Super_ROS_Class

NODE_NAME = 'labelxyw2posestamped'
NAMESPACE = ''
RATE      = 10.0 # Hz
ANON      = True

'''
labelxyw2posestamped

Republish a label messages's gt_ego xyw aarapsi_robot_pack message as a posestamped

'''

class Main_ROS_Class(Super_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars(frame_id=kwargs['frame_id'])
        self.init_rospy(label_topic=kwargs['topic_in'], pose_topic=kwargs['topic_out'])

        self.node_ready()

    def init_vars(self, frame_id: str):
        super().init_vars()

        self.label_msg      = Label()
        self.new_label_msg  = False
        self.frame_id       = frame_id

    def init_rospy(self, label_topic: str, pose_topic: str):
        super().init_rospy()

        self.xyw_sub    = rospy.Subscriber(label_topic, Label, self.label_cb, queue_size=1)
        self.pose_pub   = self.add_pub(pose_topic, PoseStamped, queue_size=1)

    def label_cb(self, msg: Label):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = msg.gt_ego.x
        pose_msg.pose.position.y = msg.gt_ego.y
        pose_msg.pose.orientation = q_from_yaw(msg.gt_ego.w)
        self.pose_pub.publish(pose_msg)

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
    parser = ap.ArgumentParser(prog="labelxyw2posestamped.py", 
                            description="Label's gt_ego (xyw) to PoseStamped",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser.add_argument('--log-level',   '-V',  type=lambda x: check_enum(x, LogType), default=2,           help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--topic-in',    '-ti', type=check_string,                     default='/label/in', help='Specify input label topic (default: %(default)s).')
    parser.add_argument('--topic-out',   '-to', type=check_string,                     default='/pose/out', help='Specify output PoseStamped topic (default: %(default)s).')
    parser.add_argument('--frame-id',     '-f', type=check_string,                     default='map',       help='Specify output PoseStamped frame_id (default: %(default)s).')

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