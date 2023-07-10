#!/usr/bin/env python3

import rospy
import argparse as ap
import numpy as np
import sys
import tf
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from pyaarapsi.core.argparse_tools  import check_positive_float, check_positive_int, check_bool, check_string
from pyaarapsi.core.ros_tools       import NodeState, roslogger, LogType, set_rospy_log_lvl
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.enum_tools      import enum_value_options
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

'''
Frame Transformer

Republishes geometric messages in a new frame of reference.

'''

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_vars(self):
        self.tf_listener     = tf.TransformListener()

    def init_rospy(self):
        self.rate_obj        = rospy.Rate(self.RATE_NUM.get())
        self.param_sub       = rospy.Subscriber(self.namespace + "/params_update",  String,     self.param_callback, queue_size=100)
        self.ekf_sub         = rospy.Subscriber("/odom/slam_ekf",                   Odometry,   self.ekf_callback,   queue_size=1)
        self.gt_odom_pub     = self.add_pub("/odom/true",                  Odometry,                        queue_size=1)
        self.gt_pose_pub     = self.add_pub("/odom/pose",                  PoseStamped,                     queue_size=1)

    def ekf_callback(self, msg):
        odom_pose                   = PoseStamped(pose=msg.pose.pose, header=msg.header)
        map_pose                    = self.tf_listener.transformPose("map", ps=odom_pose)
        rotation_matrix             = self.tf_listener.asMatrix("map", odom_pose.header)[0:3, 0:3]
        vel_3x3                     = np.diag([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        vel_3x3_rotated             = np.dot(rotation_matrix, vel_3x3)
        vel_frame                   = Twist(angular=msg.twist.twist.angular, linear=Vector3(x=vel_3x3_rotated[0,0], y=vel_3x3_rotated[1,1], z=vel_3x3_rotated[2,2]))
        gt_odom_msg                 = Odometry() # grab covariances and velocity information from here
        gt_odom_msg.pose.pose       = map_pose.pose
        gt_odom_msg.twist.twist     = vel_frame
        gt_odom_msg.header.frame_id = "map"
        gt_odom_msg.header.stamp    = rospy.Time.now()
        self.gt_odom_pub.publish(gt_odom_msg)
        self.gt_pose_pub.publish(map_pose)

    
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
    parser = ap.ArgumentParser(prog="frame_transformer.py", 
                            description="Frame Transformer",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='frame_transformer')

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