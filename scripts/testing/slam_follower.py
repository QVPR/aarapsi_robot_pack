#!/usr/bin/env python3

import rospy
import rospkg
import argparse as ap
import numpy as np
import sys
import os
import csv
from fastdist import fastdist

from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point, Twist

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, Base_ROS_Class, q_from_yaw, yaw_from_q, xyw_from_pose
from pyaarapsi.core.helper_tools import formatException, Timer

'''
SLAM Path Follower

Node description.

'''

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, node_name, rate_num, namespace, anon, log_level, reset):
        super().__init__(node_name, namespace, rate_num, anon, log_level, order_id=None, throttle=30)

        self.init_params(rate_num, log_level, reset)
        self.init_vars()
        self.init_rospy()

        self.main_ready     = True

    def init_params(self, rate_num, log_level, reset):
        super().init_params(rate_num, log_level, reset)

        self.PATH_FILE      = self.params.add(self.namespace + "/path_file", None, check_string, force=False)

    def init_vars(self):
        super().init_vars()

        self.ego        = []
        self.new_ego    = False
        self.lookahead  = 5

    def init_rospy(self):
        super().init_rospy()

        self.path_pub   = self.add_pub(     self.namespace + '/path',   Path,                   queue_size=1, latch=True)
        self.odom_sub   = rospy.Subscriber(self.ODOM_TOPIC.get(),       Odometry, self.odom_cb, queue_size=1)
        self.twist_pub  = self.add_pub(     '/cmd_vel',                 Twist,                  queue_size=1)

    def normalize_angle(self, angle):
        # Normalize angle [-pi, +pi]
        if angle > np.pi:
            norm_angle = angle - 2*np.pi
        elif angle < -np.pi:
            norm_angle = angle + 2*np.pi
        else:
            norm_angle = angle
        return norm_angle

    def global2local(self):
        Tx  = self.points[:,0] - self.ego[0]
        Ty  = self.points[:,1] - self.ego[1]
        R   = np.sqrt(np.power(Tx, 2) + np.power(Ty, 2))
        A   = np.arctan2(Ty, Tx) - self.ego[2]

        return list(np.multiply(np.cos(A), R)), list(np.multiply(np.sin(A), R))

    def calc_error(self):
        # 1. Global to relative coordinate
        rel_x, rel_y  = self.global2local()

        # 2. Find the nearest waypoint
        distances       = fastdist.matrix_to_matrix_distance(self.points, np.matrix([self.ego[0], self.ego[1], self.ego[2]]), fastdist.euclidean, "euclidean").flatten()
        near_ind        = np.argmin(distances, axis=0)

        # 3. Find target index
        target_ind      = (near_ind + self.lookahead) % self.points.shape[0]

        # 4. Calculate errors
        error_yaw       = np.arctan2(rel_y[target_ind], rel_x[target_ind])
        error_yaw       = self.normalize_angle(error_yaw) # Normalize angle to [-pi, +pi]
        error_y         = rel_y[target_ind]
        return error_y, error_yaw

    def odom_cb(self, msg):
        self.ego        = xyw_from_pose(msg.pose.pose)
        self.new_ego    = True

    def make_path(self):
        self.root           = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/paths'
        with open(self.root + '/' + self.PATH_FILE.get()) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            data = list(reader)
            x_data = data[0]
            y_data = data[1]
            w_data = data[2]
            self.points = np.transpose(np.array([x_data, y_data, w_data]))
            f.close()
        
        self.path = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        for i in self.points:
            new_pose = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id="map"))
            new_pose.pose.position = Point(x=i[0], y=i[1], z=0.0)
            new_pose.pose.orientation = q_from_yaw(i[2])
            self.path.poses.append(new_pose)

        self.path_pub.publish(self.path)

    def main(self):
        self.set_state(NodeState.MAIN)

        self.make_path()

        while not self.new_ego:
            self.rate_obj.sleep()
            self.print('Waiting for start position...')

        self.print('Entering main loop.')

        while not rospy.is_shutdown():
            self.loop_contents()

    def loop_contents(self):

        if not (self.new_ego): # denest
            self.print("Waiting.", LogType.DEBUG, throttle=10) # print every 10 seconds
            rospy.sleep(0.005)
            return
        
        self.rate_obj.sleep()
        self.new_ego  = False

        error_y, error_yaw = self.calc_error()

        new_msg             = Twist()
        new_msg.linear.x    = 0.8
        new_msg.angular.z   = 0.7 * error_y

        self.twist_pub.publish(new_msg)

def do_args():
    parser = ap.ArgumentParser(prog="slam_follower.py", 
                                description="SLAM Path Follower",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="slam_follower",  help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=12,               help='Specify node rate (default: %(default)s).')
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=False,            help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset',            '-R',  type=check_bool,                   default=False,            help='Force reset of parameters to specified ones (default: %(default)s)')

    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = Main_ROS_Class(args['node_name'], args['rate'], args['namespace'], args['anon'], args['log_level'], args['reset'])
        nmrc.print('Initialisation complete.')
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