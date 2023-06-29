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
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point, Twist, Vector3, Quaternion
from visualization_msgs.msg import MarkerArray, Marker

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string, check_float_list
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, Base_ROS_Class, q_from_yaw, yaw_from_q, pose2xyw
from pyaarapsi.core.helper_tools import formatException, Timer, angle_wrap, normalize_angle

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

        self.USE_NOISE  = self.params.add(self.nodespace + "/noise/enable", False,      check_bool,                          force=True)
        self.NOISE_VALS = self.params.add(self.nodespace + "/noise/vals",   [0.1]*3,    lambda x: check_float_list(x, 3),    force=True)
        self.REVERSE    = self.params.add(self.nodespace + "/reverse",      False,      check_bool,                          force=False)
        self.PATH_FILE  = self.params.add(self.namespace + "/path_file",    None,       check_string,                        force=False)

    def init_vars(self):
        super().init_vars()

        self.ego        = []
        self.new_ego    = False
        self.lookahead  = 5

    def init_rospy(self):
        super().init_rospy()

        self.path_pub   = self.add_pub(     self.namespace + '/path',       Path,                   queue_size=1, latch=True)
        self.goal_pub   = self.add_pub(     self.namespace + '/path_goal',  PoseStamped,            queue_size=1)
        self.speed_pub  = self.add_pub(     self.namespace + '/speeds',     MarkerArray,            queue_size=1, latch=True)
        self.odom_sub   = rospy.Subscriber(self.ODOM_TOPIC.get(),           Odometry, self.odom_cb, queue_size=1)
        self.twist_pub  = self.add_pub(     '/cmd_vel',                     Twist,                  queue_size=1)

    def global2local(self):
        Tx  = self.points[:,0] - self.ego[0]
        Ty  = self.points[:,1] - self.ego[1]
        R   = np.sqrt(np.power(Tx, 2) + np.power(Ty, 2))
        A   = np.arctan2(Ty, Tx) - self.ego[2]

        return list(np.multiply(np.cos(A), R)), list(np.multiply(np.sin(A), R))

    def calc_error(self):
        # 1. Global to relative coordinate
        rel_x, rel_y    = self.global2local()

        # 2. Find the nearest waypoint
        distances       = fastdist.matrix_to_matrix_distance(self.points[:,0:3], np.matrix([self.ego[0], self.ego[1], self.ego[2]]), fastdist.euclidean, "euclidean").flatten()
        near_ind        = np.argmin(distances, axis=0)

        # 3. Find target index
        target_ind      = (near_ind + self.lookahead) % self.points.shape[0]

        # 4. Calculate errors
        error_yaw       = np.arctan2(rel_y[target_ind], rel_x[target_ind])
        error_yaw       = normalize_angle(error_yaw) # Normalize angle to [-pi, +pi]
        error_y         = rel_y[target_ind]
        return error_y, error_yaw, near_ind

    def odom_cb(self, msg):
        self.ego        = pose2xyw(msg.pose.pose)
        if self.USE_NOISE.get():
            self.ego   += np.random.rand(3) * np.array(self.NOISE_VALS.get())
        self.new_ego    = True

    def make_path(self):
        self.root           = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/paths'
        with open(self.root + '/' + self.PATH_FILE.get()) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            data = list(reader)
            x_data = data[0]
            y_data = data[1]
            w_data = data[2]
            if self.REVERSE.get():
                w_data = list(angle_wrap(np.pi + np.array(w_data), mode='RAD'))
            self.points = np.transpose(np.array([x_data, y_data, w_data]))
            f.close()

        # Generate speed profile based on curvature of track:
        points_diff = np.abs(angle_wrap(np.roll(self.points[:,2], 1, 0) - np.roll(self.points[:,2], -1, 0), mode='RAD'))
        k_rad = 5
        points_smooth = np.sum([np.roll(points_diff, i, 0) for i in np.arange(2*k_rad + 1)-k_rad], axis=0)
        speeds = (1 - ((points_smooth - np.min(points_smooth)) / (np.max(points_smooth) - np.min(points_smooth)))) **2

        self.points = np.concatenate([self.points, speeds[:, np.newaxis]], axis=1)

        if self.REVERSE.get():
            self.points = np.flipud(self.points)

        self.path       = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        self.speeds     = MarkerArray()
        _num = self.points.shape[0]
        for i in range(_num):
            new_pose = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id="map"))

            new_pose.pose.position = Point(x=self.points[i,0], y=self.points[i,1], z=0.0)
            

            new_marker                      = Marker(header=Header(stamp=rospy.Time.now(), frame_id='map'))
            new_marker.type                 = new_marker.ARROW
            new_marker.action               = new_marker.ADD
            new_marker.id                   = i
            new_marker.color                = ColorRGBA(r=0.859, b=0.220, g=0.094, a=0.5)
            new_marker.scale                = Vector3(x=self.points[i,3], y=0.05, z=0.05)

            new_marker.pose.position        = Point(x=self.points[i,0], y=self.points[i,1], z=0.0)
            if not self.REVERSE.get():
                yaw = q_from_yaw(self.points[i,2] + np.pi/2)
                
            else:
                yaw = q_from_yaw(self.points[i,2] - np.pi/2)

            new_pose.pose.orientation       = q_from_yaw(self.points[i,2])
            new_marker.pose.orientation     = yaw

            self.path.poses.append(new_pose)
            self.speeds.markers.append(new_marker)

        self.path_pub.publish(self.path)
        self.speed_pub.publish(self.speeds)

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

        error_y, error_yaw, ind = self.calc_error()

        goal                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        goal.pose.position      = Point(x=self.points[ind,0], y=self.points[ind,1], z=0.0)
        goal.pose.orientation   = q_from_yaw(self.points[ind,2])

        new_msg             = Twist()
        new_msg.linear.x    = 0.5 * self.points[ind, 3] + 0.3
        new_msg.angular.z   = 0.7 * error_y

        self.twist_pub.publish(new_msg)
        self.goal_pub.publish(goal)

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