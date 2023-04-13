#!/usr/bin/env python3

import rospy
import copy
import numpy as np
import argparse as ap

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseArray, Pose, Point
from std_msgs.msg import Header

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import yaw_from_q, q_from_yaw        

class mrc:
    def __init__(self, node_name, rate, anon, log_level):
        
        rospy.init_node(node_name, anonymous=anon, log_level=1)
        rospy.loginfo('Starting %s node.' % (node_name))

        self.rate_num       = rate
        self.dt             = 1/rate
        self.rate_obj       = rospy.Rate(self.rate_num)

        self.odom           = Odometry()
        self.cmd            = Twist()
        self.new_odom       = False
        self.new_cmd        = False

        self.max_count      = 40
        self.num_keyposes   = 5
        self.fit_state      = np.zeros((self.max_count, 5), dtype=np.float64) # x, y, w, vx, vw
        self.raw_state      = np.zeros((self.max_count, 5), dtype=np.float64)
        self.state_count    = 0

        self.time           = rospy.Time.now().to_sec()

        self.odom_sub       = rospy.Subscriber('/odom/in',    Odometry,  self.odom_cb, queue_size=1)
        self.cmd_sub        = rospy.Subscriber('/cmd_vel/in', Twist,     self.cmd_cb,  queue_size=1)
        self.odom_pub       = rospy.Publisher('/odom/out',    Odometry,                queue_size=1)

        self.fit_state_pub  = rospy.Publisher('/state/fit',   PoseArray,               queue_size=1)
        self.raw_state_pub  = rospy.Publisher('/state/raw',   PoseArray,               queue_size=1)

        self.ready          = True

        self.main()

    def odom_cb(self, msg):
        if not self.ready:
            return
        self.odom     = msg
        self.new_odom = True

    def cmd_cb(self, msg):
        if not self.ready:
            return
        # arrives in 'odom' frame i.e. x-axis points in direction of travel
        self.cmd      = msg
        self.new_cmd  = True

    def main(self):
        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            self.update()
            self.integrate()
            self.fit()
            self.do_pub()
            self.step()
            
        rospy.loginfo("Exit state reached.")
        
    def update(self):
        if not self.new_odom:
            return # denest
        self.new_odom = False

        # if new Odometry message (for position):
        self.raw_state[0,0] = self.odom.pose.pose.position.x
        self.raw_state[0,1] = self.odom.pose.pose.position.y
        self.raw_state[0,2] = yaw_from_q(self.odom.pose.pose.orientation)

        if self.new_cmd:
            # if new Twist message (for velocity):
            self.fit_state[0,3] = self.cmd.linear.x 
            self.fit_state[0,4] = self.cmd.angular.z
            self.raw_state[0,3] = self.cmd.linear.x 
            self.raw_state[0,4] = self.cmd.angular.z
            self.new_cmd = False
        else:
            self.fit_state[0,3] = self.fit_state[1,3]
            self.fit_state[0,4] = self.fit_state[1,4]
            self.raw_state[0,3] = self.raw_state[1,3]
            self.raw_state[0,4] = self.raw_state[1,4]

    def integrate(self):
        self.fit_state[0,0] = self.fit_state[0,0] + (self.fit_state[0,3] * np.cos(self.fit_state[0,2]) * self.dt)
        self.fit_state[0,1] = self.fit_state[0,1] + (self.fit_state[0,3] * np.sin(self.fit_state[0,2]) * self.dt)
        self.fit_state[0,2] = self.fit_state[0,2] + (self.fit_state[0,4] * self.dt)

    def fit(self):
        pass

    def step(self):
        self.fit_state    = np.roll(self.fit_state, 1, axis=0)
        self.raw_state    = np.roll(self.raw_state, 1, axis=0)
        if self.state_count < self.max_count - 1:
            self.state_count += 1

    def make_pose_array(self, array_in, num_measurements):
        msg = PoseArray(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        if num_measurements <= self.num_keyposes:
            indices = list(range(num_measurements))
        else:
            _indices = list(range(num_measurements))
            indices = [_indices[int(k*len(_indices)/(self.num_keyposes-1))] for k in range((self.num_keyposes-1))]
            indices.append(_indices[-1])
            indices.reverse()
        for i in indices:
            msg.poses.append(Pose(position=Point(x=array_in[i,0], y=array_in[i,1], z=0), orientation=q_from_yaw(array_in[i,2])))
        return msg

    def do_pub(self):
        self.fit_state_pub.publish(self.make_pose_array(self.fit_state, self.state_count))
        self.raw_state_pub.publish(self.make_pose_array(self.raw_state, self.state_count))

if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(prog="odometry smoother", 
                                description="ROS Odometry Smoother Tool",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
        parser.add_argument('--node-name', '-N', type=check_string,              default="smooth",                              help="Specify node name (default: %(default)s).")
        parser.add_argument('--rate',      '-r', type=check_positive_float,      default=10.0,                                  help='Set node rate (default: %(default)s).')
        parser.add_argument('--anon',      '-a', type=check_bool,                default=False,                                 help="Specify whether node should be anonymous (default: %(default)s).")
        parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2,                                     help="Specify ROS log level (default: %(default)s).")
        
        raw_args = parser.parse_known_args()
        args = vars(raw_args[0])

        node_name   = args['node_name']
        rate        = args['rate']
        anon        = args['anon']
        log_level   = args['log_level']

        nmrc = mrc(node_name, rate, anon, log_level)
    except rospy.ROSInterruptException:
        pass