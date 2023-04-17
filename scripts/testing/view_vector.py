#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from tf.transformations import euler_from_quaternion

import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import warnings
import math

from pyaarapsi.core.argparse_tools import check_positive_float, check_string, check_bool
from pyaarapsi.core.ros_tools import yaw_from_q

class Viewer:
    def __init__(self, node_name, anon, log_level, rate, mode, topic=None):
        rospy.init_node(node_name, anonymous=anon, log_level=log_level)
        rospy.loginfo('Starting %s node.' % (node_name))
        rospy.logdebug("PARAMETERS:\n\t\t\t\tNode Name: %s\n\t\t\t\tMode: %s\n\t\t\t\tAnonymous: %s\n\t\t\t\tLogging Level: %s\n\t\t\t\tRate: %s\n\t\t\t\tTopic: %s" \
                   % (str(node_name), str(mode), str(anon), str(log_level), str(rate), str(topic)))
            
        self.node_name  = node_name
        self.anon       = anon
        self.log_level  = log_level
        self.rate_num   = rate
        self.rate_obj   = rospy.Rate(self.rate_num)
        self.mode       = mode
        self.topic      = topic

        self.msg        = None
        self.gtmsg      = None

        self.new_msg    = False
        self.gtnew_msg  = False

    def topic_cb(self, msg):
        self.msg        = msg
        self.new_msg    = True

    def gt_topic_cb(self, msg):
        self.gtmsg      = msg
        self.gtnew_msg  = True

    def main(self):
        if self.mode == "odom":
            self.main_odometry()
        elif self.mode == "imu": 
            self.main_imu()
        elif self.mode == 'dOdom':
            self.main_odometry_derivative()
        elif self.mode == 'vec3s':
            self.main_vector3stamped()

    def main_imu(self):

        if self.topic is None:
            self.topic_to_view = "/imu/data"
        else:
            self.topic_to_view = self.topic
        
        self.sub = rospy.Subscriber(self.topic_to_view, Imu, self.topic_cb, queue_size=1)

        self.fig, self.axes = plt.subplots(1,1)
        self.fig.show()
        self.axes_twins = [[self.axes, self.axes.twinx()]]

        self.msg        = None
        self.new_msg    = False

        data_dict   = {'yaw': [], 'vyaw': []}
        _MIN = 10000
        _MAX = -10000
        mins_dict = {'yaw': _MIN, 'vyaw': _MIN}
        maxs_dict = {'yaw': _MAX, 'vyaw': _MAX}
        hist_len    = 100
        while not rospy.is_shutdown():
            self.rate_obj.sleep() # reduce cpu load
            if not self.new_msg:
                continue #denest
            self.new_msg = False

            ## todo each loop:
            # get new data:
            new__yaw = round(euler_from_quaternion([float(self.msg.orientation.x), \
                                                    float(self.msg.orientation.y), \
                                                    float(self.msg.orientation.z), \
                                                    float(self.msg.orientation.w)])[2], 3)
            new_vyaw = round(self.msg.angular_velocity.z,3)

            # store new data:
            data_dict['yaw'].append(new__yaw)
            data_dict['vyaw'].append(new_vyaw)

            # crunch plot limits:
            mins_dict = {key: data_dict[key][-1] if data_dict[key][-1] < mins_dict[key] else mins_dict[key] for key in list(mins_dict.keys())}
            maxs_dict = {key: data_dict[key][-1] if data_dict[key][-1] > maxs_dict[key] else maxs_dict[key] for key in list(maxs_dict.keys())}

            # clean axes:
            length_arr = len(data_dict['yaw'])
            while length_arr > hist_len:
                [data_dict[key].pop(0) for key in list(data_dict.keys())]
                length_arr -= 1
            spacing = np.arange(length_arr)
            [[j.clear() for j in i] for i in self.axes_twins] # clear old data from axes

            if length_arr < 10:
                continue

            # plot data:
            self.axes_twins[0][0].plot(spacing, data_dict['yaw'], 'r')
            self.axes_twins[0][1].plot(spacing, data_dict['vyaw'], 'b')

            # update plot limits:
            self.axes_twins[0][0].set_ylim(mins_dict['yaw'],    maxs_dict['yaw'])
            self.axes_twins[0][1].set_ylim(mins_dict['vyaw'],   maxs_dict['vyaw'])

            # draw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.001)

    def main_odometry_derivative(self):
        
        if self.topic is None:
            self.topic_to_view = "/odometry/filtered"
        else:
            self.topic_to_view = self.topic
        
        self.sub = rospy.Subscriber(self.topic_to_view, Odometry, self.topic_cb, queue_size=1)

        self.fig, self.axes = plt.subplots(3,1)
        self.fig.show()
        self.axes_twins = [[i, i.twinx()] for i in self.axes]

        self.msg        = None
        self.omsg    = None
        self.new_msg    = False
        self.old_msg    = False

        data_dict   = {'x': [], 'y': [], 'yaw': [], 'vx': [], 'vy': [], 'vyaw': []}
        _MIN = 10000
        _MAX = -10000
        mins_dict = {'x': _MIN, 'y': _MIN, 'yaw': _MIN, 'vx': _MIN, 'vy': _MIN, 'vyaw': _MIN}
        maxs_dict = {'x': _MAX, 'y': _MAX, 'yaw': _MAX, 'vx': _MAX, 'vy': _MAX, 'vyaw': _MAX}
        hist_len    = 100

        while not rospy.is_shutdown():
            self.rate_obj.sleep() # reduce cpu load
            if not (self.new_msg and self.old_msg):
                if self.new_msg:
                    self.omsg = self.msg
                    self.old_msg = True
                continue #denest
            self.new_msg = False

            ## todo each loop:
            # get new data:
            new____x = round(self.msg.pose.pose.position.x,3)
            new____y = round(self.msg.pose.pose.position.y,3)
            new__yaw = round(yaw_from_q(self.msg.pose.pose.orientation), 3)
            new___vx = round(self.msg.pose.pose.position.x-self.omsg.pose.pose.position.x,3)
            new___vy = round(self.msg.pose.pose.position.y-self.omsg.pose.pose.position.y,3)
            new_vyaw = round(yaw_from_q(self.msg.pose.pose.orientation) - yaw_from_q(self.omsg.pose.pose.orientation),3)
            self.omsg = self.msg

            # store new data:
            data_dict['x'].append(new____x)
            data_dict['y'].append(new____y)
            data_dict['yaw'].append(new__yaw)
            data_dict['vx'].append(new___vx)
            data_dict['vy'].append(new___vy)
            data_dict['vyaw'].append(new_vyaw)

            # crunch plot limits:
            mins_dict = {key: data_dict[key][-1] if data_dict[key][-1] < mins_dict[key] else mins_dict[key] for key in list(mins_dict.keys())}
            maxs_dict = {key: data_dict[key][-1] if data_dict[key][-1] > maxs_dict[key] else maxs_dict[key] for key in list(maxs_dict.keys())}

            # clean axes:
            length_arr = len(data_dict['x'])
            while length_arr > hist_len:
                [data_dict[key].pop(0) for key in list(data_dict.keys())]
                length_arr -= 1
            spacing = np.arange(length_arr)
            [[j.clear() for j in i] for i in self.axes_twins] # clear old data from axes

            if length_arr < 10:
                continue

            # plot data:
            self.axes_twins[0][0].plot(spacing, data_dict['x'], 'r')
            self.axes_twins[0][1].plot(spacing, data_dict['vx'], 'b')
            self.axes_twins[1][0].plot(spacing, data_dict['y'], 'r')
            self.axes_twins[1][1].plot(spacing, data_dict['vy'], 'b')
            self.axes_twins[2][0].plot(spacing, data_dict['yaw'], 'r')
            self.axes_twins[2][1].plot(spacing, data_dict['vyaw'], 'b')

            # update plot limits:
            E = 0.2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.axes_twins[0][0].set_ylim(mins_dict['x']    - (maxs_dict['x']    - mins_dict['x'])*E,    maxs_dict['x']    + (maxs_dict['x']    - mins_dict['x'])*E)
                self.axes_twins[0][1].set_ylim(mins_dict['vx']   - (maxs_dict['vx']   - mins_dict['vx'])*E,   maxs_dict['vx']   + (maxs_dict['vx']   - mins_dict['vx'])*E)
                self.axes_twins[1][0].set_ylim(mins_dict['y']    - (maxs_dict['y']    - mins_dict['y'])*E,    maxs_dict['y']    + (maxs_dict['y']    - mins_dict['y'])*E)
                self.axes_twins[1][1].set_ylim(mins_dict['vy']   - (maxs_dict['vy']   - mins_dict['vy'])*E,   maxs_dict['vy']   + (maxs_dict['vy']   - mins_dict['vy'])*E)
                self.axes_twins[2][0].set_ylim(mins_dict['yaw']  - (maxs_dict['yaw']  - mins_dict['yaw'])*E,  maxs_dict['yaw']  + (maxs_dict['yaw']  - mins_dict['yaw'])*E)
                self.axes_twins[2][1].set_ylim(mins_dict['vyaw'] - (maxs_dict['vyaw'] - mins_dict['vyaw'])*E, maxs_dict['vyaw'] + (maxs_dict['vyaw'] - mins_dict['vyaw'])*E)
            
            # disable ticks and labels on x axis
            self.axes_twins[0][0].set_xticks([])
            self.axes_twins[0][0].set_xticklabels([])
            self.axes_twins[0][0].set_ylabel('x   ', rotation=0)
            self.axes_twins[0][1].set_xticks([])
            self.axes_twins[0][1].set_xticklabels([])
            self.axes_twins[0][1].set_ylabel('   vx', rotation=0)
            self.axes_twins[1][0].set_xticks([])
            self.axes_twins[1][0].set_xticklabels([])
            self.axes_twins[1][0].set_ylabel('y   ', rotation=0)
            self.axes_twins[1][1].set_xticks([])
            self.axes_twins[1][1].set_xticklabels([])
            self.axes_twins[1][1].set_ylabel('   vy', rotation=0)
            self.axes_twins[2][0].set_xticks([])
            self.axes_twins[2][0].set_xticklabels([])
            self.axes_twins[2][0].set_ylabel('w   ', rotation=0)
            self.axes_twins[2][1].set_xticks([])
            self.axes_twins[2][1].set_xticklabels([])
            self.axes_twins[2][1].set_ylabel('   vw', rotation=0)
            self.fig.subplots_adjust(left=0.15, right=0.85)

            # draw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.001)

    def main_odometry(self):
        
        if self.topic is None:
            self.topic_to_view = "/odometry/filtered"
        else:
            self.topic_to_view = self.topic
        
        self.sub = rospy.Subscriber(self.topic_to_view, Odometry, self.topic_cb, queue_size=1)

        self.fig, self.axes = plt.subplots(3,1)
        self.fig.show()
        self.axes_twins = [[i, i.twinx()] for i in self.axes]

        self.msg        = None
        self.new_msg    = False

        data_dict   = {'x': [], 'y': [], 'yaw': [], 'vx': [], 'vy': [], 'vyaw': []}
        _MIN = 10000
        _MAX = -10000
        mins_dict = {'x': _MIN, 'y': _MIN, 'yaw': _MIN, 'vx': _MIN, 'vy': _MIN, 'vyaw': _MIN}
        maxs_dict = {'x': _MAX, 'y': _MAX, 'yaw': _MAX, 'vx': _MAX, 'vy': _MAX, 'vyaw': _MAX}
        hist_len    = 100
        while not rospy.is_shutdown():
            self.rate_obj.sleep() # reduce cpu load
            if not self.new_msg:
                continue #denest
            self.new_msg = False

            ## todo each loop:
            # get new data:
            new____x = round(self.msg.pose.pose.position.x,3)
            new____y = round(self.msg.pose.pose.position.y,3)
            new__yaw = round(euler_from_quaternion([float(self.msg.pose.pose.orientation.x), \
                                                    float(self.msg.pose.pose.orientation.y), \
                                                    float(self.msg.pose.pose.orientation.z), \
                                                    float(self.msg.pose.pose.orientation.w)])[2], 3)
            new___vx = round(self.msg.twist.twist.linear.x,3)
            new___vy = round(self.msg.twist.twist.linear.y,3)
            new_vyaw = round(self.msg.twist.twist.angular.z,3)

            # store new data:
            data_dict['x'].append(new____x)
            data_dict['y'].append(new____y)
            data_dict['yaw'].append(new__yaw)
            data_dict['vx'].append(new___vx)
            data_dict['vy'].append(new___vy)
            data_dict['vyaw'].append(new_vyaw)

            # crunch plot limits:
            mins_dict = {key: data_dict[key][-1] if data_dict[key][-1] < mins_dict[key] else mins_dict[key] for key in list(mins_dict.keys())}
            maxs_dict = {key: data_dict[key][-1] if data_dict[key][-1] > maxs_dict[key] else maxs_dict[key] for key in list(maxs_dict.keys())}

            # clean axes:
            length_arr = len(data_dict['x'])
            while length_arr > hist_len:
                [data_dict[key].pop(0) for key in list(data_dict.keys())]
                length_arr -= 1
            spacing = np.arange(length_arr)
            [[j.clear() for j in i] for i in self.axes_twins] # clear old data from axes

            if length_arr < 10:
                continue

            # plot data:
            self.axes_twins[0][0].plot(spacing, data_dict['x'], 'r')
            self.axes_twins[0][1].plot(spacing, data_dict['vx'], 'b')
            self.axes_twins[1][0].plot(spacing, data_dict['y'], 'r')
            self.axes_twins[1][1].plot(spacing, data_dict['vy'], 'b')
            self.axes_twins[2][0].plot(spacing, data_dict['yaw'], 'r')
            self.axes_twins[2][1].plot(spacing, data_dict['vyaw'], 'b')

            # update plot limits:
            E = 0.2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.axes_twins[0][0].set_ylim(mins_dict['x']    - (maxs_dict['x']    - mins_dict['x'])*E,    maxs_dict['x']    + (maxs_dict['x']    - mins_dict['x'])*E)
                self.axes_twins[0][1].set_ylim(mins_dict['vx']   - (maxs_dict['vx']   - mins_dict['vx'])*E,   maxs_dict['vx']   + (maxs_dict['vx']   - mins_dict['vx'])*E)
                self.axes_twins[1][0].set_ylim(mins_dict['y']    - (maxs_dict['y']    - mins_dict['y'])*E,    maxs_dict['y']    + (maxs_dict['y']    - mins_dict['y'])*E)
                self.axes_twins[1][1].set_ylim(mins_dict['vy']   - (maxs_dict['vy']   - mins_dict['vy'])*E,   maxs_dict['vy']   + (maxs_dict['vy']   - mins_dict['vy'])*E)
                self.axes_twins[2][0].set_ylim(mins_dict['yaw']  - (maxs_dict['yaw']  - mins_dict['yaw'])*E,  maxs_dict['yaw']  + (maxs_dict['yaw']  - mins_dict['yaw'])*E)
                self.axes_twins[2][1].set_ylim(mins_dict['vyaw'] - (maxs_dict['vyaw'] - mins_dict['vyaw'])*E, maxs_dict['vyaw'] + (maxs_dict['vyaw'] - mins_dict['vyaw'])*E)
            
            # disable ticks and labels on x axis
            self.axes_twins[0][0].set_xticks([])
            self.axes_twins[0][0].set_xticklabels([])
            self.axes_twins[0][0].set_ylabel('x   ', rotation=0)
            self.axes_twins[0][1].set_xticks([])
            self.axes_twins[0][1].set_xticklabels([])
            self.axes_twins[0][1].set_ylabel('   vx', rotation=0)
            self.axes_twins[1][0].set_xticks([])
            self.axes_twins[1][0].set_xticklabels([])
            self.axes_twins[1][0].set_ylabel('y   ', rotation=0)
            self.axes_twins[1][1].set_xticks([])
            self.axes_twins[1][1].set_xticklabels([])
            self.axes_twins[1][1].set_ylabel('   vy', rotation=0)
            self.axes_twins[2][0].set_xticks([])
            self.axes_twins[2][0].set_xticklabels([])
            self.axes_twins[2][0].set_ylabel('w   ', rotation=0)
            self.axes_twins[2][1].set_xticks([])
            self.axes_twins[2][1].set_xticklabels([])
            self.axes_twins[2][1].set_ylabel('   vw', rotation=0)
            self.fig.subplots_adjust(left=0.15, right=0.85)

            # draw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.001)

    def main_vector3stamped(self):
        
        if self.topic is None:
            self.topic_to_view = "/imu_um7/mag"
        else:
            self.topic_to_view = self.topic
        
        self.sub = rospy.Subscriber(self.topic_to_view, Vector3Stamped, self.topic_cb, queue_size=1)
        self.subgt = rospy.Subscriber('/odom/filtered', Odometry, self.gt_topic_cb, queue_size=1)

        self.fig, self.axes = plt.subplots(2,1)
        self.fig.show()

        data_dict   = {'x': [], 'y': [], 'z': [], 'w': [], 'gtw': []}
        _MIN = 10000
        _MAX = -10000
        mins_dict = {'x': _MIN, 'y': _MIN, 'z': _MIN, 'w': _MIN, 'gtw': _MIN}
        maxs_dict = {'x': _MAX, 'y': _MAX, 'z': _MAX, 'w': _MAX, 'gtw': _MAX}
        hist_len  = 1000
        while not rospy.is_shutdown():
            self.rate_obj.sleep() # reduce cpu load
            if not (self.new_msg and self.gtnew_msg):
                continue #denest
            self.new_msg = False
            self.gtnew_msg = False

            ## todo each loop:
            # get new data:
            new____x = round(self.msg.vector.x,3)
            new____y = round(self.msg.vector.y,3)
            new____z = round(self.msg.vector.z,3)
            new____w = round(math.atan2(self.msg.vector.y, self.msg.vector.x),3)
            new__gtw = round(yaw_from_q(self.gtmsg.pose.pose.orientation), 3)

            # store new data:
            data_dict['x'].append(new____x)
            data_dict['y'].append(new____y)
            data_dict['z'].append(new____z)
            data_dict['w'].append(new____w)
            data_dict['gtw'].append(new__gtw)

            # crunch plot limits:
            mins_dict = {key: data_dict[key][-1] if data_dict[key][-1] < mins_dict[key] else mins_dict[key] for key in list(mins_dict.keys())}
            maxs_dict = {key: data_dict[key][-1] if data_dict[key][-1] > maxs_dict[key] else maxs_dict[key] for key in list(maxs_dict.keys())}

            # clean axes:
            length_arr = len(data_dict['x'])
            while length_arr > hist_len:
                [data_dict[key].pop(0) for key in list(data_dict.keys())]
                length_arr -= 1
            spacing = np.arange(length_arr)
            [i.clear() for i in self.axes] # clear old data from axes

            if length_arr < 10:
                continue

            # plot data:
            self.axes[0].plot(spacing, data_dict['x'], 'r')
            self.axes[0].plot(spacing, data_dict['y'], 'g')
            self.axes[0].plot(spacing, data_dict['z'], 'b')
            self.axes[1].plot(spacing, data_dict['w'], 'r')
            self.axes[1].plot(spacing, data_dict['gtw'], 'b')

            # draw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.001)

if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(prog="vector viewer", 
                                description="ROS Vector Viewer Tool",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
        parser.add_argument('--mode', '-m', type=check_string, choices=["imu","odom", "dOdom", "vec3s"], default="odom", help="Specify ROS log level (default: %(default)s).")
        parser.add_argument('--topic', '-t', type=check_string, default=None, help='Set node rate (default: %(default)s).')
        parser.add_argument('--rate', '-r', type=check_positive_float, default=10.0, help='Set node rate (default: %(default)s).')
        parser.add_argument('--node-name', '-N', default="view_vector", help="Specify node name (default: %(default)s).")
        parser.add_argument('--anon', '-a', type=check_bool, default=True, help="Specify whether node should be anonymous (default: %(default)s).")
        parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2, help="Specify ROS log level (default: %(default)s).")

        raw_args    = parser.parse_known_args()
        args        = vars(raw_args[0])

        mode        = args['mode']
        rate        = args['rate']
        log_level   = args['log_level']
        node_name   = args['node_name']
        anon        = args['anon']
        topic       = args['topic']

        viewer = Viewer(node_name, anon, log_level, rate, mode, topic)
        viewer.main()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        plt.close('all')
