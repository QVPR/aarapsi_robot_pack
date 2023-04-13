#!/usr/bin/env python3

import rospy
import copy
import numpy as np
import argparse as ap

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseArray, Pose, Point
from std_msgs.msg import Header

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import LogType, roslogger, yaw_from_q, q_from_yaw

class State_Estimate:
    def __init__(self):
        self.rx     = 0.0 # position x
        self.ry     = 0.0 # position y
        self.rw     = 0.0 # position yaw
        self.vx     = 0.0 # velocity x
        self.vw     = 0.0 # velocity yaw

    def __del__(self):
        del self.rx
        del self.ry
        del self.rw
        del self.vx
        del self.vw

class State_Obj:
    def __init__(self, nmrc, force_period=None, key_distance=None, max_keys=20):

        self.odom       = Odometry()
        self.cmd        = Twist()
        self.new_odom   = False
        self.new_cmd    = False

        self.init_r     = False
        self.init_v     = False
        self.init_a     = False

        self.x          = 0.0
        self.y          = 0.0
        self.w          = 0.0
        self.vx         = 0.0
        self.vw         = 0.0

        self.nmrc       = nmrc
        self.dt         = 1/self.nmrc.rate_num

        if force_period is None:
            self.fp     = self.dt * 20
        else:
            self.fp     = force_period

        if key_distance is None:
            self.kd     = self.dt * 10
        else:
            self.kd     = key_distance

        self.new        = State_Estimate()
        self.old        = State_Estimate()
        self.dist       = {'mean': 0.0, 'std': 0.0, 'count': 0}

        self.lost       = -1
        self.rlines     = 2 # for logging

        self.calc_keys  = PoseArray(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        self.data_keys  = PoseArray(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        self.max_keys   = max_keys

        self.time       = rospy.Time.now().to_sec()

    def iterate(self):
        self.update_v(self.cmd.linear.x, self.cmd.angular.z)
        self.update_r(self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, yaw_from_q(self.odom.pose.pose.orientation))
        self.integrate()
        self.step()
        self.nmrc.print_comp_to_gt()

    def update_v(self, vx, vw):
        if not self.new_cmd:
            return # denest
        self.new_cmd = False

        # if new Twist message (for velocity):
        self.vx = vx
        self.vw = vw
        self.new.vx = vx
        self.new.vw = vw

    def update_r(self, x, y, w):
        if not self.new_odom:
            return # denest
        self.new_odom = False

        # if new Odometry message (for position):
        self.x = x
        self.y = y
        self.w = w
        if not self.init_r:
            self.new.rx = x
            self.new.ry = y
            self.new.rw = w
            self.init_r = True
        else:
            # crunch rolling mean & standard deviation
            euc_dist    = np.sqrt((self.new.rx - x)**2 + (self.new.ry - y)**2 + (self.new.rw - w)**2)
            new_mean    = ((self.dist['mean'] * self.dist['count']) + euc_dist) / (self.dist['count'] + 1)
            new_std     = np.sqrt((((self.dist['std']**2)*self.dist['count']) + (euc_dist - self.dist['mean'])**2) / (self.dist['count'] + 1))
            if self.dist['count'] < 19:
                new_count   = self.dist['count'] + 1
            else:
                new_count   = self.dist['count']
            weight      = euc_dist / (new_mean + new_std)

            self.nmrc.rosprint(str((weight, euc_dist, new_mean, new_std)), LogType.INFO)

            if weight < 1:
                self.lost   = -1
                self.new.rx = round((self.new.rx * (weight) + x * (1-weight)),3)
                self.new.ry = round((self.new.ry * (weight) + y * (1-weight)),3)
                self.new.rw = round((self.new.rw * (weight) + w * (1-weight)),3)
                self.dist.update({'mean': new_mean, 'std': new_std, 'count': new_count})
            else:
                if self.lost < 0:
                    self.lost = rospy.Time.now().to_sec()
                elif rospy.Time.now().to_sec() - self.lost < self.fp:
                    self.nmrc.rosprint("Bad state. Duration: %0.4f (Force Period: %0.4f)" % (rospy.Time.now().to_sec() - self.lost, self.fp), LogType.WARN, throttle=5)
                else:
                    return
                    self.nmrc.rosprint("Bad state duration too long. Forcing position estimate back on path.", LogType.ERROR, throttle=5)
                    self.lost   = -1  
                    self.new.rx = x
                    self.new.ry = y
                    self.new.rw = w
                
    def integrate(self):
        self.new.rx = self.new.rx + (self.new.vx * np.cos(self.new.rw) * self.dt)
        self.new.ry = self.new.ry + (self.new.vx * np.sin(self.new.rw) * self.dt)
        self.new.rw = self.new.rw + (self.new.vw * self.dt)

    def step(self):
        del self.old
        self.old    = copy.deepcopy(self.new)

        if (rospy.Time.now().to_sec() - self.time) > self.kd:
            self.time = rospy.Time.now().to_sec()
            self.add_pose()

    def add_pose(self):
        while len(self.calc_keys.poses) > (self.max_keys - 1): 
            self.calc_keys.poses.pop(0)
            self.data_keys.poses.pop(0)
        calc_pose = Pose(position=Point(x=self.new.rx, y=self.new.ry, z=0), orientation=q_from_yaw(self.new.rw))
        data_pose = Pose(position=Point(x=self.x, y=self.y, z=0), orientation=q_from_yaw(self.w))
        self.calc_keys.poses.append(calc_pose)
        self.data_keys.poses.append(data_pose)

class mrc:
    def __init__(self, node_name, rate, anon, log_level):
        
        rospy.init_node(node_name, anonymous=anon, log_level=1)
        rospy.loginfo('Starting %s node.' % (node_name))

        self.rate_num       = rate
        self.rate_obj       = rospy.Rate(self.rate_num)

        self.state          = State_Obj(self, force_period=2)

        self.odom_sub       = rospy.Subscriber('/odom/in',    Odometry,  self.odom_cb, queue_size=1)
        self.cmd_sub        = rospy.Subscriber('/cmd_vel/in', Twist,     self.cmd_cb,  queue_size=1)
        self.odom_pub       = rospy.Publisher('/odom/out',    Odometry,                queue_size=1)

        self.calc_keys_pub  = rospy.Publisher('/keys/calc',   PoseArray,               queue_size=1)
        self.data_keys_pub  = rospy.Publisher('/keys/data',   PoseArray,               queue_size=1)

    def odom_cb(self, msg):
        self.state.odom     = msg
        self.state.new_odom = True

    def cmd_cb(self, msg):
        # arrives in 'odom' frame i.e. x-axis points in direction of travel
        self.state.cmd      = msg
        self.state.new_cmd  = True

    def do_pub(self):
        msg_to_pub                       = Odometry()
        msg_to_pub.header.stamp          = rospy.Time.now()
        msg_to_pub.header.frame_id       = "map"
        msg_to_pub.pose.pose.position.x  = self.state.new.rx
        msg_to_pub.pose.pose.position.y  = self.state.new.ry
        msg_to_pub.pose.pose.orientation = q_from_yaw(self.state.new.rw)
        msg_to_pub.twist.twist.linear.x  = self.state.new.vx
        msg_to_pub.twist.twist.angular.z = self.state.new.vw
        self.odom_pub.publish(msg_to_pub)

        if len(self.state.calc_keys.poses) > 0: 
            self.calc_keys_pub.publish(self.state.calc_keys)
            self.data_keys_pub.publish(self.state.data_keys)

    def rosprint(self, text, logtype, throttle=0):
        roslogger(text + '\n\n', logtype, ros=True, throttle=throttle, no_stamp=True)

    def print(self, input, indent=5, rb=True):
        # roll back:
        if rb:
            lines = len(input)
            print('\033[1A'*lines, end='\x1b[2K')
        # print text:
    
        for i in input: 
            roslogger(('     '*indent) + i, LogType.INFO, ros=True, throttle=0, no_stamp=True)
        #print('\033[1A', end='\x1b[2K')

    def print_comp_to_gt(self):
        self.print(["ours: (%10.4f, %10.4f, %10.4f)" % (self.state.new.rx, self.state.new.ry, self.state.new.rw),
                    "true: (%10.4f, %10.4f, %10.4f)" % (self.state.x, self.state.y, self.state.w)])
    
    def print_vel(self):
        self.print(["v: (%10.4f, %10.4f)" % (self.state.new.vx / self.state.dt, self.state.new.vw / self.state.dt)])

    def print_new_old(self):
        self.print(["r: (%10.4f, %10.4f, %10.4f) (%10.4f, %10.4f, %10.4f)" % (self.state.new.rx, self.state.new.ry, self.state.new.rw, self.state.old.rx, self.state.old.ry, self.state.old.rw),
                    "v: (%10.4f, %10.4f) (%10.4f, %10.4f)" % (self.state.new.vx, self.state.new.vw, self.state.old.vx, self.state.old.vw)])

    def print_diff(self):
        self.print(["r: (%10.4f, %10.4f, %10.4f)" % (self.state.new.rx - self.state.old.rx, self.state.new.ry - self.state.old.ry, self.state.new.rw - self.state.old.rw),
                    "v: (%10.4f, %10.4f)" % (self.state.new.vx - self.state.old.vx, self.state.new.vw - self.state.old.vw)])
        
    def main(self):
        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            self.state.iterate()
            self.do_pub()

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
        nmrc.main()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass