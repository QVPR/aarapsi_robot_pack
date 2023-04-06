#!/usr/bin/env python3

import rospy
import copy
import sys
import numpy as np
import argparse as ap
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Twist
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import LogType, roslogger

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
    def __init__(self, nmrc, force_period=None):

        self.init_r = False
        self.init_v = False
        self.init_a = False

        self.x      = 0.0
        self.y      = 0.0
        self.w      = 0.0
        self.vx     = 0.0
        self.vw     = 0.0

        self.nmrc   = nmrc
        self.dt     = 1/self.nmrc.rate_num

        if force_period is None:
            self.fp = self.dt * 5
        else:
            self.fp = force_period

        self.new    = State_Estimate()
        self.old    = State_Estimate()
        self.dist   = {'mean': 0.0, 'std': 0.0, 'count': 0}

        self.lost   = -1
        self.rlines = 2 # for logging

    def update_v(self, vx, vw):
        if not self.nmrc.new_cmd:
            return # denest
        self.nmrc.new_cmd = False

        # if new Twist message (for velocity):
        self.vx = vx
        self.vw = vw
        self.new.vx = vx
        self.new.vw = vw

    def update_r(self, x, y, w):
        if not self.nmrc.new_odom:
            return # denest
        self.nmrc.new_odom = False

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
            new_count   = self.dist['count'] + 1
            weight      = euc_dist / (new_mean + 2*new_std)

            if weight < 1:
                self.new.rx = round((self.new.rx * (weight) + x * (1-weight)),3)
                self.new.ry = round((self.new.ry * (weight) + y * (1-weight)),3)
                self.new.rw = round((self.new.rw * (weight) + w * (1-weight)),3)
                self.dist.update({'mean': new_mean, 'std': new_std, 'count': new_count})
                self.lost = -1
            else:
                if self.lost < 0:
                    self.lost = rospy.Time.now().to_sec()
                elif rospy.Time.now().to_sec() - self.lost < self.fp:
                    self.rosprint("Bad state. Duration: %0.4f (Force Period: %0.4f)" % (rospy.Time.now().to_sec() - self.lost, self.fp), LogType.WARN, throttle=5)
                else:
                    self.rosprint("Bad state duration too long. Forcing position estimate back on path.", LogType.ERROR, throttle=5)
                    self.new.rx = x
                    self.new.ry = y
                    self.new.rw = w
                    self.lost   = -1  
                
    def integrate(self):
        self.new.rx = self.new.rx + (self.new.vx * np.cos(self.new.rw) * self.dt)
        self.new.ry = self.new.ry + (self.new.vx * np.sin(self.new.rw) * self.dt)
        self.new.rw = self.new.rw + (self.new.vw * self.dt)

    def step(self):
        del self.old
        self.old    = copy.deepcopy(self.new)

    def rosprint(self, text, logtype, throttle=0):
        roslogger(text + '\n\n', logtype, ros=True, throttle=throttle, no_stamp=True)

    def print(self, input, indent=5, rb=True):
        # roll back:
        if rb:
            lines = len(input)
            print('\033[1A'*lines, end='\x1b[2K')
        # print text:
        for i in input: print(('\t'*indent) + i)

    def print_comp_to_gt(self):
        self.print(["ours: (%10.4f, %10.4f, %10.4f)" % (self.new.rx, self.new.ry, self.new.rw),
                    "true: (%10.4f, %10.4f, %10.4f)" % (self.x, self.y, self.w)])
    
    def print_vel(self):
        self.print(["v: (%10.4f, %10.4f, %10.4f)" % (self.new.vx / self.dt, self.new.vy / self.dt, self.new.vw / self.dt)])

    def print_new_old(self):
        self.print(["r: (%10.4f, %10.4f, %10.4f) (%10.4f, %10.4f, %10.4f)" % (self.new.rx, self.new.ry, self.new.rw, self.old.rx, self.old.ry, self.old.rw),
                    "v: (%10.4f, %10.4f, %10.4f) (%10.4f, %10.4f, %10.4f)" % (self.new.vx, self.new.vy, self.new.vw, self.old.vx, self.old.vy, self.old.vw)])

    def print_diff(self):
        self.print(["r: (%10.4f, %10.4f, %10.4f)" % (self.new.rx - self.old.rx, self.new.ry - self.old.ry, self.new.rw - self.old.rw),
                    "v: (%10.4f, %10.4f, %10.4f)" % (self.new.vx - self.old.vx, self.new.vy - self.old.vy, self.new.vw - self.old.vw),
                    "a: (%10.4f, %10.4f, %10.4f)" % (self.new.ax - self.old.ax, self.new.ay - self.old.ay, self.new.aw - self.old.aw)])

    def __del__(self):
        del self.init_r
        del self.init_v
        del self.init_a
        del self.x
        del self.y
        del self.w
        del self.vx
        del self.vw
        del self.dt
        del self.new
        del self.old
        del self.dist
        del self.lost

class mrc:
    def __init__(self, node_name, rate, anon, log_level, odom_topic, cmd_topic, pub_topic):
        
        rospy.init_node(node_name, anonymous=anon, log_level=log_level)
        rospy.loginfo('Starting %s node.\n\n' % (node_name))

        self.rate_num   = rate
        self.rate_obj   = rospy.Rate(self.rate_num)

        self.odom       = Odometry()
        self.cmd        = Twist()
        self.new_odom   = False
        self.new_cmd    = False

        self.odom_topic = odom_topic
        self.cmd_topic  = cmd_topic
        self.pub_topic  = pub_topic

        self.odom_sub   = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.cmd_sub    = rospy.Subscriber(self.cmd_topic, Twist, self.cmd_cb, queue_size=1)
        self.odom_pub   = rospy.Publisher(self.pub_topic, Odometry, queue_size=1)

        self.state      = State_Obj(self, force_period=2)

    def odom_cb(self, msg):
        self.odom       = msg
        self.new_odom   = True

    def cmd_cb(self, msg):
        # arrives in 'odom' frame i.e. x-axis points in direction of travel
        self.cmd        = msg
        self.new_cmd    = True

    def do_pub(self):
        msg_to_pub                          = Odometry()
        msg_to_pub.header.stamp             = rospy.Time.now()
        msg_to_pub.header.frame_id          = "map"
        msg_to_pub.pose.pose.position.x     = self.state.new.rx
        msg_to_pub.pose.pose.position.y     = self.state.new.ry
        msg_to_pub.pose.pose.orientation    = self.q_from_yaw(self.state.new.rw)
        msg_to_pub.twist.twist.linear.x     = self.state.new.vx
        msg_to_pub.twist.twist.angular.z    = self.state.new.vw
        self.odom_pub.publish(msg_to_pub)
        
    def yaw_from_q(self, orientation):
        return euler_from_quaternion([float(orientation.x), float(orientation.y), float(orientation.z), float(orientation.w)])[2]
    
    def q_from_yaw(self, yaw):
        q = quaternion_from_euler(0, 0, yaw) # roll, pitch, yaw
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
    def main(self):
        while not rospy.is_shutdown():
            self.rate_obj.sleep()

            self.state.update_v(self.cmd.linear.x, self.cmd.angular.z)
            self.state.update_r(self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.yaw_from_q(self.odom.pose.pose.orientation))

            self.state.integrate()
            self.state.step()

            self.do_pub()
            self.state.print_comp_to_gt()

if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(prog="odometry smoother", 
                                description="ROS Odometry Smoother Tool",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
        parser.add_argument('--node-name', '-N', type=check_string,              default="smooth",                              help="Specify node name (default: %(default)s).")
        parser.add_argument('--rate', '-r',      type=check_positive_float,      default=10.0,                                  help='Set node rate (default: %(default)s).')
        parser.add_argument('--anon', '-a',      type=check_bool,                default=False,                                 help="Specify whether node should be anonymous (default: %(default)s).")
        parser.add_argument('--odom-topic',      type=check_string,              default='/vpr_nodes/vpr_odom',                 help="Specify input odometry topic (default: %(default)s).")
        parser.add_argument('--cmd-topic',       type=check_string,              default='/jackal_velocity_controller/cmd_vel', help="Specify input command velocity topic (default: %(default)s).")
        parser.add_argument('--pub-topic',       type=check_string,              default='/vpr_nodes/vpr_odom/filtered',        help="Specify output topic (default: %(default)s).")
        parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2,                                     help="Specify ROS log level (default: %(default)s).")
        
        raw_args = parser.parse_known_args()
        args = vars(raw_args[0])

        node_name   = args['node_name']
        rate        = args['rate']
        anon        = args['anon']
        odom_topic   = args['odom_topic']
        cmd_topic   = args['cmd_topic']
        pub_topic   = args['pub_topic']
        log_level   = args['log_level']

        nmrc = mrc(node_name, rate, anon, log_level, odom_topic, cmd_topic, pub_topic)
        nmrc.main()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass