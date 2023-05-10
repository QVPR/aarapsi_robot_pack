#!/usr/bin/env python3

import rospy
import copy
import sys

import numpy as np
import argparse as ap
from cv_bridge import CvBridge

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseArray, Pose, Point
from std_msgs.msg import Header, Float64
from scipy.spatial.distance import cdist

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools import roslogger, LogType, yaw_from_q, ROS_Param
from pyaarapsi.core.helper_tools import formatException, np_ndarray_to_uint8_list

from aarapsi_robot_pack.srv import GenerateObj, GenerateObjRequest, GetSafetyStates, GetSafetyStatesRequest
from aarapsi_robot_pack.msg import ControllerStateInfo, xyw, MonitorDetails, CompressedMonitorDetails

class mrc:
    def __init__(self, node_name, rate, anon, log_level):
        
        rospy.init_node(node_name, anonymous=anon, log_level=log_level)
        roslogger('Starting %s node.' % (node_name), LogType.INFO, ros=True)

        self.rate_num       = rate
        self.dt             = 1/rate
        self.rate_obj       = rospy.Rate(self.rate_num)

        self.current_yaw    = 0.0
        self.target_yaw     = 0.0
        self.true_yaw       = 0.0
        self.delta_yaw      = 0.0
        self.old_sensor_cmd = None

        self.target_update  = False
        self.lookahead_inds = 5
        self.vpr_ego        = [0.0, 0.0, 0.0] # x y w
        self.vpr_ego_hist   = []

        self.time           = rospy.Time.now().to_sec()
        
        self.ready          = True
        self.path_received  = False
        self.path_processed = False
        self.svm_details    = None

        self.vpr_odom       = '/vpr_nodes/vpr_odom'
        self.jackal_odom    = '/jackal_velocity_controller/odom'
        self.gt_odom        = '/odom/filtered'

        self.bridge         = CvBridge()

        if bool(rospy.get_param('/vpr_nodes/vpr_monitor/compress/out')):
            self.state_topic    = '/vpr_nodes/state/compressed'
            self.state_type     = CompressedMonitorDetails
            self.img_convert    = lambda img: np_ndarray_to_uint8_list(self.bridge.compressed_imgmsg_to_cv2(img, "bgr8"))
        else:
            self.state_topic    = '/vpr_nodes/state'
            self.state_type     = CompressedMonitorDetails
            self.img_convert    = lambda img: np_ndarray_to_uint8_list(self.bridge.imgmsg_to_cv2(img, "passthrough"))

        self.vpr_path_sub   = rospy.Subscriber('/vpr_nodes/path',                   Path,               self.path_cb,       queue_size=1) # from vpr_cruncher
        self.sensors_sub    = rospy.Subscriber(self.jackal_odom,                    Odometry,           self.sensors_cb,    queue_size=1) # wheel encoders (and maybe imu ??? don't think so)
        self.gt_sub         = rospy.Subscriber(self.gt_odom,                        Odometry,           self.gt_cb,         queue_size=1) # ONLY for ground truth
        self.state_sub      = rospy.Subscriber(self.state_topic,                    self.state_type,    self.state_cb,      queue_size=1)
        self.info_pub       = rospy.Publisher('/vpr_nodes/' + node_name + '/info',  ControllerStateInfo,                    queue_size=1)
        self.twist_pub      = rospy.Publisher('/twist2joy/in',                      Twist,                                  queue_size=1)
        self.srv_path       = rospy.ServiceProxy('/vpr_nodes/path',                 GenerateObj)
        self.srv_safety     = rospy.ServiceProxy('/vpr_nodes/safety',               GetSafetyStates)
        
        self.main()

    def srv_GetPath(self, generate=False):
        try:
            if not self.ready:
                return
            requ = GenerateObjRequest()
            requ.generate = generate
            resp = self.srv_path(requ)
            if resp.success == False:
                roslogger('[srv_GetPath] Service executed, success=False!', LogType.ERROR, ros=True)
        except:
            rospy.logerr(formatException())

    def pose2xyw(self, pose, stamped=False):
        if stamped:
            pose = pose.pose
        return [pose.position.x, pose.position.y, yaw_from_q(pose.orientation)]

    def path_process(self):
    # Abstracted to separate function as the for loop execution time may be long
        self.path_array     = np.array([self.pose2xyw(pose, stamped=True) for pose in self.path_msg.poses])
        self.path_processed = True

    def gt_cb(self, msg):
        self.true_yaw = yaw_from_q(msg.pose.pose.orientation)

    def path_cb(self, msg):
        self.path_msg       = msg
        self.path_received  = True

    def state_cb(self, msg):
        if not self.ready:
            return
        
        self.svm_details    = msg
        self.vpr_ego        = [msg.data.vpr_ego.x, msg.data.vpr_ego.y, msg.data.vpr_ego.w]
        self.vpr_ego_hist.append(self.vpr_ego)
    
    def sensors_cb(self, msg):
        if not self.ready:
            return
        if self.old_sensor_cmd is None:
            self.old_sensor_cmd = msg
            return
        
        self.delta_yaw      = self.angle_wrap(yaw_from_q(msg.pose.pose.orientation) - yaw_from_q(self.old_sensor_cmd.pose.pose.orientation))
        self.current_yaw    = self.angle_wrap(0.5 * self.angle_wrap(self.current_yaw + self.vpr_ego[2]) + self.delta_yaw)
        self.old_sensor_cmd = msg

    def update_target(self):
        spd                 = cdist(self.path_array[:,0:2], np.matrix([self.vpr_ego[0], self.vpr_ego[1]]))
        target_index        = (np.argmin(spd[:]) + self.lookahead_inds) % self.path_array.shape[0]
        self.target_yaw     = self.path_array[target_index, 2]

    def angle_wrap(self, angle_in, mode='DEG'):
        if mode == 'DEG':
            return ((angle_in + 180.0) % 360) - 180
        elif mode == 'RAD':
            return ((angle_in + np.pi) % (np.pi * 2)) - np.pi
        else:
            raise Exception('Mode must be either DEG or RAD.')

    def main(self):
        while not rospy.is_shutdown():
            if not self.path_received:
                roslogger("Waiting for path ...", LogType.INFO, ros=True, throttle=10)
                self.srv_GetPath()
                rospy.sleep(1)
                continue
            
            self.path_process()
            break
        roslogger("Path processed. Entering main loop.", LogType.INFO, ros=True)

        while not rospy.is_shutdown():
            self.rate_obj.sleep()

            self.update_target()

            yaw_cmd             = -1 * self.angle_wrap(self.target_yaw - self.current_yaw)
            new_twist           = Twist()
            new_twist.linear.x  = 0.5
            new_twist.angular.z = yaw_cmd

            roslogger("Target: %0.2f, Current: %0.2f" % (self.target_yaw, self.current_yaw), LogType.INFO, ros=True)
            roslogger("True Current: %0.2f, Error: %0.2f" % (self.true_yaw, self.angle_wrap(self.true_yaw - self.current_yaw)), LogType.INFO, ros=True)

            safety_states_srv       = self.srv_safety(GetSafetyStatesRequest())

            msg                     = ControllerStateInfo()
            try:
                msg.query_image     = self.img_convert(self.svm_details.queryImage)
                msg.label_details   = self.svm_details.data
                msg.mState          = self.svm_details.mState
                msg.prob            = self.svm_details.prob
                msg.mStateBin       = self.svm_details.mStateBin
                msg.factors         = self.svm_details.factors
            except:
                pass
            msg.safety_states       = safety_states_srv.states

            msg.current_yaw         = self.current_yaw
            msg.target_yaw          = self.target_yaw
            msg.true_yaw            = self.true_yaw
            msg.delta_yaw           = self.delta_yaw

            msg.lookahead           = self.lookahead_inds
            msg.lookahead_mode      = 'index-based'

            msg.vpr_topic           = self.vpr_odom
            msg.jackal_topic        = self.jackal_odom
            msg.groundtruth_topic   = self.gt_odom
            self.info_pub.publish(msg)
            
        self.exit()

    def exit(self):
        roslogger("Exit state received.", LogType.INFO, ros=False)
        sys.exit()

if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(prog="Path Follower", 
                                description="ROS Path Follower Tool",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
        parser.add_argument('--node-name', '-N', type=check_string,              default="follower",  help="Specify node name (default: %(default)s).")
        parser.add_argument('--rate',      '-r', type=check_positive_float,      default=10.0,        help='Set node rate (default: %(default)s).')
        parser.add_argument('--anon',      '-a', type=check_bool,                default=False,       help="Specify whether node should be anonymous (default: %(default)s).")
        parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2,           help="Specify ROS log level (default: %(default)s).")
        
        raw_args = parser.parse_known_args()
        args = vars(raw_args[0])

        node_name   = args['node_name']
        rate        = args['rate']
        anon        = args['anon']
        log_level   = args['log_level']

        nmrc = mrc(node_name, rate, anon, log_level)
    except rospy.ROSInterruptException:
        pass