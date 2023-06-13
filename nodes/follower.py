#!/usr/bin/env python3

import rospy
import sys

import numpy as np
import argparse as ap
import cv2
from cv_bridge import CvBridge

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String
from fastdist import fastdist

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string, check_positive_int
from pyaarapsi.core.ros_tools import roslogger, LogType, NodeState, yaw_from_q, q_from_yaw, set_rospy_log_lvl, init_node
from pyaarapsi.core.helper_tools import formatException, np_ndarray_to_uint8_list, angle_wrap

from aarapsi_robot_pack.srv import GenerateObj, GenerateObjRequest, GetSafetyStates, GetSafetyStatesRequest
from aarapsi_robot_pack.msg import ControllerStateInfo, MonitorDetails

class mrc:
    def __init__(self, node_name, rate_num, namespace, anon, log_level, reset, order_id=0):

        if not init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30):
            raise Exception('init_node failed.')

        self.init_params(rate_num, log_level, reset)
        self.init_vars()
        self.init_rospy()

        self.main_ready      = True
        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_params(self, rate_num, log_level, reset):
        self.ODOM_TOPIC      = self.ROS_HOME.params.add(self.namespace + "/odom_topic",          None,     check_string,         force=False)
        
        self.LOG_LEVEL       = self.ROS_HOME.params.add(self.nodespace + "/log_level",          log_level, check_positive_int,   force=reset)
        self.RATE_NUM        = self.ROS_HOME.params.add(self.nodespace + "/rate",               rate_num,  check_positive_float, force=reset)
        self.SVM_OVERRIDE    = self.ROS_HOME.params.add(self.nodespace + "/svm_override",       False,     check_bool,           force=reset)
        

    def init_vars(self):
        self.dt             = 1/self.RATE_NUM.get()

        self.current_yaw    = 0.0
        self.target_yaw     = 0.0
        self.true_yaw       = 0.0
        self.delta_yaw      = 0.0

        self.lookahead_inds = 5
        self.vpr_ego        = [0.0, 0.0, 0.0] # x y w
        self.vpr_ego_hist   = []
        
        self.ready          = True
        self.path_received  = False
        self.path_processed = False
        self.new_sensor_msg = False
        self.new_state_msg  = False

        self.state_msg      = None
        self.sensor_msg     = None
        self.old_sensor_msg = None

        self.jackal_odom    = '/jackal_velocity_controller/odom'

        self.bridge         = CvBridge()

    def init_rospy(self):
        self.rate_obj       = rospy.Rate(self.RATE_NUM.get())
        self.time           = rospy.Time.now().to_sec()

        self.vpr_path_sub   = rospy.Subscriber(self.namespace + '/path',           Path,                self.path_cb,        queue_size=1) # from vpr_cruncher
        self.sensors_sub    = rospy.Subscriber(self.jackal_odom,                   Odometry,            self.sensors_cb,     queue_size=1) # wheel encoders (and maybe imu ??? don't think so)
        self.gt_sub         = rospy.Subscriber(self.ODOM_TOPIC.get(),              Odometry,            self.gt_cb,          queue_size=1) # ONLY for ground truth
        self.state_sub      = rospy.Subscriber(self.namespace + '/state',          MonitorDetails,      self.state_cb,       queue_size=1)
        self.param_sub      = rospy.Subscriber(self.namespace + "/params_update",  String,              self.param_callback, queue_size=100)
        self.info_pub       = self.ROS_HOME.add_pub(self.nodespace + '/info',      ControllerStateInfo,                      queue_size=1)
        self.twist_pub      = self.ROS_HOME.add_pub('/twist2joy/in',               TwistStamped,                             queue_size=1)
        self.ego_good_pub   = self.ROS_HOME.add_pub(self.nodespace + '/odom/good', Odometry,                                 queue_size=1)
        self.ego_bad_pub    = self.ROS_HOME.add_pub(self.nodespace + '/odom/bad',  Odometry,                                 queue_size=1)
        self.srv_path       = rospy.ServiceProxy(self.namespace + '/path',         GenerateObj)
        self.srv_safety     = rospy.ServiceProxy(self.namespace + '/safety',       GetSafetyStates)
        
        self.main()

    def param_callback(self, msg):
        self.parameters_ready = False
        if self.ROS_HOME.params.exists(msg.data):
            if not self.ROS_HOME.params.update(msg.data):
                self.print("Change to parameter [%s]; bad value." % msg.data, LogType.DEBUG)
        
            else:
                self.print("Change to parameter [%s]; updated." % msg.data, LogType.DEBUG)

                if msg.data == self.LOG_LEVEL.name:
                    set_rospy_log_lvl(self.LOG_LEVEL.get())
                elif msg.data == self.RATE_NUM.name:
                    self.rate_obj = rospy.Rate(self.RATE_NUM.get())
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)
        self.parameters_ready = True

    def srv_GetPath(self, generate=False):
        try:
            if not self.ready:
                return
            requ = GenerateObjRequest()
            requ.generate = generate
            resp = self.srv_path(requ)
            if resp.success == False:
                self.print('[srv_GetPath] Service executed, success=False!', LogType.ERROR)
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
        
        self.state_msg      = msg
        self.vpr_ego        = [msg.data.vpr_ego.x, msg.data.vpr_ego.y, msg.data.vpr_ego.w]
        self.vpr_ego_hist.append(self.vpr_ego)
        self.new_state_msg  = True
    
    def sensors_cb(self, msg):
        if not self.ready:
            return
        if self.sensor_msg is None:
            self.old_sensor_msg = msg
        else:
            self.old_sensor_msg = msg
        self.sensor_msg     = msg
        self.new_sensor_msg = True


    def update_target(self):
        spd                 = fastdist.matrix_to_matrix_distance(self.path_array[:,0:2], \
                                                                    np.matrix([self.vpr_ego[0], self.vpr_ego[1]]), \
                                                                    fastdist.euclidean, "euclidean")
        target_index        = (np.argmin(spd[:]) + self.lookahead_inds) % self.path_array.shape[0]
        self.target_yaw     = self.path_array[target_index, 2]

    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            if not self.path_received:
                self.print("Waiting for path ...", throttle=10)
                self.srv_GetPath()
                rospy.sleep(1)
                continue
            
            self.path_process()
            break
        self.print("Path processed. Entering main loop.")

        while not rospy.is_shutdown():
            self.loop_contents()
        
        self.exit()

    def publish_controller_info(self):
        msg                         = ControllerStateInfo()

        msg.query_image             = self.state_msg.queryImage
        # Extract Label Details:
        msg.dvc                     = self.state_msg.data.dvc
        msg.group.gt_ego            = self.state_msg.data.gt_ego
        msg.group.vpr_ego           = self.state_msg.data.vpr_ego
        msg.group.matchId           = self.state_msg.data.matchId
        msg.group.trueId            = self.state_msg.data.trueId
        msg.group.gt_state          = self.state_msg.data.gt_state
        msg.group.gt_error          = self.state_msg.data.gt_error
        # Extract Monitor Details:
        msg.group.mState            = self.state_msg.mState
        msg.group.prob              = self.state_msg.prob
        msg.group.mStateBin         = self.state_msg.mStateBin
        msg.group.factors           = self.state_msg.factors

        safety_states_srv           = self.srv_safety(GetSafetyStatesRequest())
        msg.group.safety_states     = safety_states_srv.states

        msg.group.current_yaw       = self.current_yaw
        msg.group.target_yaw        = self.target_yaw
        msg.group.true_yaw          = self.true_yaw
        msg.group.delta_yaw         = self.delta_yaw

        msg.group.lookahead         = self.lookahead_inds
        msg.group.lookahead_mode    = 'index-based'

        msg.group.vpr_topic         = self.namespace + '/vpr_odom'
        msg.group.jackal_topic      = self.jackal_odom
        msg.group.groundtruth_topic = self.ODOM_TOPIC.get()

        self.info_pub.publish(msg)
    
    def loop_contents(self):
        
        if not (self.new_sensor_msg and self.new_state_msg):
            self.print("Waiting.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return # denest
        self.rate_obj.sleep()

        self.new_sensor_msg = False
        self.new_state_msg  = False

        self.update_target()
        
        current_ego                       = Odometry()
        current_ego.header.stamp          = rospy.Time.now()
        current_ego.header.frame_id       = 'map'
        current_ego.pose.pose.position.x  = self.vpr_ego[0]
        current_ego.pose.pose.position.y  = self.vpr_ego[1]
        current_ego.pose.pose.orientation = q_from_yaw(self.vpr_ego[2])

        new_twist           = TwistStamped()
        new_twist.header.stamp = rospy.Time.now()
        new_twist.header.frame_id = "odom"
        self.delta_yaw      = angle_wrap(yaw_from_q(self.sensor_msg.pose.pose.orientation) - yaw_from_q(self.old_sensor_msg.pose.pose.orientation))
        self.current_yaw    = angle_wrap(0.5 * angle_wrap(self.current_yaw + self.vpr_ego[2]) + self.delta_yaw)
        new_twist.twist.angular.z = -1 * angle_wrap(self.target_yaw - self.current_yaw)

        #self.print("Target: %0.2f, Current: %0.2f" % (self.target_yaw, self.current_yaw))
        #self.print("True Current: %0.2f, Error: %0.2f" % (self.true_yaw, angle_wrap(self.true_yaw - self.current_yaw)))

        self.publish_controller_info() # requires self.state_msg

        if self.state_msg.mStateBin == True or self.SVM_OVERRIDE.get():
            new_twist.twist.linear.x  = 0.5
            self.ego_good_pub.publish(current_ego)
        else:
            new_twist.twist.linear.x  = 0.2
            self.ego_bad_pub.publish(current_ego)

        self.twist_pub.publish(new_twist)

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

    def exit(self):
        self.print("Quit received.")
        sys.exit()


def do_args():
    parser = ap.ArgumentParser(prog="Path Follower.py", 
                            description="ROS Path Follower Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="follower",       help="Specify node name (default: %(default)s).")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=10.0,             help='Specify node rate (default: %(default)s).')
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=False,            help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset',            '-R',  type=check_bool,                   default=False,            help='Force reset of parameters to specified ones (default: %(default)s)')
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')

    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = mrc(args['node_name'], args['rate'], args['namespace'], args['anon'], args['log_level'], args['reset'], order_id=args['order_id'])
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