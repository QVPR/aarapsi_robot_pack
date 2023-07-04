#!/usr/bin/env python3

import rospy
import rospkg
import argparse as ap
import numpy as np
import sys
import os
import csv
import copy
import pydbus

from enum       import Enum
from fastdist   import fastdist

from nav_msgs.msg           import Path, Odometry
from std_msgs.msg           import Header, ColorRGBA
from geometry_msgs.msg      import PoseStamped, Point, Twist, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg        import Joy
from aarapsi_robot_pack.msg import ControllerStateInfo, MonitorDetails

from pyaarapsi.core.argparse_tools          import check_positive_float, check_bool, check_string, check_float_list, check_enum
from pyaarapsi.core.ros_tools               import NodeState, roslogger, LogType, q_from_yaw, pose2xyw, set_rospy_log_lvl
from pyaarapsi.core.helper_tools            import formatException, angle_wrap, normalize_angle, Bool
from pyaarapsi.core.enum_tools              import enum_name, enum_value, enum_value_options
from pyaarapsi.core.vars                    import C_I_RED, C_I_GREEN, C_I_YELLOW, C_I_BLUE, C_I_WHITE, C_RESET, C_CLEAR, C_UP_N
from pyaarapsi.vpr_classes.base             import Base_ROS_Class, base_optional_args

from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType

'''
Path Follower

Node description.

'''

class PS4_Buttons(Enum):
    X               = 0
    O               = 1
    Square          = 2
    Triangle        = 3
    LeftBumper      = 4
    RightBumper     = 5
    Share           = 6
    Options         = 7
    PS              = 8
    LeftStickIn     = 9
    RightStickIn    = 10
    LeftArrow       = 11
    RightArrow      = 12
    UpArrow         = 13
    DownArrow       = 14

class PS4_Triggers(Enum):
    LeftStickXAxis  = 0
    LeftStickYAxis  = 1
    LeftTrigger     = 2 # Released = 1, Pressed = -1
    RightStickXAxis = 3
    RightStickYAxis = 4
    RightTrigger    = 5 # Released = 1, Pressed = -1

class Safety_Mode(Enum):
    UNSET           = -1
    STOP            = 0
    SLOW            = 1
    FAST            = 2

class Lookahead_Mode(Enum):
    INDEX           = 0
    DISTANCE        = 1

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_params(self, rate_num, log_level, reset):
        super().init_params(rate_num, log_level, reset)

        self.PATH_FILE              = self.params.add(self.namespace + "/path/file",                None,               check_string,                           force=False)
        self.PATH_DENSITY           = self.params.add(self.namespace + "/path/density",             None,               check_positive_float,                   force=False)
        self.SLOW_LIN_VEL_MAX       = self.params.add(self.namespace + "/limits/slow/linear",       None,               check_positive_float,                   force=False)
        self.SLOW_ANG_VEL_MAX       = self.params.add(self.namespace + "/limits/slow/angular",      None,               check_positive_float,                   force=False)
        self.FAST_LIN_VEL_MAX       = self.params.add(self.namespace + "/limits/fast/linear",       None,               check_positive_float,                   force=False)
        self.FAST_ANG_VEL_MAX       = self.params.add(self.namespace + "/limits/fast/angular",      None,               check_positive_float,                   force=False)
        self.CONTROLLER_MAC         = self.params.add(self.namespace + "/controller_mac",           None,               check_string,                           force=False)
        self.JOY_TOPIC              = self.params.add(self.namespace + "/joy_topic",                None,               check_string,                           force=False)
        self.CMD_TOPIC              = self.params.add(self.namespace + "/cmd_topic",                None,               check_string,                           force=False)
        self.ROBOT_ODOM_TOPIC       = self.params.add(self.namespace + "/robot_odom_topic",         None,               check_string,                           force=False)
        self.VPR_ODOM_TOPIC         = self.params.add(self.namespace + "/vpr_odom_topic",           None,               check_string,                           force=False)

        self.USE_NOISE              = self.params.add(self.nodespace + "/noise/enable",             False,              check_bool,                             force=reset)
        self.NOISE_VALS             = self.params.add(self.nodespace + "/noise/vals",               [0.1]*3,            lambda x: check_float_list(x, 3),       force=reset)
        self.PUB_INFO               = self.params.add(self.nodespace + "/publish_info",             True,               check_bool,                             force=reset)
        self.SVM_OVERRIDE           = self.params.add(self.nodespace + "/svm_override",             False,              check_bool,                             force=reset)
        self.REVERSE                = self.params.add(self.nodespace + "/reverse",                  False,              check_bool,                             force=reset)
        self.SAFETY_OVERRIDE        = self.params.add(self.nodespace + "/override/safety",          Safety_Mode.UNSET,  lambda x: check_enum(x, Safety_Mode),   force=reset)
        self.AUTONOMOUS_OVERRIDE    = self.params.add(self.nodespace + "/override/autonomous",      Bool.UNSET,         lambda x: check_enum(x, Bool),          force=reset)

    def init_vars(self):
        super().init_vars()

        self.vpr_ego            = []
        self.vpr_ego_hist       = []
        self.slam_ego           = []
        self.robot_ego          = []
        self.old_robot_ego      = []
        self.lookahead          = 5
        self.lookahead_mode     = Lookahead_Mode.INDEX
        self.dt                 = 1/self.RATE_NUM.get()
        self.print_lines        = 0

        self.plan_path          = Path()
        self.ref_path           = Path()
        self.state_msg          = MonitorDetails()

        self.ready              = False
        self.new_state_msg      = False
        self.new_robot_ego      = False
        self.new_slam_ego       = False
        self.new_vpr_ego        = False
        self.autonomous_cmd     = False

        self.safety_mode        = Safety_Mode.STOP

        self.enable_ind         = enum_value(PS4_Buttons.X)
        self.disable_ind        = enum_value(PS4_Buttons.O)
        self.slow_mode_ind      = enum_value(PS4_Buttons.LeftBumper)
        self.fast_mode_ind      = enum_value(PS4_Buttons.RightBumper)
        self.netvlad_ind        = enum_value(PS4_Buttons.LeftArrow)
        self.hybridnet_ind      = enum_value(PS4_Buttons.RightArrow)
        self.raw_ind            = enum_value(PS4_Buttons.UpArrow)
        self.patchnorm_ind      = enum_value(PS4_Buttons.DownArrow)
        self.lin_cmd_ind        = enum_value(PS4_Triggers.LeftStickYAxis)
        self.ang_cmd_ind        = enum_value(PS4_Triggers.RightStickXAxis)

        self.feat_arr           = { self.raw_ind: FeatureType.RAW,           self.patchnorm_ind: FeatureType.PATCHNORM, 
                                    self.netvlad_ind: FeatureType.NETVLAD,   self.hybridnet_ind: FeatureType.HYBRIDNET }

        self.twist_msg          = Twist()

        # Set up bluetooth connection check variables:
        if self.SIMULATION.get():
            self.controller_ok      = True
        else:
            try:
                self.bus            = pydbus.SystemBus()
                self.adapter        = self.bus.get('org.bluez', '/org/bluez/hci0')
                self.mngr           = self.bus.get('org.bluez', '/')
                self.controller_ok  = self.check_controller()
            except:
                self.print('Unable to establish safe controller connection, exitting.', LogType.FATAL)
                self.print(formatException(), LogType.DEBUG)
                self.exit()

        # Process path data:
        if self.PATH_FILE.get() == '':
            try:
                self.ip                 = VPRDatasetProcessor(self.make_dataset_dict(path=True), try_gen=False, ros=True, printer=self.print)
                self.path_dataset       = copy.deepcopy(self.ip.dataset)
                self.ip.load_dataset(self.make_dataset_dict(path=False))
            except:
                self.print(formatException(), LogType.ERROR)
                self.exit()

            self.plan_path      = self.generate_path(dataset = self.path_dataset)
            self.ref_path       = self.generate_path(dataset = self.ip.dataset)

            self.make_path_from_data()
        else:
            self.make_path_from_file()

    def init_rospy(self):
        super().init_rospy()

        self.time               = rospy.Time.now().to_sec()

        self.path_pub           = self.add_pub(     self.namespace + '/path',       Path,                                       queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.ref_path_pub       = self.add_pub(     self.namespace + '/ref/path',   Path,                                       queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.goal_pub           = self.add_pub(     self.namespace + '/path_goal',  PoseStamped,                                queue_size=1)
        self.speed_pub          = self.add_pub(     self.namespace + '/speeds',     MarkerArray,                                queue_size=1, latch=True)
        self.cmd_pub            = self.add_pub(     self.CMD_TOPIC.get(),           Twist,                                      queue_size=1)
        self.info_pub           = self.add_pub(     self.nodespace + '/info',       ControllerStateInfo,                        queue_size=1)
        self.state_sub          = rospy.Subscriber( self.namespace + '/state',      MonitorDetails,         self.state_cb,      queue_size=1)
        self.robot_odom_sub     = rospy.Subscriber( self.ROBOT_ODOM_TOPIC.get(),    Odometry,               self.robot_odom_cb, queue_size=1) # wheel encoders fused
        self.slam_odom_sub      = rospy.Subscriber( self.SLAM_ODOM_TOPIC.get(),     Odometry,               self.slam_odom_cb,  queue_size=1)
        self.joy_sub            = rospy.Subscriber( self.JOY_TOPIC.get(),           Joy,                    self.joy_cb,        queue_size=1)
        self.timer_chk          = rospy.Timer(rospy.Duration(2), self.check_controller)

        self.sublis.add_operation(self.namespace + '/path',     method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.namespace + '/ref/path', method_sub=self.path_peer_subscribe)

    def param_callback(self, msg):
        self.parameters_ready = False
        if self.params.exists(msg.data):
            if not self.params.update(msg.data):
                self.print("Change to parameter [%s]; bad value." % msg.data, LogType.DEBUG)
        
            else:
                self.print("Change to parameter [%s]; updated." % msg.data, LogType.DEBUG)

                if msg.data == self.LOG_LEVEL.name:
                    set_rospy_log_lvl(self.LOG_LEVEL.get())
                elif msg.data == self.RATE_NUM.name:
                    self.rate_obj = rospy.Rate(self.RATE_NUM.get())
                elif msg.data == self.AUTONOMOUS_OVERRIDE.name:
                    if not self.AUTONOMOUS_OVERRIDE.get() == Bool.UNSET:
                        self.autonomous_cmd = bool(self.AUTONOMOUS_OVERRIDE.get().value)
                    else:
                        self.autonomous_cmd = False
                elif msg.data == self.SAFETY_OVERRIDE.name:
                    if not self.SAFETY_OVERRIDE.get() == Safety_Mode.UNSET:
                        self.safety_mode = self.SAFETY_OVERRIDE.get()
                    else:
                        self.safety_mode = Safety_Mode.STOP

        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)
        self.parameters_ready = True

    def publish_controller_info(self, target_ind):
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
        # Extract (remaining) Monitor Details:
        msg.group.mState            = self.state_msg.mState
        msg.group.prob              = self.state_msg.prob
        msg.group.mStateBin         = self.state_msg.mStateBin
        msg.group.factors           = self.state_msg.factors

        msg.group.safety_mode       = enum_name(self.safety_mode)
        msg.group.autonomous_cmd    = self.autonomous_cmd

        msg.group.current_yaw       = self.vpr_ego[2]
        msg.group.target_yaw        = self.points[target_ind, 2]

        if self.new_slam_ego:
            msg.group.true_yaw          = self.slam_ego[2]
            msg.group.delta_yaw         = self.slam_ego[2] - self.vpr_ego[2]
        self.new_slam_ego = False

        msg.group.lookahead         = self.lookahead
        msg.group.lookahead_mode    = enum_name(self.lookahead_mode)

        self.info_pub.publish(msg)

    def state_cb(self, msg: MonitorDetails):
        
        self.vpr_ego                = [msg.data.vpr_ego.x, msg.data.vpr_ego.y, msg.data.vpr_ego.w]
        self.new_vpr_ego            = True

        if not self.ready:
            return

        self.vpr_ego_hist.append(self.vpr_ego)
        self.state_msg              = msg
        self.new_state_msg          = True
    
    def robot_odom_cb(self, msg: Odometry):
        if not self.ready:
            self.old_robot_ego  = pose2xyw(msg.pose.pose)
            self.new_robot_ego  = True
            return

        self.old_robot_ego          = self.robot_ego
        self.robot_ego              = pose2xyw(msg.pose.pose)
        self.new_robot_ego          = True

    def slam_odom_cb(self, msg: Odometry):
        self.slam_ego               = pose2xyw(msg.pose.pose)
        self.new_slam_ego           = True

    def joy_cb(self, msg: Joy):
        if abs(rospy.Time.now().to_sec() - msg.header.stamp.to_sec()) > 0.5: # if joy message was generated longer ago than half a second:
            self.safety_mode = Safety_Mode.STOP
            if not self.SIMULATION.get():
                self.print("Bad joy data.", LogType.WARN, throttle=5)
            else:
                self.print("Bad joy data.", LogType.DEBUG, throttle=5)
            return # bad data.

        # Toggle enable:
        if msg.buttons[self.enable_ind] > 0:
            if not self.autonomous_cmd:
                self.autonomous_cmd = True
                rospy.logwarn("Autonomous mode: Enabled")
        elif msg.buttons[self.disable_ind] > 0:
            if self.autonomous_cmd:
                self.autonomous_cmd = False
                rospy.loginfo("Autonomous mode: Disabled")

        # Toggle mode:
        try:
            for i in self.feat_arr.keys():
                if msg.buttons[i] and (not self.FEAT_TYPE.get() == self.feat_arr[i]):
                    rospy.set_param(self.namespace + '/feature_type', enum_name(FeatureType.RAW))
                    rospy.loginfo("Switched to %s." % enum_name(self.FEAT_TYPE.get()))
                    break
        except:
            rospy.logdebug_throttle(60, "Param switching is disabled for rosbags :-(")

        # Toggle safety:
        if msg.buttons[self.fast_mode_ind] > 0:
            if not self.safety_mode == Safety_Mode.FAST:
                self.safety_mode = Safety_Mode.FAST
                rospy.logerr('Fast mode enabled.')
        elif msg.buttons[self.slow_mode_ind] > 0:
            if not self.safety_mode == Safety_Mode.SLOW:
                self.safety_mode = Safety_Mode.SLOW
                rospy.logwarn('Slow mode enabled.')
        else:
            if not self.safety_mode == Safety_Mode.STOP:
                self.safety_mode = Safety_Mode.STOP
                rospy.loginfo('Safety released.')

    def path_peer_subscribe(self, topic_name):
        if topic_name == self.namespace + '/path':
            self.plan_path  = self.generate_path(dataset = self.path_dataset)
            self.path_pub.publish(self.plan_path)
        elif topic_name == self.namespace + '/ref/path':
            self.ref_path   = self.generate_path(dataset = self.ip.dataset)
            self.ref_path_pub.publish(self.ref_path)
        else:
            raise Exception('Unknown path_peer_subscribe topic: %s' % str(topic_name))

    def check_controller(self, event=None):
        if self.SIMULATION.get():
            return True
        mngd_objs = self.mngr.GetManagedObjects()
        for path in mngd_objs:
            if mngd_objs[path].get('org.bluez.Device1', {}).get('Connected', False):
                if str(mngd_objs[path].get('org.bluez.Device1', {}).get('Address')) == self.CONTROLLER_MAC.get():
                    return True
        self.print('Bluetooth controller not found! Shutting down.', LogType.FATAL)
        sys.exit()

    def global2local(self, ego):

        Tx  = self.points[:,0] - ego[0]
        Ty  = self.points[:,1] - ego[1]
        R   = np.sqrt(np.power(Tx, 2) + np.power(Ty, 2))
        A   = np.arctan2(Ty, Tx) - ego[2]

        return list(np.multiply(np.cos(A), R)), list(np.multiply(np.sin(A), R))

    def calc_error(self, ego):
        assert len(ego) == 3, 'Input ego vector has an incorrect length.'
        assert self.points.shape[1] == 4, 'Points array has incorrect dimensions.'

        # 1. Global to relative coordinate
        rel_x, rel_y    = self.global2local(ego)

        # 2. Find the nearest waypoint
        distances       = fastdist.matrix_to_matrix_distance(self.points[:,0:3], np.matrix([ego[0], ego[1], ego[2]]), fastdist.euclidean, "euclidean").flatten()
        current_ind     = np.argmin(distances, axis=0)

        # 3. Find target index
        target_ind      = (current_ind + self.lookahead) % self.points.shape[0]

        # 4. Calculate errors
        error_yaw       = np.arctan2(rel_y[target_ind], rel_x[target_ind])
        error_yaw       = normalize_angle(error_yaw) # Normalize angle to [-pi, +pi]
        error_y         = rel_y[target_ind]
        return error_y, error_yaw, current_ind, target_ind

    def make_path_from_file(self):
        self.root           = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/paths'
        with open(self.root + '/' + self.PATH_FILE.get()) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            data = list(reader)
            x_data = data[0]
            y_data = data[1]
            w_data = data[2]
            self.points = np.transpose(np.array([x_data, y_data, w_data]))
            f.close()
        self.make_path()

    def make_path_from_data(self):
        px      = list(self.path_dataset['dataset']['px'])
        py      = list(self.path_dataset['dataset']['py'])
        pw      = list(self.path_dataset['dataset']['pw'])

        self.points = np.transpose(np.array([px, py, pw]))
        self.make_path()

    def make_path(self):
        if self.REVERSE.get():
            self.points[:,2] = list(angle_wrap(np.pi + self.points[:,2], mode='RAD'))
            self.points = np.flipud(self.points)

        dists = np.round(fastdist.matrix_to_matrix_distance(self.points[:,0:2], self.points[:,0:2], fastdist.euclidean, "euclidean"),1)
        _len  = dists.shape[0]
        dists = dists + np.eye(_len)
        inds_to_bin = []
        for i in range(_len):
            if i in inds_to_bin:
                continue
            inds_to_bin += np.argwhere(dists[i,:]<self.PATH_DENSITY.get()).flatten().tolist()
        
        inds_to_bin = np.unique(inds_to_bin).tolist()
        self.points = np.delete(self.points, inds_to_bin, 0)

        # Generate speed profile based on curvature of track:
        points_diff = np.abs(angle_wrap(np.roll(self.points[:,2], 1, 0) - np.roll(self.points[:,2], -1, 0), mode='RAD'))
        k_rad = 5
        points_smooth = np.sum([np.roll(points_diff, i, 0) for i in np.arange(2*k_rad + 1)-k_rad], axis=0)
        speeds = (1 - ((points_smooth - np.min(points_smooth)) / (np.max(points_smooth) - np.min(points_smooth)))) **2

        self.points = np.concatenate([self.points, speeds[:, np.newaxis]], axis=1)

        self.path       = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        self.speeds     = MarkerArray()
        _num            = self.points.shape[0]
        for i in range(_num):
            new_pose                        = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id="map"))
            new_pose.pose.position          = Point(x=self.points[i,0], y=self.points[i,1], z=0.0)
            new_pose.pose.orientation       = q_from_yaw(self.points[i,2])

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

            new_marker.pose.orientation     = yaw

            self.path.poses.append(new_pose)
            self.speeds.markers.append(new_marker)

    def print_display(self):
        if self.autonomous_cmd:
            auto_mode_string = C_I_RED + 'ENABLED' + C_RESET
        else:
            auto_mode_string = C_I_GREEN + 'DISABLED' + C_RESET
        if self.safety_mode == Safety_Mode.STOP:
            safety_mode_string = C_I_GREEN + 'STOPPED' + C_RESET
        elif self.safety_mode == Safety_Mode.SLOW:
            safety_mode_string = C_I_YELLOW + 'SLOW MODE' + C_RESET
        elif self.safety_mode == Safety_Mode.FAST:
            safety_mode_string = C_I_RED + 'FAST MODE' + C_RESET
        else:
            raise Exception('Bad mode state, %s' % str(self.safety_mode))

        base_pos_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 5.2f ' for i in 'xyw']) + C_RESET
        vpr_pos_string  = base_pos_string % tuple(self.vpr_ego)
        slam_pos_string = base_pos_string % tuple(self.slam_ego)
        TAB = ' ' * 8
        lines = [
                 TAB + '-'*20 + C_I_BLUE + ' STATUS INFO ' + C_RESET + '-'*20,
                 TAB + 'Autonomous Mode: %s' % auto_mode_string,
                 TAB + '    Safety Mode: %s' % safety_mode_string,
                 TAB + '   VPR Position: %s' % vpr_pos_string,
                 TAB + '  SLAM Position: %s' % slam_pos_string
                ]
        _lines  = len(lines)
        for line in lines:
            print(C_CLEAR + line)
        print((C_UP_N%1)*_lines + C_CLEAR, end='')

    def print(self, *args, **kwargs):
        arg_list = list(args) + [kwargs[k] for k in kwargs]
        log_level = enum_value(LogType.INFO)
        for i in arg_list:
            if isinstance(i, LogType):
                log_level = enum_value(i)
                break
        if (self.LOG_LEVEL.get() <= log_level) and super().print(*args, **kwargs):
            self.print_lines += 1

    def main(self):
        self.set_state(NodeState.MAIN)

        self.path_pub.publish(self.path)
        self.speed_pub.publish(self.speeds)

        while not (self.new_robot_ego and self.new_vpr_ego):
            self.rate_obj.sleep()
            self.print('Waiting for start position information...', throttle=5)

        self.ready          = True
        self.new_vpr_ego    = False
        self.new_robot_ego  = False

        self.print('Entering main loop.')

        while not rospy.is_shutdown():
            self.loop_contents()

            self.print_display()

    def loop_contents(self):

        if not (self.new_robot_ego and self.new_vpr_ego): # denest
            self.print("Waiting for new position information...", LogType.DEBUG, throttle=10) # print every 10 seconds
            rospy.sleep(0.005)
            return
        
        self.rate_obj.sleep()
        self.new_vpr_ego    = False
        self.new_robot_ego  = False

        error_y, error_yaw, current_ind, target_index = self.calc_error(self.vpr_ego)

        goal                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        goal.pose.position      = Point(x=self.points[current_ind,0], y=self.points[current_ind,1], z=0.0)
        goal.pose.orientation   = q_from_yaw(self.points[current_ind,2])
        self.goal_pub.publish(goal)

        if self.safety_mode == Safety_Mode.SLOW:
            lin_max = self.SLOW_LIN_VEL_MAX.get()
            ang_max = self.SLOW_LIN_VEL_MAX.get()
        elif self.safety_mode == Safety_Mode.FAST:
            lin_max = self.FAST_LIN_VEL_MAX.get()
            ang_max = self.FAST_LIN_VEL_MAX.get()
        else:
            return

        new_msg                 = Twist()
        new_msg.linear.x        = np.min([(0.8 * self.points[current_ind, 3] + 0.4) * lin_max])
        new_msg.angular.z       = np.min([0.7 * error_y * ang_max])

        self.publish_controller_info(target_index)

        if (not self.autonomous_cmd) or (self.safety_mode == Safety_Mode.STOP):
            return
        
        self.cmd_pub.publish(new_msg)

def do_args():
    parser = ap.ArgumentParser(prog="path_follower.py", 
                                description="Path Follower",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='path_follower')

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