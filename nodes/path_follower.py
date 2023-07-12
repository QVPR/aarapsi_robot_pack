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
import cv2
import warnings
import matplotlib.pyplot as plt

from enum                   import Enum

from scipy.interpolate      import splprep, splev

from nav_msgs.msg           import Path, Odometry
from std_msgs.msg           import Header, ColorRGBA, String
from geometry_msgs.msg      import PoseStamped, Point, Twist, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg        import Joy
from aarapsi_robot_pack.msg import ControllerStateInfo, MonitorDetails

from pyaarapsi.core.argparse_tools          import check_positive_float, check_bool, check_string, check_float_list, check_enum, check_positive_int, check_float
from pyaarapsi.core.ros_tools               import NodeState, roslogger, LogType, q_from_yaw, pose2xyw, compressed2np, np2compressed, q_from_rpy
from pyaarapsi.core.helper_tools            import formatException, angle_wrap, normalize_angle, m2m_dist
from pyaarapsi.core.enum_tools              import enum_name, enum_value
from pyaarapsi.core.vars                    import C_I_RED, C_I_GREEN, C_I_YELLOW, C_I_BLUE, C_I_WHITE, C_RESET, C_CLEAR, C_UP_N, C_DOWN_N
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
    Triangle        = 2
    Square          = 3
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

class Command_Mode(Enum):
    UNSET           = -1
    STOP            = 0
    VPR             = 1
    SLAM            = 2
    ZONE_RETURN     = 3

class Lookahead_Mode(Enum):
    INDEX           = 0
    DISTANCE        = 1

class Return_Stage(Enum):
    UNSET           = 0
    DIST            = 1
    TURN            = 2
    DONE            = 3

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_params(self, rate_num: float, log_level: float, reset):
        super().init_params(rate_num, log_level, reset)

        self.PATH_DENSITY           = self.params.add(self.namespace + "/path/density",             None,               check_positive_float,                   force=False)
        self.ZONE_LENGTH            = self.params.add(self.namespace + "/path/zones/length",        None,               check_positive_int,                     force=False)
        self.ZONE_NUMBER            = self.params.add(self.namespace + "/path/zones/number",        None,               check_positive_int,                     force=False)
        self.SLOW_LIN_VEL_MAX       = self.params.add(self.namespace + "/limits/slow/linear",       None,               check_positive_float,                   force=False)
        self.SLOW_ANG_VEL_MAX       = self.params.add(self.namespace + "/limits/slow/angular",      None,               check_positive_float,                   force=False)
        self.FAST_LIN_VEL_MAX       = self.params.add(self.namespace + "/limits/fast/linear",       None,               check_positive_float,                   force=False)
        self.FAST_ANG_VEL_MAX       = self.params.add(self.namespace + "/limits/fast/angular",      None,               check_positive_float,                   force=False)
        self.COR_OFFSET             = self.params.add(self.namespace + "/cor_offset",               0.045,              check_float,                            force=False)
        self.CONTROLLER_MAC         = self.params.add(self.namespace + "/controller_mac",           None,               check_string,                           force=False)
        self.JOY_TOPIC              = self.params.add(self.namespace + "/joy_topic",                None,               check_string,                           force=False)
        self.CMD_TOPIC              = self.params.add(self.namespace + "/cmd_topic",                None,               check_string,                           force=False)
        self.ROBOT_ODOM_TOPIC       = self.params.add(self.namespace + "/robot_odom_topic",         None,               check_string,                           force=False)
        self.VPR_ODOM_TOPIC         = self.params.add(self.namespace + "/vpr_odom_topic",           None,               check_string,                           force=False)

        self.PRINT_DISPLAY          = self.params.add(self.nodespace + "/print_display",            True,               check_bool,                             force=reset)
        self.USE_NOISE              = self.params.add(self.nodespace + "/noise/enable",             False,              check_bool,                             force=reset)
        self.NOISE_VALS             = self.params.add(self.nodespace + "/noise/vals",               [0.1]*3,            lambda x: check_float_list(x, 3),       force=reset)
        self.PUB_INFO               = self.params.add(self.nodespace + "/publish_info",             True,               check_bool,                             force=reset)
        self.SVM_OVERRIDE           = self.params.add(self.nodespace + "/svm_override",             False,              check_bool,                             force=reset)
        self.REVERSE                = self.params.add(self.nodespace + "/reverse",                  False,              check_bool,                             force=reset)
        self.SAFETY_OVERRIDE        = self.params.add(self.nodespace + "/override/safety",          Safety_Mode.UNSET,  lambda x: check_enum(x, Safety_Mode),   force=reset)
        self.AUTONOMOUS_OVERRIDE    = self.params.add(self.nodespace + "/override/autonomous",      Command_Mode.UNSET, lambda x: check_enum(x, Command_Mode),  force=reset)

    def init_vars(self):
        super().init_vars()

        self.vpr_ego            = []
        self.vpr_ego_hist       = []
        self.slam_ego           = []
        self.robot_ego          = []
        self.old_robot_ego      = []
        self.lookahead          = 4
        self.lookahead_mode     = Lookahead_Mode.INDEX
        self.dt                 = 1/self.RATE_NUM.get()
        self.print_lines        = 0

        self.old_linear         = 0.0
        self.old_angular        = 0.0
        self.zone_index         = None
        self.return_stage       = Return_Stage.UNSET

        self.plan_path          = Path()
        self.ref_path           = Path()
        self.state_msg          = MonitorDetails()

        self.ready              = False
        self.new_state_msg      = False
        self.new_robot_ego      = False
        self.new_slam_ego       = False
        self.new_vpr_ego        = False

        self.command_mode       = Command_Mode.STOP

        self.safety_mode        = Safety_Mode.STOP

        self.vpr_mode_ind       = enum_value(PS4_Buttons.Square)
        self.stop_mode_ind      = enum_value(PS4_Buttons.X)
        self.slam_mode_ind      = enum_value(PS4_Buttons.O)
        self.zone_mode_ind      = enum_value(PS4_Buttons.Triangle)

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
        try:
            self.ip                 = VPRDatasetProcessor(self.make_dataset_dict(path=True), try_gen=False, ros=True, printer=self.print)
            self.path_dataset       = copy.deepcopy(self.ip.dataset)
            self.ip.load_dataset(self.make_dataset_dict(path=False))
            if not self.FEAT_TYPE.get() == FeatureType.RAW:
                self.ip.extend_dataset(FeatureType.RAW, try_gen=True, save=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()        

        self.plan_path, self.plan_speeds, self.zones = self.make_path()
        self.ref_path = self.generate_path(dataset = self.ip.dataset)

    def init_rospy(self):
        super().init_rospy()

        self.time               = rospy.Time.now().to_sec()

        self.path_pub           = self.add_pub(     self.namespace + '/path',       Path,                                       queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.ref_path_pub       = self.add_pub(     self.namespace + '/ref/path',   Path,                                       queue_size=1, latch=True, subscriber_listener=self.sublis)
        self.COR_pub            = self.add_pub(     self.namespace + '/cor',        PoseStamped,                                queue_size=1)
        self.goal_pub           = self.add_pub(     self.namespace + '/path_goal',  PoseStamped,                                queue_size=1)
        self.speed_pub          = self.add_pub(     self.namespace + '/speeds',     MarkerArray,                                queue_size=1, latch=True)
        self.zones_pub          = self.add_pub(     self.namespace + '/zones',      MarkerArray,                                queue_size=1, latch=True)
        self.cmd_pub            = self.add_pub(     self.CMD_TOPIC.get(),           Twist,                                      queue_size=1)
        self.info_pub           = self.add_pub(     self.nodespace + '/info',       ControllerStateInfo,                        queue_size=1)
        self.state_sub          = rospy.Subscriber( self.namespace + '/state',      MonitorDetails,         self.state_cb,      queue_size=1)
        self.robot_odom_sub     = rospy.Subscriber( self.ROBOT_ODOM_TOPIC.get(),    Odometry,               self.robot_odom_cb, queue_size=1) # wheel encoders fused
        self.slam_odom_sub      = rospy.Subscriber( self.SLAM_ODOM_TOPIC.get(),     Odometry,               self.slam_odom_cb,  queue_size=1)
        self.joy_sub            = rospy.Subscriber( self.JOY_TOPIC.get(),           Joy,                    self.joy_cb,        queue_size=1)
        self.timer_chk          = rospy.Timer(rospy.Duration(2), self.check_controller)

        self.sublis.add_operation(self.namespace + '/path',     method_sub=self.path_peer_subscribe)
        self.sublis.add_operation(self.namespace + '/ref/path', method_sub=self.path_peer_subscribe)

    def param_helper(self, msg: String):
        if msg.data == self.AUTONOMOUS_OVERRIDE.name:
            if not self.AUTONOMOUS_OVERRIDE.get() == Command_Mode.UNSET:
                self.command_mode = self.AUTONOMOUS_OVERRIDE.get()
                self.return_stage = Return_Stage.UNSET
            else:
                self.command_mode = Command_Mode.STOP
        elif msg.data == self.SAFETY_OVERRIDE.name:
            if not self.SAFETY_OVERRIDE.get() == Safety_Mode.UNSET:
                self.safety_mode = self.SAFETY_OVERRIDE.get()
            else:
                self.safety_mode = Safety_Mode.STOP

    def publish_controller_info(self, target_yaw: float):
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
        msg.group.command_mode      = enum_name(self.command_mode)

        msg.group.current_yaw       = self.vpr_ego[2]
        msg.group.target_yaw        = target_yaw

        if self.new_slam_ego:
            msg.group.true_yaw      = self.slam_ego[2]
            msg.group.delta_yaw     = self.slam_ego[2] - self.vpr_ego[2]
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
        if not self.ready:
            return
        
        if abs(rospy.Time.now().to_sec() - msg.header.stamp.to_sec()) > 0.5: # if joy message was generated longer ago than half a second:
            self.safety_mode = Safety_Mode.STOP
            if not self.SIMULATION.get():
                self.print("Bad joy data.", LogType.WARN, throttle=5)
            else:
                self.print("Bad joy data.", LogType.DEBUG, throttle=5)
            return # bad data.

        # Toggle command mode:
        if msg.buttons[self.slam_mode_ind] > 0:
            if not self.command_mode == Command_Mode.SLAM:
                self.command_mode = Command_Mode.SLAM
                self.print("Autonomous Commands: SLAM", LogType.WARN)
        elif msg.buttons[self.vpr_mode_ind] > 0:
            if not self.command_mode == Command_Mode.VPR:
                self.command_mode = Command_Mode.VPR
                self.print("Autonomous Commands: VPR", LogType.ERROR)
        elif msg.buttons[self.zone_mode_ind] > 0:
            if not self.command_mode == Command_Mode.ZONE_RETURN:
                self.command_mode = Command_Mode.ZONE_RETURN
                self.zone_index   = None
                self.return_stage = Return_Stage.UNSET
                self.print("Autonomous Commands: Zone Reset", LogType.WARN)
        elif msg.buttons[self.stop_mode_ind] > 0:
            if not self.command_mode == Command_Mode.STOP:
                self.command_mode = Command_Mode.STOP
                self.print("Autonomous Commands: Disabled", LogType.INFO)

        # Toggle feature type:
        try:
            for i in self.feat_arr.keys():
                if msg.buttons[i] and (not self.FEAT_TYPE.get() == self.feat_arr[i]):
                    rospy.set_param(self.namespace + '/feature_type', enum_name(FeatureType.RAW))
                    self.print("Switched to %s." % enum_name(self.FEAT_TYPE.get()), LogType.INFO)
                    break
        except:
            self.print("Param switching is disabled for rosbags :-(", LogType.WARN, throttle=60)

        # Toggle speed safety mode:
        if msg.buttons[self.fast_mode_ind] > 0:
            if not self.safety_mode == Safety_Mode.FAST:
                self.safety_mode = Safety_Mode.FAST
                self.print('Fast mode enabled.', LogType.ERROR)
        elif msg.buttons[self.slow_mode_ind] > 0:
            if not self.safety_mode == Safety_Mode.SLOW:
                self.safety_mode = Safety_Mode.SLOW
                self.print('Slow mode enabled.', LogType.WARN)
        else:
            if not self.safety_mode == Safety_Mode.STOP:
                self.safety_mode = Safety_Mode.STOP
                self.print('Safety released.', LogType.INFO)

    def path_peer_subscribe(self, topic_name):
        if topic_name == self.namespace + '/path':
            self.path_pub.publish(self.plan_path)
            self.speed_pub.publish(self.plan_speeds)
            self.zones_pub.publish(self.zones)
        elif topic_name == self.namespace + '/ref/path':
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

        Tx  = self.path_xyws[:,0] - ego[0]
        Ty  = self.path_xyws[:,1] - ego[1]
        R   = np.sqrt(np.power(Tx, 2) + np.power(Ty, 2))
        A   = np.arctan2(Ty, Tx) - ego[2]

        return list(np.multiply(np.cos(A), R)), list(np.multiply(np.sin(A), R))
    
    def calc_current_ind(self, ego):
        current_ind         = np.argmin(m2m_dist(self.path_xyws[:,0:2], ego[0:2], True), axis=0)
        zone                = np.max(np.arange(self.num_zones)[np.array(self.zone_indices[0:-1]) <= current_ind] + 1)
        return current_ind, zone

    def calc_errors(self, ego, target_ind: int = None):
        rel_x, rel_y    = self.global2local(ego)
        error_yaw       = normalize_angle(np.arctan2(rel_y[target_ind], rel_x[target_ind]))
        error_y         = rel_y[target_ind]
        errors          = {'error_yaw': error_yaw, 'error_y': error_y}
        return errors
    
    def calc_vpr_errors(self, ego, current_ind):
        target_ind      = (current_ind + self.lookahead) % self.path_xyws.shape[0]
        error_v         = self.path_xyws[current_ind, 3]
        errors          = self.calc_errors(ego, target_ind)
        errors.update({'error_v': error_v})
        return errors, target_ind

    def make_path(self):
        self.vpr_points = np.transpose(np.stack([   self.path_dataset['dataset']['px'].flatten(), 
                                                    self.path_dataset['dataset']['py'].flatten(), 
                                                    self.path_dataset['dataset']['pw'].flatten()]))

        if self.REVERSE.get():
            self.vpr_points[:,2] = list(angle_wrap(np.pi + self.vpr_points[:,2], mode='RAD'))
            self.vpr_points = np.flipud(self.vpr_points)

        vpr_path_distances  = np.sqrt( \
                                np.square(self.vpr_points[:,0] - np.roll(self.vpr_points[:,0], 1)) + \
                                np.square(self.vpr_points[:,1] - np.roll(self.vpr_points[:,1], 1)) \
                            )
        vpr_path_length     = np.sum(vpr_path_distances)
        num_interp_points   = int(vpr_path_length / self.PATH_DENSITY.get())

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')#, r'/usr/lib/python2.7/dist-packages/scipy/interpolate/_fitpack_impl.py:227')
            rolled_xy = np.transpose(np.roll(self.vpr_points[:,0:2], int(self.vpr_points.shape[0]/2), 0))
            tck, u = splprep(rolled_xy, u=None, s=1.0, per=1) 
            u_dense = np.linspace(u.min(), u.max(), num_interp_points + 1)
            x_interp, y_interp = splev(u_dense, tck, der=0)
            x_interp = np.delete(x_interp, -1)
            y_interp = np.delete(y_interp, -1)
            w_interp = np.arctan2(y_interp - np.roll(y_interp, 1), x_interp - np.roll(x_interp, 1))

        # Generate speed profile based on curvature of track:
        points_diff     = np.abs(angle_wrap(np.roll(w_interp, 1, 0) - np.roll(w_interp, -1, 0), mode='RAD'))
        points_smooth   = np.sum([np.roll(points_diff, i, 0) for i in np.arange(11)-5], axis=0)
        s_interp        = (1 - ((points_smooth - np.min(points_smooth)) / (np.max(points_smooth) - np.min(points_smooth)))) **2
        s_interp[s_interp<np.mean(s_interp)/2] = np.mean(s_interp)/2
            
        self.path_xyws  = np.transpose(np.stack([x_interp, y_interp, w_interp, s_interp]))

        path2vpr_distances = m2m_dist(self.path_xyws[:,0:2], self.vpr_points[:,0:2])
        self.path2vpr_inds = np.argmin(path2vpr_distances, 0)
        self.vpr2path_inds = np.argmin(path2vpr_distances, 1)

        plan_path_distances = np.sqrt( \
                                np.square(self.path_xyws[:,0] - np.roll(self.path_xyws[:,0], 1)) + \
                                np.square(self.path_xyws[:,1] - np.roll(self.path_xyws[:,1], 1)) \
                            )
        plan_path_length    = np.sum(plan_path_distances)
        if plan_path_length / self.ZONE_LENGTH.get() > self.ZONE_NUMBER.get():
            self.num_zones = int(self.ZONE_NUMBER.get())
        else:
            self.num_zones = int(plan_path_length / self.ZONE_LENGTH.get())
        _len = self.path_xyws.shape[0]
        self.zone_indices = [int(((_len - self.num_zones) / self.num_zones) * i + np.floor(i)) for i in range(self.num_zones + 1)]

        path       = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        speeds     = MarkerArray()
        zones      = MarkerArray()
        for i in range(self.path_xyws.shape[0]):
            new_pose                        = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id="map"))
            new_pose.pose.position          = Point(x=self.path_xyws[i,0], y=self.path_xyws[i,1], z=0.0)
            new_pose.pose.orientation       = q_from_yaw(self.path_xyws[i,2])

            new_speed                       = Marker(header=Header(stamp=rospy.Time.now(), frame_id='map'))
            new_speed.type                  = new_speed.ARROW
            new_speed.action                = new_speed.ADD
            new_speed.id                    = i
            new_speed.color                 = ColorRGBA(r=0.859, g=0.094, b=0.220, a=0.5)
            new_speed.scale                 = Vector3(x=self.path_xyws[i,3], y=0.05, z=0.05)
            new_speed.pose.position         = Point(x=self.path_xyws[i,0], y=self.path_xyws[i,1], z=0.0)
            if not self.REVERSE.get():
                yaw = q_from_yaw(self.path_xyws[i,2] + np.pi/2)
            else:
                yaw = q_from_yaw(self.path_xyws[i,2] - np.pi/2)

            new_speed.pose.orientation      = yaw
            if i in self.zone_indices:
                new_zone                    = Marker(header=Header(stamp=rospy.Time.now(), frame_id='map'))
                new_zone.type               = new_zone.CUBE
                new_zone.action             = new_zone.ADD
                new_zone.id                 = i
                new_zone.color              = ColorRGBA(r=1.000, g=0.616, b=0.000, a=0.5)
                new_zone.scale              = Vector3(x=0.0, y=1.0, z=1.0)
                new_zone.pose.position      = Point(x=self.path_xyws[i,0], y=self.path_xyws[i,1], z=0.0)
                new_zone.pose.orientation   = new_pose.pose.orientation

            path.poses.append(new_pose)
            speeds.markers.append(new_speed)
            zones.markers.append(new_zone)

        return path, speeds, zones

    def path2vpr(self, path_ind):
        return self.path2vpr_inds[path_ind]

    def vpr2path(self, vpr_ind):
        return self.vpr2path_inds[vpr_ind]

    def print_display(self, new_linear, new_angular, current_ind, error_v, error_y, error_yaw, zone):

        if self.command_mode == Command_Mode.STOP:
            command_mode_string = C_I_GREEN + 'STOPPED' + C_RESET
        elif self.command_mode == Command_Mode.SLAM:
            command_mode_string = C_I_YELLOW + 'SLAM mode' + C_RESET
        elif self.command_mode == Command_Mode.ZONE_RETURN:
            command_mode_string = C_I_YELLOW + 'Resetting zone...' + C_RESET
        elif self.command_mode == Command_Mode.VPR:
            command_mode_string = C_I_RED + 'VPR MODE' + C_RESET
        else:
            raise Exception('Bad command mode state, %s' % str(self.command_mode))

        if self.safety_mode == Safety_Mode.STOP:
            safety_mode_string = C_I_GREEN + 'STOPPED' + C_RESET
        elif self.safety_mode == Safety_Mode.SLOW:
            safety_mode_string = C_I_YELLOW + 'SLOW mode' + C_RESET
        elif self.safety_mode == Safety_Mode.FAST:
            safety_mode_string = C_I_RED + 'FAST mode' + C_RESET
        else:
            raise Exception('Bad safety mode state, %s' % str(self.safety_mode))

        base_pos_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 5.2f ' for i in 'xyw']) + C_RESET
        base_vel_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 5.2f ' for i in ['LIN','ANG']]) + C_RESET
        base_err_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '% 5.2f ' for i in ['VEL', 'C-T','YAW']]) + C_RESET
        base_ind_string = ''.join([C_I_YELLOW + i + ': ' + C_I_WHITE + '%4d ' for i in ['CUR']]) + C_RESET
        vpr_pos_string  = base_pos_string % tuple(self.vpr_ego)
        slam_pos_string = base_pos_string % tuple(self.slam_ego)
        speed_string    = base_vel_string % (new_linear, new_angular)
        errors_string   = base_err_string % (error_v, error_y, error_yaw)
        index_string    = base_ind_string % (current_ind)
        TAB = ' ' * 8
        lines = [
                 '',
                 TAB + '-'*20 + C_I_BLUE + ' STATUS INFO ' + C_RESET + '-'*20,
                 TAB + 'Autonomous Mode: %s' % command_mode_string,
                 TAB + '    Safety Mode: %s' % safety_mode_string,
                 TAB + '   VPR Position: %s' % vpr_pos_string,
                 TAB + '  SLAM Position: %s' % slam_pos_string,
                 TAB + ' Speed Commands: %s' % speed_string,
                 TAB + '         Errors: %s' % errors_string,
                 TAB + '    Zone Number: %d' % zone,
                 TAB + '     Index Info: %s' % index_string
                ]
        print(''.join([C_CLEAR + line + '\n' for line in lines]) + (C_UP_N%1)*(len(lines)), end='')

    def print(self, *args, **kwargs):
        arg_list = list(args) + [kwargs[k] for k in kwargs]
        log_level = enum_value(LogType.INFO)
        for i in arg_list:
            if isinstance(i, LogType):
                log_level = enum_value(i)
                break
        if (enum_value(self.LOG_LEVEL.get()) <= log_level) and super().print(*args, **kwargs):
            self.print_lines += 1

    def main(self):
        # Main loop process
        self.set_state(NodeState.MAIN)

        self.path_pub.publish(self.plan_path)
        self.speed_pub.publish(self.plan_speeds)

        while not (self.new_robot_ego and self.new_vpr_ego):
            self.rate_obj.sleep()
            self.print('Waiting for start position information...', throttle=5)
            if rospy.is_shutdown():
                return

        self.ready          = True
        self.new_vpr_ego    = False
        self.new_robot_ego  = False

        self.print('Entering main loop.')

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

    def roll_match(self, resize: list = [360,8]):
        img_dims        = self.IMG_DIMS.get()
        query_raw       = cv2.cvtColor(compressed2np(self.state_msg.queryImage), cv2.COLOR_BGR2GRAY)
        image_to_align  = cv2.resize(query_raw, resize)
        against_image   = cv2.resize(np.reshape(self.ip.dataset['dataset']['RAW'][self.state_msg.data.matchId], [img_dims[1], img_dims[0]]), resize)
        options_stacked = np.stack([np.roll(against_image, i, 1).flatten() for i in range(against_image.shape[1])])
        matches         = m2m_dist(image_to_align.flatten()[np.newaxis, :], options_stacked, True)
        yaw_fix_deg     = np.argpartition(matches, 1)[0:1][0]
        yaw_fix_rad     = yaw_fix_deg * np.pi / 180.0
        self.print([query_raw.shape, image_to_align.shape, self.ip.dataset['dataset']['RAW'][self.state_msg.data.matchId].shape, against_image.shape])
        return yaw_fix_rad

    def update_position(self, curr_ind: int) -> None:
        goal                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        goal.pose.position      = Point(x=self.path_xyws[curr_ind,0], y=self.path_xyws[curr_ind,1], z=0.0)
        goal.pose.orientation   = q_from_yaw(self.path_xyws[curr_ind,2])
        self.goal_pub.publish(goal)

    def make_new_command(self, error_v: float, error_y: float, error_yaw: float, override_svm: bool = False) -> Twist:
        if self.safety_mode == Safety_Mode.SLOW:
            lin_max = self.SLOW_LIN_VEL_MAX.get()
            ang_max = self.SLOW_LIN_VEL_MAX.get()
        elif self.safety_mode == Safety_Mode.FAST:
            lin_max = self.FAST_LIN_VEL_MAX.get()
            ang_max = self.FAST_LIN_VEL_MAX.get()
        else:
            lin_max = 0
            ang_max = 0
        
        if self.state_msg.mStateBin or override_svm:
            new_linear          = np.sign(error_v)   * np.min([abs(error_v),   lin_max])
            new_angular         = np.sign(error_yaw) * np.min([abs(error_yaw), ang_max])
        else:
            new_linear          = self.old_linear / 2
            new_angular         = self.old_angular / 2

        self.old_linear         = new_linear
        self.old_angular        = new_angular

        new_msg                 = Twist()
        new_msg.linear.x        = new_linear
        new_msg.angular.z       = new_angular
        return new_msg

    def path_follow(self, ego, current_ind, override_svm):
        if self.command_mode in [Command_Mode.VPR, Command_Mode.SLAM]:
            errors, target_ind = self.calc_vpr_errors(ego, current_ind)

            self.update_position(current_ind)
            new_msg = self.make_new_command(override_svm=override_svm, **errors)
            new_linear = new_msg.linear.x
            new_angular = new_msg.angular.z

            self.publish_controller_info(self.path_xyws[target_ind,2])

            if (not self.command_mode == Command_Mode.STOP) and (not self.safety_mode == Safety_Mode.STOP):
                self.cmd_pub.publish(new_msg)

        else:
            new_linear  = 0
            new_angular = 0
            errors      = {'error_yaw': 0, 'error_y': 0, 'error_v': 0}

        return new_linear, new_angular, errors
    
    def zone_return(self, ego, current_ind):
        if self.return_stage == Return_Stage.DONE:
            return
        
        if self.return_stage == Return_Stage.UNSET:
            self.zone_index  = self.zone_indices[np.argmin(m2m_dist(current_ind, np.transpose(np.matrix(self.zone_indices))))]
            self.return_stage = Return_Stage.DIST

        self.update_position(self.zone_index)

        errors      = self.calc_errors(ego, target_ind=self.zone_index)
        ego_cor     = self.update_COR(ego)
        dist        = np.sqrt(np.square(ego_cor[0]-self.path_xyws[self.zone_index, 0]) + np.square(ego_cor[1]-self.path_xyws[self.zone_index, 1]))
        yaw_err     = errors.pop('error_yaw')
        head_err    = self.path_xyws[self.zone_index, 2] - ego[2]

        if self.return_stage == Return_Stage.DIST:
            if abs(yaw_err) < np.pi/6:
                ang_err = np.sign(yaw_err) * np.max([0, -0.19*abs(yaw_err)**2 + 0.4*abs(yaw_err) - 0.007])
                lin_err = np.max([0.1, -(1/3)*dist**2 + (19/30)*dist - 0.06]) 
            else:
                lin_err = 0
                ang_err = np.sign(yaw_err) * 0.2

            if dist < 0.1:
                self.return_stage = Return_Stage.TURN

        elif self.return_stage == Return_Stage.TURN:
            lin_err = 0
            if abs(head_err) < np.pi/180:
                self.command_mode = Command_Mode.STOP
                self.return_stage = Return_Stage.DONE
                self.AUTONOMOUS_OVERRIDE.set(Command_Mode.UNSET)
                ang_err = 0
            else:
                ang_err = head_err
        else:
            raise Exception('Bad return stage [%s].' % str(self.return_stage))

        new_msg     = self.make_new_command(override_svm=True, error_v=lin_err, error_yaw=ang_err, error_y=errors.pop('error_y'))
        self.cmd_pub.publish(new_msg)

    def update_COR(self, ego):
        COR_x = ego[0] + self.COR_OFFSET.get() * np.cos(ego[2])
        COR_y = ego[1] + self.COR_OFFSET.get() * np.sin(ego[2])

        pose                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        pose.pose.position      = Point(x=COR_x, y=COR_y)
        pose.pose.orientation   = q_from_rpy(0, -np.pi/2, 0)
        self.COR_pub.publish(pose)

        return [COR_x, COR_y]

    def loop_contents(self):

        if not (self.new_robot_ego and self.new_vpr_ego): # denest
            self.print("Waiting for new position information...", LogType.DEBUG, throttle=10) # print every 10 seconds
            rospy.sleep(0.005)
            return
        
        self.rate_obj.sleep()
        self.new_vpr_ego    = False
        self.new_robot_ego  = False

        if self.command_mode == Command_Mode.VPR:
            rm_corr             = self.roll_match()
            heading_fixed       = normalize_angle(angle_wrap(self.vpr_ego[2] - rm_corr, 'RAD'))
            self.print({'vpr': self.vpr_ego[2], 'slam': self.slam_ego[2], 'corr': rm_corr, 'true_corr': self.slam_ego[2] - self.vpr_ego[2]})
            ego                 = [self.vpr_ego[0], self.vpr_ego[1], heading_fixed]
            override_svm        = False
        else:
            ego                 = self.slam_ego
            override_svm        = True
            
        if self.SVM_OVERRIDE.get():
            override_svm        = True

        current_ind, zone       = self.calc_current_ind(ego)
        lin_cmd, ang_cmd, errs  = self.path_follow(ego, current_ind, override_svm)

        if self.command_mode == Command_Mode.ZONE_RETURN:
            self.zone_return(ego, current_ind)

        if self.PRINT_DISPLAY.get():
            self.print_display(new_linear=lin_cmd, new_angular=ang_cmd, current_ind=current_ind, zone=zone, **errs)

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