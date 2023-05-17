#!/usr/bin/env python3

import rospy
import argparse as ap
import numpy as np
import sys

from std_msgs.msg           import String, ColorRGBA, Header
from geometry_msgs.msg      import Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from aarapsi_robot_pack.msg import ControllerStateInfo

from pyaarapsi.core.argparse_tools          import check_positive_float, check_positive_int, check_bool, check_string, check_positive_two_int_list, check_enum
from pyaarapsi.core.ros_tools               import NodeState, roslogger, LogType, set_rospy_log_lvl, init_node, q_from_yaw
from pyaarapsi.core.helper_tools            import formatException, angle_wrap
from pyaarapsi.core.enum_tools              import enum_name
from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor

'''
Visualiser

Create visualisations that may be time expensive separately
to reduce interruptions to normal pipeline execution.

'''

class mrc():
    def __init__(self, node_name, rate_num, namespace, anon, log_level, reset, order_id=0):

        if not init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30):
            raise Exception('init_node failed.')

        self.init_params(rate_num, log_level, reset)
        self.init_vars()
        self.init_rospy()

        self.main_ready      = True
        self.last_ego        = [0.0, 0.0, 0.0]
        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_params(self, rate_num, log_level, reset):
        self.FEAT_TYPE       = self.ROS_HOME.params.add(self.namespace + "/feature_type",        None,                   lambda x: check_enum(x, FeatureType), force=False)
        self.IMG_DIMS        = self.ROS_HOME.params.add(self.namespace + "/img_dims",            None,                   check_positive_two_int_list,          force=False)
        self.NPZ_DBP         = self.ROS_HOME.params.add(self.namespace + "/npz_dbp",             None,                   check_string,                         force=False)
        self.BAG_DBP         = self.ROS_HOME.params.add(self.namespace + "/bag_dbp",             None,                   check_string,                         force=False)
        self.IMG_TOPIC       = self.ROS_HOME.params.add(self.namespace + "/img_topic",           None,                   check_string,                         force=False)
        self.ODOM_TOPIC      = self.ROS_HOME.params.add(self.namespace + "/odom_topic",          None,                   check_string,                         force=False)
        
        self.REF_BAG_NAME    = self.ROS_HOME.params.add(self.namespace + "/ref/bag_name",        None,                   check_string,                         force=False)
        self.REF_FILTERS     = self.ROS_HOME.params.add(self.namespace + "/ref/filters",         None,                   check_string,                         force=False)
        self.REF_SAMPLE_RATE = self.ROS_HOME.params.add(self.namespace + "/ref/sample_rate",     None,                   check_positive_float,                 force=False) # Hz
        
        self.SVM_MODE        = self.ROS_HOME.params.add(self.nodespace + "/svm_mode",            False,                  check_bool,                           force=reset)
        self.NUM_MARKERS     = self.ROS_HOME.params.add(self.nodespace + "/num_markers",         100,                    check_positive_int,                   force=reset)
        self.LOG_LEVEL       = self.ROS_HOME.params.add(self.nodespace + "/log_level",           log_level,              check_positive_int,                   force=reset)
        self.RATE_NUM        = self.ROS_HOME.params.add(self.nodespace + "/rate",                rate_num,               check_positive_float,                 force=reset)
        
        self.REF_DATA_PARAMS = [self.NPZ_DBP, self.BAG_DBP, self.REF_BAG_NAME, self.REF_FILTERS, self.REF_SAMPLE_RATE, self.IMG_TOPIC, self.ODOM_TOPIC, self.FEAT_TYPE, self.IMG_DIMS]
        self.REF_DATA_NAMES  = [i.name for i in self.REF_DATA_PARAMS]

    def init_vars(self):
        self.control_msg     = None
        self.new_control_msg = False
        self.markers         = MarkerArray()
        self.marker_id       = 0
        self.colour_good     = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.80)
        self.colour_svm_good = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.20)
        self.colour_bad      = ColorRGBA(r=0.9, g=0.2, b=0.1, a=0.20)
        self.colour_lost     = ColorRGBA(r=0.1, g=0.1, b=0.1, a=0.08)
        
        try:
            # Process reference data
            dataset_dict            = self.make_dataset_dict()
            self.image_processor    = VPRDatasetProcessor(dataset_dict, try_gen=False, ros=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

    def init_rospy(self):
        self.rate_obj        = rospy.Rate(self.RATE_NUM.get())
        self.param_sub       = rospy.Subscriber(self.namespace + "/params_update",   String,              self.param_callback,   queue_size=100)
        self.control_sub     = rospy.Subscriber(self.namespace + "/follower/info",   ControllerStateInfo, self.control_callback, queue_size=1)
        self.confidence_pub  = self.ROS_HOME.add_pub(self.namespace + '/confidence', MarkerArray,                                queue_size=1)

    def make_dataset_dict(self):
        return dict(bag_name=self.REF_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                    odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.REF_SAMPLE_RATE.get(), \
                    ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')

    def control_callback(self, msg):
        self.control_msg     = msg
        self.new_control_msg = True

    def update_VPR(self):
        dataset_dict = self.make_dataset_dict()
        if not self.image_processor.swap(dataset_dict, generate=False, allow_false=True):
            self.print("VPR reference data swap failed. Previous set will be retained (changed ROS parameter will revert)", LogType.WARN)
            return False
        else:
            self.print("VPR reference data swapped.", LogType.INFO)
            return True

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

                ref_data_comp   = [i == msg.data for i in self.REF_DATA_NAMES]
                try:
                    param = np.array(self.REF_DATA_PARAMS)[ref_data_comp][0]
                    self.print("Change to VPR reference data parameters detected.", LogType.WARN)
                    if not self.update_VPR():
                        #param.revert()
                        pass
                except IndexError:
                    pass
                except:
                    param.revert()
                    self.print(formatException(), LogType.ERROR)
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)
        self.parameters_ready = True

    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.loop_contents()

    def make_control_visualisation(self):
        # Generate / record statistics:
        gt_ego              = self.last_ego
        vpr_ego             = [self.control_msg.group.vpr_ego.x, self.control_msg.group.vpr_ego.y, self.control_msg.group.vpr_ego.w]
        err_ego             = [gt_ego[0] - vpr_ego[0], gt_ego[1] - vpr_ego[1], angle_wrap(gt_ego[2] - vpr_ego[2], 'RAD')]

        gt_ind              = self.control_msg.group.trueId
        vpr_ind             = self.control_msg.group.matchId
        err_id              = abs(gt_ind - vpr_ind)

        in_gt_tolerance     = self.control_msg.group.gt_state > 0
        gt_error            = self.control_msg.group.gt_error
        in_svm_tolerance    = self.control_msg.group.mStateBin
        auto_mode           = self.control_msg.group.safety_states.autonomous

        svm_prob            = self.control_msg.group.prob
        svm_zvalue          = self.control_msg.group.mState

        target_yaw          = self.control_msg.group.target_yaw
        current_yaw         = self.control_msg.group.current_yaw
        err_yaw             = angle_wrap(gt_ego[2] - current_yaw)

        if self.SVM_MODE.get():
            if in_svm_tolerance:
                colour = self.colour_svm_good
            else:
                colour = self.colour_bad
            scale = abs(svm_zvalue)
        else:
            if in_gt_tolerance:
                colour = self.colour_good
                scale  = 0.2
            elif gt_error < 2:
                colour = self.colour_bad
                scale = gt_error + 0.01
            else:
                colour = self.colour_lost
                scale = 2


        new_marker                      = Marker()
        new_marker.header               = Header(stamp=rospy.Time.now(), frame_id='map')
        new_marker.type                 = new_marker.SPHERE
        new_marker.action               = new_marker.ADD
        new_marker.id                   = self.marker_id
        new_marker.color                = colour
        new_marker.scale                = Vector3(x=scale, y=scale, z=0.01)

        new_marker.pose.position        = Point(x=gt_ego[0], y=gt_ego[1])
        new_marker.pose.orientation     = q_from_yaw(gt_ego[2])
        self.markers.markers.append(new_marker)

        self.marker_id = (self.marker_id + 1) % self.NUM_MARKERS.get()
        while len(self.markers.markers) > self.NUM_MARKERS.get():
            self.markers.markers.pop(0)

        self.confidence_pub.publish(self.markers)


    def loop_contents(self):
        if not (self.new_control_msg):
            self.print("Waiting.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return # denest
        self.rate_obj.sleep()

        if self.new_control_msg:
            new_ego = [self.control_msg.group.gt_ego.x,  self.control_msg.group.gt_ego.y,  self.control_msg.group.gt_ego.w]
            if np.sqrt(np.sum(np.square(np.array(new_ego) - np.array(self.last_ego)))) > 0.02:
                self.make_control_visualisation()
            self.last_ego = new_ego

        self.new_control_msg = False

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
    parser = ap.ArgumentParser(prog="visualiser.py", 
                            description="ROS Visualiser Node",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="visualiser",     help="Specify node name (default: %(default)s).")
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
        nmrc.print("Initialisation complete. Generating visualisations...")
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