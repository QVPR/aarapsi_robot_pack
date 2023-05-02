#!/usr/bin/env python3

import rospy
import rospkg
import argparse as ap
import sys
import os
from std_msgs.msg import String
from pyaarapsi.core.argparse_tools import check_positive_float, check_positive_int, check_bool, check_string, check_positive_two_int_tuple, check_enum
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, ROS_Home, set_rospy_log_lvl
from pyaarapsi.core.helper_tools import formatException
from pyaarapsi.core.enum_tools import enum_name, enum_value_options, enum_get

from pyaarapsi.vpr_simple.imageprocessor_helpers import FeatureType, ViewMode
from pyaarapsi.vpr_simple.svm_model_tool         import SVMModelProcessor

'''

ROS SVM Trainer Node

This node subscribes to the same parameter updates as the vpr_monitor node.
It performs the same logic, however it triggers training when no model exists
that matches the required parameters. This enables training to be performed
in a separate thread to the monitor, allowing for fewer interruptions to the
monitor's performance and activities.

'''

class mrc():
    def __init__(self, node_name, rate, namespace, anon, log_level, reset, \
                   ft_type, img_dims, db_path, cq_data, cr_data):

        self.node_name      = node_name
        self.namespace      = namespace
        self.nodespace      = self.namespace + "/" + self.node_name

        rospy.init_node(self.node_name, anonymous=anon, log_level=log_level)
        self.ROS_HOME       = ROS_Home(self.node_name, self.namespace, rate)
        self.print('Starting %s node.' % (node_name))

        self.init_params(log_level, rate, ft_type, img_dims, db_path, cq_data, cr_data, reset)
        self.init_vars()
        self.init_rospy()

    def init_rospy(self):
        self.rate_obj       = rospy.Rate(self.rate_num.get())
        self.param_sub      = rospy.Subscriber(self.namespace + "/params_update", String, self.param_cb, queue_size=100)

    def init_params(self, log_level, rate, ft_type, img_dims, database_path, cal_qry_dataset_name, cal_ref_dataset_name, reset):
        # Set parameters:
        self.log_level      = self.ROS_HOME.params.add(self.nodespace + "/log_level", log_level, check_positive_int,     force=reset)
        self.rate_num       = self.ROS_HOME.params.add(self.nodespace + "/rate",      rate,      check_positive_float,   force=reset)
        self.feat_type      = self.ROS_HOME.params.add(self.namespace + "/feature_type",      enum_name(ft_type),   lambda x: check_enum(x, FeatureType, skip=[FeatureType.NONE]), force=reset)
        self.img_dims       = self.ROS_HOME.params.add(self.namespace + "/img_dims",          img_dims,             check_positive_two_int_tuple,                                  force=reset)
        self.database_path  = self.ROS_HOME.params.add(self.namespace + "/database_path",     database_path,        check_string,                                                  force=reset)
        self.cal_qry_data   = self.ROS_HOME.params.add(self.namespace + "/cal/qry/data_name", cal_qry_dataset_name, check_string,                                                  force=reset)
        self.cal_ref_data   = self.ROS_HOME.params.add(self.namespace + "/cal/ref/data_name", cal_ref_dataset_name, check_string,                                                  force=reset)
        self.img_view       = self.ROS_HOME.params.add(self.namespace + "/img_view",          enum_name(ViewMode.forward),      lambda x: check_enum(x, ViewMode),                 force=reset)

    def init_vars(self):
        # Set up SVM
        self.svm_params     = dict(ref=self.cal_ref_data.get(), qry=self.cal_qry_data.get(), img_dims=self.img_dims.get(), \
                                           view=self.img_view.get(), ft_type=self.feat_type.get(), database_path=self.database_path.get())
        self.svm_dir        = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/cfg/svm_models"
        self.svm            = SVMModelProcessor(self.svm_dir, model=self.svm_params, ros=True)
    
    def update_SVM(self):
        # Trigger parameter updates:
        self.svm_params     = dict(ref=self.cal_ref_data.get(), qry=self.cal_qry_data.get(), img_dims=self.img_dims.get(), \
                                           view=self.img_view.get(), ft_type=self.feat_type.get(), database_path=self.database_path.get())
        if not self.svm.swap(self.svm_params, generate=True):
            self.print("Model swap failed. Previous model will be retained (ROS parameters won't be updated!)", LogType.ERROR)
        else:
            self.print("SVM model swapped.", LogType.INFO)

    def param_cb(self, msg):
        if self.ROS_HOME.params.exists(msg.data):
            self.print("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG)
            self.ROS_HOME.params.update(msg.data)

            if msg.data == self.log_level.name:
                set_rospy_log_lvl(self.log_level.get())
            elif msg.data == self.rate_num.name:
                self.rate_obj = rospy.Rate(self.rate_num.get())
            elif msg.data in [self.cal_ref_data.name, self.cal_qry_data.name, self.img_dims.name, \
                            self.img_view.name, self.feat_type.name, self.database_path.name]:
                self.print("Change to SVM parameters detected.", LogType.WARN)
                self.update_SVM()
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)

    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.rate_obj.sleep()

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

def do_args():
    parser = ap.ArgumentParser(prog="svm_trainer.py", 
                            description="ROS SVM Trainer Node",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    # Positional Arguments:
    parser.add_argument('cal-qry-dataset-name', help='Specify name of calibration query dataset (for fast loading; matches are made on names starting with provided string).')
    parser.add_argument('cal-ref-dataset-name', help='Specify name of calibration reference dataset (for fast loading; matches are made on names starting with provided string).')
    parser.add_argument('database-path',        help="Specify path to where compressed databases exist (for fast loading).")

    ft_options, ft_options_text = enum_value_options(FeatureType, skip=FeatureType.NONE)

    # Optional Arguments:
    parser.add_argument('--node-name', '-N', type=check_string,                 default="svm_trainer",  help="Set node name (default: %(default)s).")
    parser.add_argument('--rate',      '-r', type=check_positive_float,         default=2.0,            help='Set node rate (default: %(default)s).')
    parser.add_argument('--anon',      '-a', type=check_bool,                   default=False,          help="Set whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace', '-n', type=check_string,                 default="/vpr_nodes",   help="Set ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16],    default=2,              help="Set ROS log level (default: %(default)s).")
    parser.add_argument('--reset',     '-R', type=check_bool,                   default=False,          help='Force reset of parameters to specified ones (default: %(default)s)')
    parser.add_argument('--img-dims',  '-i', type=check_positive_two_int_tuple, default=(64,64),        help='Set image dimensions (default: %(default)s).')
    parser.add_argument('--ft-type',   '-F', type=int, choices=ft_options,      default=ft_options[0],  help='Set feature type for extraction, types: %s (default: %s).' % (ft_options_text, '%(default)s'))
    
    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])
    return args

if __name__ == '__main__':
    
    args        = do_args()

    node_name   = args['node_name']
    rate        = args['rate']
    namespace   = args['namespace']
    anon        = args['anon']
    log_level   = args['log_level']
    reset       = args['reset']

    ft_type     = enum_get(args['ft_type'], FeatureType)
    img_dims    = args['img_dims']
    db_path     = args['database-path']
    cq_data     = args['cal-qry-dataset-name']
    cr_data     = args['cal-ref-dataset-name']

    try:
        nmrc = mrc(node_name, rate, namespace, anon, log_level, reset, \
                   ft_type, img_dims, db_path, cq_data, cr_data)
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