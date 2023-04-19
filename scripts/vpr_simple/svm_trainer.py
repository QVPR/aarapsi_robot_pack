#!/usr/bin/env python3

import rospy
import rospkg
import argparse as ap
import sys
import os
from std_msgs.msg import String
from pyaarapsi.core.argparse_tools import check_positive_float, check_positive_int, check_bool, check_string, check_positive_two_int_tuple, check_enum
from pyaarapsi.core.ros_tools import Heartbeat, NodeState, roslogger, LogType, ROS_Param_Server, set_rospy_log_lvl
from pyaarapsi.core.helper_tools import formatException
from pyaarapsi.core.enum_tools import enum_name, enum_value_options, enum_get
from pyaarapsi.vpr_simple import SVMModelProcessor, FeatureType

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
        self.anon           = anon

        self.params         = ROS_Param_Server()
        self.log_level      = self.params.add(self.nodespace + "/log_level", log_level, check_positive_int,     force=reset)
        self.rate_num       = self.params.add(self.nodespace + "/rate",      rate,      check_positive_float,   force=reset)
    
        rospy.init_node(self.node_name, anonymous=self.anon, log_level=self.log_level.get())
        roslogger('Starting [%s] %s node.' % (self.namespace, self.node_name), LogType.INFO, ros=True)
        self.rate_obj       = rospy.Rate(self.rate_num.get())
        self.heartbeat      = Heartbeat(self.node_name, self.namespace, NodeState.INIT, self.rate_num)
        self.param_sub      = rospy.Subscriber(self.namespace + "/params_update", String, self.param_cb, queue_size=100)

        self.prep_svm(ft_type, img_dims, db_path, cq_data, cr_data)

    def prep_svm(self, ft_type, img_dims, database_path, cal_qry_dataset_name, cal_ref_dataset_name):
        # Set parameters:
        self.feat_type      = self.params.add(self.namespace + "/feature_type",                  enum_name(ft_type),   lambda x: check_enum(x, FeatureType, skip=[FeatureType.NONE]), force=reset)
        self.img_dims       = self.params.add(self.namespace + "/img_dims",                      img_dims,             check_positive_two_int_tuple,                                  force=reset)
        self.database_path  = self.params.add(self.namespace + "/database_path",                 database_path,        check_string,                                                  force=reset)
        self.cal_qry_data   = self.params.add(self.namespace + "/vpr_monitor/cal/qry/data_name", cal_qry_dataset_name, check_string,                                                  force=reset)
        self.cal_ref_data   = self.params.add(self.namespace + "/vpr_monitor/cal/ref/data_name", cal_ref_dataset_name, check_string,                                                  force=reset)
        self.cal_folder     = self.params.add(self.namespace + "/vpr_monitor/cal/folder",        "forward",            check_string,                                                  force=reset)
        # Set up SVM
        self.svm_params     = dict(ref=self.cal_ref_data.get(), qry=self.cal_qry_data.get(), img_dims=self.img_dims.get(), \
                                           folder=self.cal_folder.get(), ft_type=self.feat_type.get(), database_path=self.database_path.get())
        self.svm_dir        = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/cfg/svm_models"
        self.svm            = SVMModelProcessor(self.svm_dir, model=self.svm_params, ros=True)
    
    def update_SVM(self):
        # Trigger parameter updates:
        self.svm_params     = dict(ref=self.cal_ref_data.get(), qry=self.cal_qry_data.get(), img_dims=self.img_dims.get(), \
                                           folder=self.cal_folder.get(), ft_type=self.feat_type.get(), database_path=self.database_path.get())
        if not self.svm.swap(self.svm_params, generate=True):
            roslogger("Model swap failed. Previous model will be retained (ROS parameters won't be updated!)", LogType.ERROR, ros=True)
        else:
            roslogger("SVM model swapped.", LogType.INFO, ros=True)

    def param_cb(self, msg):
        if self.params.exists(msg.data):
            roslogger("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG, ros=True)
            self.params.update(msg.data)

            if msg.data == self.log_level.name:
                set_rospy_log_lvl(self.log_level.get())
            elif msg.data == self.rate_num.name:
                self.rate_obj = rospy.Rate(self.rate_num.get())
            elif msg.data in [self.cal_ref_data.name, self.cal_qry_data.name, self.img_dims.name, \
                            self.cal_folder.name, self.feat_type.name, self.database_path.name]:
                roslogger("Change to SVM parameters detected.", LogType.WARN, ros=True)
                self.update_SVM()
        else:
            roslogger("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG, ros=True)

    def main(self):
        self.heartbeat.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.rate_obj.sleep()

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