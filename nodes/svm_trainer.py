#!/usr/bin/env python3

import rospy
import rospkg
import argparse as ap
import numpy as np
import sys
import os
from std_msgs.msg import String
from pyaarapsi.core.argparse_tools import check_positive_float, check_positive_int, check_bool, check_string, check_positive_two_int_tuple, check_enum
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, set_rospy_log_lvl, init_node
from pyaarapsi.core.helper_tools import formatException

from pyaarapsi.vpr_simple.imageprocessor_helpers import FeatureType
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
    def __init__(self, node_name, rate_num, namespace, anon, log_level, reset, order_id=0):

        init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30)

        self.init_params(log_level, rate_num, reset)
        self.init_vars()
        self.init_rospy()

        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_rospy(self):
        self.rate_obj       = rospy.Rate(self.RATE_NUM.get())
        self.param_sub      = rospy.Subscriber(self.namespace + "/params_update", String, self.param_callback, queue_size=100)

    def init_params(self, rate_num, log_level, reset):
        self.FEAT_TYPE              = self.ROS_HOME.params.add(self.namespace + "/feature_type",        None,             lambda x: check_enum(x, FeatureType),       force=False)
        self.IMG_DIMS               = self.ROS_HOME.params.add(self.namespace + "/img_dims",            None,             check_positive_two_int_tuple,               force=False)
        self.NPZ_DBP                = self.ROS_HOME.params.add(self.namespace + "/npz_dbp",             None,             check_string,                               force=False)
        self.BAG_DBP                = self.ROS_HOME.params.add(self.namespace + "/bag_dbp",             None,             check_string,                               force=False)
        self.SVM_DBP                = self.ROS_HOME.params.add(self.namespace + "/svm_dbp",             None,             check_string,                               force=False)
        self.IMG_TOPIC              = self.ROS_HOME.params.add(self.namespace + "/img_topic",           None,             check_string,                               force=False)
        self.ODOM_TOPIC             = self.ROS_HOME.params.add(self.namespace + "/odom_topic",          None,             check_string,                               force=False)
        
        self.CAL_QRY_BAG_NAME       = self.ROS_HOME.params.add(self.namespace + "/cal/qry/bag_name",    None,             check_string,                               force=False)
        self.CAL_QRY_FILTERS        = self.ROS_HOME.params.add(self.namespace + "/cal/qry/filters",     None,             check_string,                               force=False)
        self.CAL_QRY_SAMPLE_RATE    = self.ROS_HOME.params.add(self.namespace + "/cal/qry/sample_rate", None,             check_positive_float,                       force=False)

        self.CAL_REF_BAG_NAME       = self.ROS_HOME.params.add(self.namespace + "/cal/ref/bag_name",    None,             check_string,                               force=False)
        self.CAL_REF_FILTERS        = self.ROS_HOME.params.add(self.namespace + "/cal/ref/filters",     None,             check_string,                               force=False)
        self.CAL_REF_SAMPLE_RATE    = self.ROS_HOME.params.add(self.namespace + "/cal/ref/sample_rate", None,             check_positive_float,                       force=False)
        
        self.RATE_NUM               = self.ROS_HOME.params.add(self.nodespace + "/rate",                rate_num,         check_positive_float,                       force=reset)
        self.LOG_LEVEL              = self.ROS_HOME.params.add(self.nodespace + "/log_level",           log_level,        check_positive_int,                         force=reset)
        
        self.SVM_DATA_PARAMS        = [self.FEAT_TYPE, self.IMG_DIMS, self.NPZ_DBP, self.BAG_DBP, self.SVM_DBP, self.IMG_TOPIC, self.ODOM_TOPIC, \
                                       self.CAL_QRY_BAG_NAME, self.CAL_QRY_FILTERS, self.CAL_QRY_SAMPLE_RATE, \
                                       self.CAL_REF_BAG_NAME, self.CAL_REF_FILTERS, self.CAL_REF_SAMPLE_RATE]
        self.SVM_DATA_NAMES         = [i.name for i in self.SVM_DATA_PARAMS]
        
    def init_vars(self):
        # Set up SVM
        self.svm_model_params       = self.make_svm_model_params()
        self.svm                    = SVMModelProcessor(self.svm_model_params, try_gen=True, ros=True, \
                                                        init_hybridnet=True, init_netvlad=True, cuda=True, \
                                                        autosave=True, printer=self.print)

    def make_svm_model_params(self):
        return dict(ref=self.make_ref_dataset_dict(), qry=self.make_qry_dataset_dict(), bag_dbp=self.BAG_DBP.get(), npz_dbp=self.NPZ_DBP.get(), svm_dbp=self.SVM_DBP.get())

    def make_ref_dataset_dict(self):
        return dict(bag_name=self.CAL_REF_BAG_NAME.get(), odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], \
                    sample_rate=self.CAL_REF_SAMPLE_RATE.get(), ft_types=[self.FEAT_TYPE.get()], img_dims=self.IMG_DIMS.get(), filters={})

    def make_qry_dataset_dict(self):
        return dict(bag_name=self.CAL_QRY_BAG_NAME.get(), odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], \
                    sample_rate=self.CAL_QRY_SAMPLE_RATE.get(), ft_types=[self.FEAT_TYPE.get()], img_dims=self.IMG_DIMS.get(), filters={})

    def update_SVM(self, param_to_change=None):
        self.svm_model_params       = self.make_svm_model_params()
        if not self.svm.swap(self.svm_model_params, generate=True):
            self.print("Model generation failed. Reverting change to ROS parameter %s." % param_to_change.name, LogType.WARN, throttle=5)
            param_to_change.revert()
        else:
            self.print("SVM model swapped.")
            self.print(str(self.svm_model_params), LogType.DEBUG)

    def param_callback(self, msg):
        if self.ROS_HOME.params.exists(msg.data):
            self.print("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG)
            self.ROS_HOME.params.update(msg.data)

            if msg.data == self.LOG_LEVEL.name:
                set_rospy_log_lvl(self.LOG_LEVEL.get())
            elif msg.data == self.RATE_NUM.name:
                self.rate_obj = rospy.Rate(self.RATE_NUM.get())

            svm_data_comp   = [i == msg.data for i in self.SVM_DATA_NAMES]
            try:
                param = np.array(self.SVM_DATA_PARAMS)[svm_data_comp][0]
                self.print("Change to SVM parameters detected.", LogType.WARN)
                self.update_SVM(param)
            except:
                self.print(formatException(), LogType.DEBUG)
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

    # Optional Arguments:
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="svm_trainer",    help="Set node name (default: %(default)s).")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=2.0,              help='Set node rate (default: %(default)s).')
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=False,            help="Set whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Set ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Set ROS log level (default: %(default)s).")
    parser.add_argument('--reset',            '-R',  type=check_bool,                   default=False,            help='Force reset of parameters to specified ones (default: %(default)s)')
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')

    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])
    return args

if __name__ == '__main__':
    args        = do_args()
    try:
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