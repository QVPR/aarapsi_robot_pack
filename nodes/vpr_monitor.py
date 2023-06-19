#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

import numpy as np
import argparse as ap
import sys
import cv2
import copy

from rospy_message_converter import message_converter

from aarapsi_robot_pack.msg import RequestSVM, ResponseSVM, ImageLabelDetails, MonitorDetails  # Our custom structures

from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.svm_model_tool    import SVMModelProcessor
from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType, SVM_Tolerance_Mode

from pyaarapsi.vpred                import *

from pyaarapsi.core.argparse_tools  import check_positive_float, check_positive_two_int_list, check_bool, check_enum, check_string, check_positive_int, check_string_list
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.ros_tools       import roslogger, set_rospy_log_lvl, init_node, LogType, NodeState, SubscribeListener
from pyaarapsi.core.enum_tools      import enum_name

class mrc: # main ROS class
    def __init__(self, rate_num, namespace, node_name, anon, print_prediction, log_level, reset, order_id=0):

        if not init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30):
            sys.exit()

        self.init_params(rate_num, log_level, print_prediction, reset)
        self.init_vars()
        self.init_rospy()

        self.main_ready             = True
        rospy.set_param(self.namespace + '/launch_step', order_id + 1)
        
    def init_params(self, rate_num, log_level, print_prediction, reset):
        self.FEAT_TYPE              = self.ROS_HOME.params.add(self.namespace + "/feature_type",            None,             lambda x: check_enum(x, FeatureType),         force=False)
        self.IMG_DIMS               = self.ROS_HOME.params.add(self.namespace + "/img_dims",                None,             check_positive_two_int_list,                  force=False)
        self.NPZ_DBP                = self.ROS_HOME.params.add(self.namespace + "/npz_dbp",                 None,             check_string,                                 force=False)
        self.BAG_DBP                = self.ROS_HOME.params.add(self.namespace + "/bag_dbp",                 None,             check_string,                                 force=False)
        self.SVM_DBP                = self.ROS_HOME.params.add(self.namespace + "/svm_dbp",                 None,             check_string,                                 force=False)
        self.IMG_TOPIC              = self.ROS_HOME.params.add(self.namespace + "/img_topic",               None,             check_string,                                 force=False)
        self.ODOM_TOPIC             = self.ROS_HOME.params.add(self.namespace + "/odom_topic",              None,             check_string,                                 force=False)
        
        self.REF_BAG_NAME           = self.ROS_HOME.params.add(self.namespace + "/ref/bag_name",            None,             check_string,                                 force=False)
        self.REF_FILTERS            = self.ROS_HOME.params.add(self.namespace + "/ref/filters",             None,             check_string,                                 force=False)
        self.REF_SAMPLE_RATE        = self.ROS_HOME.params.add(self.namespace + "/ref/sample_rate",         None,             check_positive_float,                         force=False) # Hz

        self.SVM_QRY_BAG_NAME       = self.ROS_HOME.params.add(self.namespace + "/svm/qry/bag_name",        None,             check_string,                                 force=False)
        self.SVM_QRY_FILTERS        = self.ROS_HOME.params.add(self.namespace + "/svm/qry/filters",         None,             check_string,                                 force=False)
        self.SVM_QRY_SAMPLE_RATE    = self.ROS_HOME.params.add(self.namespace + "/svm/qry/sample_rate",     None,             check_positive_float,                         force=False)

        self.SVM_REF_BAG_NAME       = self.ROS_HOME.params.add(self.namespace + "/svm/ref/bag_name",        None,             check_string,                                 force=False)
        self.SVM_REF_FILTERS        = self.ROS_HOME.params.add(self.namespace + "/svm/ref/filters",         None,             check_string,                                 force=False)
        self.SVM_REF_SAMPLE_RATE    = self.ROS_HOME.params.add(self.namespace + "/svm/ref/sample_rate",     None,             check_positive_float,                         force=False)
        
        self.SVM_FACTORS            = self.ROS_HOME.params.add(self.namespace + "/svm/factors",             None,             check_string_list,                            force=False)
        self.SVM_TOL_MODE           = self.ROS_HOME.params.add(self.namespace + "/svm/tolerance/mode",      None,             lambda x: check_enum(x, SVM_Tolerance_Mode),  force=False)
        self.SVM_TOL_THRES          = self.ROS_HOME.params.add(self.namespace + "/svm/tolerance/threshold", None,             check_positive_float,                         force=False)
        
        self.RATE_NUM               = self.ROS_HOME.params.add(self.nodespace + "/rate",                    rate_num,         check_positive_float,                         force=reset)
        self.LOG_LEVEL              = self.ROS_HOME.params.add(self.nodespace + "/log_level",               log_level,        check_positive_int,                           force=reset)
        self.PRINT_PREDICTION       = self.ROS_HOME.params.add(self.nodespace + "/print_prediction",        print_prediction, check_bool,                                   force=reset)
       
        self.SVM_DATA_PARAMS        = [self.FEAT_TYPE, self.IMG_DIMS, self.NPZ_DBP, self.BAG_DBP, self.SVM_DBP, self.IMG_TOPIC, self.ODOM_TOPIC, \
                                       self.SVM_QRY_BAG_NAME, self.SVM_QRY_FILTERS, self.SVM_QRY_SAMPLE_RATE, \
                                       self.SVM_REF_BAG_NAME, self.SVM_REF_FILTERS, self.SVM_REF_SAMPLE_RATE, \
                                       self.SVM_FACTORS, self.SVM_TOL_MODE, self.SVM_TOL_THRES]
        self.SVM_DATA_NAMES         = [i.name for i in self.SVM_DATA_PARAMS]

        self.REF_DATA_PARAMS        = [self.NPZ_DBP, self.BAG_DBP, self.REF_BAG_NAME, self.REF_FILTERS, self.REF_SAMPLE_RATE, self.IMG_TOPIC, self.ODOM_TOPIC, self.FEAT_TYPE, self.IMG_DIMS]
        self.REF_DATA_NAMES         = [i.name for i in self.REF_DATA_PARAMS]

    def init_vars(self):
        self.contour_msg            = None

        self.state_hist             = np.zeros((10,3)) # x, y, w
        self.state_size             = 0

        # Set up SVM
        self.svm                    = SVMModelProcessor(ros=True)
        if not self.svm.load_model(self.make_svm_model_params()):
            raise Exception("Failed to find file with parameters matching: \n%s" % str(self.make_svm_model_params()))
        self.svm_requests           = []

        # flags to denest main loop:
        self.new_label              = False # new label received
        self.main_ready             = False # ensure pubs and subs don't go off early
        self.svm_swap_pending       = False
        self.parameters_ready       = True
        ref_dataset_dict            = self.make_dataset_dict()
        try:
            self.ip                 = VPRDatasetProcessor(ref_dataset_dict, try_gen=False, ros=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

    def init_rospy(self):

        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())
        self.last_time              = rospy.Time.now()
        self.sublis                 = SubscribeListener()

        self.param_checker_sub      = rospy.Subscriber(     self.namespace + "/params_update",          String,              self.param_callback,           queue_size=100)
        self.vpr_label_sub          = rospy.Subscriber(     self.namespace + "/label",                  ImageLabelDetails,   self.label_callback,           queue_size=1)
        self.svm_request_sub        = rospy.Subscriber(     self.namespace + "/requests/svm/ready",     ResponseSVM,         self.svm_request_callback,     queue_size=1)
        self.svm_request_pub        = self.ROS_HOME.add_pub(self.namespace + "/requests/svm/request",   RequestSVM,                                         queue_size=1)
        self.svm_state_pub          = self.ROS_HOME.add_pub(self.namespace + "/state",                  MonitorDetails,                                     queue_size=1)

    def make_svm_model_params(self):
        qry_dict = dict(bag_name=self.SVM_QRY_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                        odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.SVM_REF_SAMPLE_RATE.get(), \
                        ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')
        ref_dict = dict(bag_name=self.SVM_REF_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                        odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.SVM_REF_SAMPLE_RATE.get(), \
                        ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')
        svm_dict = dict(factors=self.SVM_FACTORS.get(), tol_thres=self.SVM_TOL_THRES.get(), tol_mode=enum_name(self.SVM_TOL_MODE.get()))
        return dict(ref=ref_dict, qry=qry_dict, svm=svm_dict, npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), svm_dbp=self.SVM_DBP.get())

    def make_dataset_dict(self):
        return dict(bag_name=self.REF_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                    odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.REF_SAMPLE_RATE.get(), \
                    ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')

    def update_SVM(self):
        svm_model_params       = self.make_svm_model_params()
        if not self.svm.swap(svm_model_params, generate=False, allow_false=True):
            self.print("Model swap failed. Previous model will be retained (Changed ROS parameter will revert)", LogType.WARN, throttle=5)
            self.svm_swap_pending   = True
            self.svm_requests.append(svm_model_params)
            svm_msg = message_converter.convert_dictionary_to_ros_message('aarapsi_robot_pack/RequestSVM', svm_model_params)
            self.svm_request_pub.publish(svm_msg)
            return False
        else:
            self.print("SVM model swapped.")
            self.svm_swap_pending = False
            return True

    def update_VPR(self):
        dataset_dict = self.make_dataset_dict()
        if not self.ip.swap(dataset_dict, generate=False, allow_false=True):
            self.print("VPR reference data swap failed. Previous set will be retained (changed ROS parameter will revert)", LogType.WARN)
            return False
        else:
            self.print("VPR reference data swapped.", LogType.INFO)
            return True

    def debug_cb(self, msg):
        if msg.node_name == self.node_name:
            if msg.instruction == 0:
                self.print(self.make_svm_model_params(), LogType.DEBUG)
            else:
                self.print(msg.instruction, LogType.DEBUG)

    def svm_request_callback(self, msg):
        pass

    def label_callback(self, msg):
    # /vpr_nodes/label (aarapsi_robot_pack/ImageLabelDetails)
    # Store new label message and act as drop-in replacement for odom_callback + img_callback

        self.label            = msg

        self.state_hist       = np.roll(self.state_hist, 1, 0)
        self.state_hist[0,:]  = [msg.data.vpr_ego.x, msg.data.vpr_ego.y, msg.data.vpr_ego.w]
        self.state_size       = np.min([self.state_size + 1, self.state_hist.shape[0]])

        self.new_label        = True


    def param_callback(self, msg):
        self.parameters_ready = False
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
                if not self.update_SVM():
                    param.revert()
            except IndexError:
                pass
            except:
                self.print(formatException(), LogType.DEBUG)

            ref_data_comp   = [i == msg.data for i in self.REF_DATA_NAMES]
            try:
                param = np.array(self.REF_DATA_PARAMS)[ref_data_comp][0]
                self.print("Change to VPR reference data parameters detected.", LogType.WARN)
                if not self.update_VPR():
                    param.revert()
            except IndexError:
                pass
            except:
                param.revert()
                self.print(formatException(), LogType.ERROR)
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)
        self.parameters_ready = True

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.loop_contents()

    def loop_contents(self):

        if self.svm_swap_pending:
            self.update_SVM()

        if not (self.new_label and self.main_ready): # denest
            self.print("Waiting for a new label.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return
        nmrc.rate_obj.sleep()
        self.new_label = False

        # Predict model information, but have a try-except to catch if model is transitioning state
        predict_success = False
        while not predict_success:
            if rospy.is_shutdown():
                self.exit()
            try:
                rXY = np.stack([self.ip.dataset['dataset']['px'], self.ip.dataset['dataset']['py']], 1)
                (pred, zvalues, [factor1, factor2], prob) = self.svm.predict(self.label.data.dvc, self.label.data.matchId, rXY, init_pos=self.state_hist[1, 0:2])
                predict_success = True
            except:
                self.print("Predict failed. Trying again ...", LogType.WARN, throttle=1)
                self.print(formatException(), LogType.DEBUG, throttle=1)
                rospy.sleep(0.005)

        if self.PRINT_PREDICTION.get():
            if self.label.data.state == 0:
                self.print('integrity prediction: %s', pred, LogType.INFO)
            else:
                gt_state_bool = bool(self.label.data.state - 1)
                if pred == gt_state_bool:
                    self.print('integrity prediction: %r [gt: %r]' % (pred, gt_state_bool), LogType.INFO)
                elif pred == False and gt_state_bool == True:
                    self.print('integrity prediction: %r [gt: %r]' % (pred, gt_state_bool), LogType.WARN)
                else:
                    self.print('integrity prediction: %r [gt: %r]' % (pred, gt_state_bool), LogType.ERROR)

        # Populate and publish SVM State details
        ros_msg                 = MonitorDetails()
        ros_msg.queryImage      = self.label.queryImage
        ros_msg.header.stamp    = rospy.Time.now()
        ros_msg.header.frame_id	= 'map'
        ros_msg.data            = self.label.data
        ros_msg.mState	        = zvalues # Continuous monitor state estimate 
        ros_msg.prob	        = prob # Monitor probability estimate
        ros_msg.mStateBin       = pred# Binary monitor state estimate
        ros_msg.factors         = [factor1, factor2]

        self.svm_state_pub.publish(ros_msg)
        del ros_msg

    def exit(self):
        self.print("Quit received")
        sys.exit()

def do_args():
    parser = ap.ArgumentParser(prog="vpr_monitor.py", 
                                description="ROS implementation of Helen Carson's Integrity Monitor, for integration with QVPR's VPR Primer",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=10.0,             help='Set node rate (default: %(default)s).')
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="vpr_all_in_one", help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=True,             help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Specify namespace for topics (default: %(default)s).")
    parser.add_argument('--print-prediction', '-p',  type=check_bool,                   default=True,             help="Specify whether the monitor's prediction should be printed (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset',            '-R',  type=check_bool,                   default=False,            help='Force reset of parameters to specified ones (default: %(default)s)')
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')
    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = mrc(rate_num=args['rate'], namespace=args['namespace'], node_name=args['node_name'], anon=args['anon'],  
                    print_prediction=args['print_prediction'], log_level=args['log_level'], reset=args['reset'], order_id=args['order_id']\
                )
        nmrc.print("Initialisation complete. Listening for queries...", LogType.INFO)
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