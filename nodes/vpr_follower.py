#!/usr/bin/env python3

import rospy

from fastdist import fastdist
import numpy as np
import argparse as ap
import sys

from rospy_message_converter import message_converter
from aarapsi_robot_pack.srv import DoExtraction, DoExtractionRequest
from aarapsi_robot_pack.msg import RequestDataset, ResponseDataset, xyw, ImageOdom, ImageLabelDetails

from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

from pyaarapsi.core.enum_tools              import enum_name
from pyaarapsi.core.argparse_tools          import check_bounded_float, check_positive_float, check_positive_int, check_enum, check_string, check_bool
from pyaarapsi.core.ros_tools               import yaw_from_q, q_from_yaw, roslogger, LogType, NodeState, compressed2np
from pyaarapsi.core.helper_tools            import formatException, uint8_list_to_np_ndarray, m2m_dist, Timer
from pyaarapsi.vpr_simple.vpr_helpers       import VPR_Tolerance_Mode, FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.svm_model_tool    import SVMModelProcessor
from pyaarapsi.vpr_classes.base             import Base_ROS_Class, base_optional_args

'''
Node Name

Node description.

'''

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_params(self, rate_num, log_level, reset):
        super().init_params(rate_num, log_level, reset)
        self.TOL_MODE        = self.params.add(self.namespace + "/tolerance/mode",      None,                   lambda x: check_enum(x, VPR_Tolerance_Mode),    force=False)
        self.TOL_THRES       = self.params.add(self.namespace + "/tolerance/threshold", None,                   check_positive_float,                           force=False)
        self.VPR_ODOM_TOPIC  = self.params.add(self.namespace + "/vpr_odom_topic",      None,                   check_string,                                   force=False)
        self.TIME_HIST_LEN   = self.params.add(self.nodespace + "/time_history_length", max(1,int(5*rate_num)), check_positive_int,                             force=reset)

    def init_vars(self):
        super().init_vars()

        self.gt_ego                 = [0.0, 0.0, 0.0] # ground truth robot position
        self.vpr_ego                = [0.0, 0.0, 0.0] # our estimate of robot position
        self.state_hist             = np.zeros((10,3)) # x, y, w

        # flags to denest main loop:
        self.new_query              = False # new query odom+image
        self.main_ready             = False # make sure everything commences together, safely
        self.ego_known              = False # whether or not an initial position has been found
        self.dataset_swap_pending   = False # whether we are waiting on a vpr dataset to be built

        self.dataset_requests       = []
        self.time_history           = []

        # Process reference data
        try:
            self.ip                 = VPRDatasetProcessor(self.make_dataset_dict(path=False), 
                                                            init_hybridnet=False, init_netvlad=True, cuda=True, \
                                                            try_gen=True, autosave=True, use_tqdm=True, ros=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

        # Set up SVM
        self.svm                    = SVMModelProcessor(ros=True)
        if not self.svm.load_model(self.make_svm_model_params()):
            raise Exception("Failed to find file with parameters matching: \n%s" % str(self.make_svm_model_params()))
        self.svm_requests           = []

    def init_rospy(self):
        super().init_rospy()
        
        self.last_time              = rospy.Time.now()
        self.data_sub               = rospy.Subscriber(     self.namespace + "/img_odom",                   ImageOdom,          self.data_callback,             queue_size=1)
        
    def data_callback(self, msg: ImageOdom):

        self.gt_ego                 = [round(msg.odom.pose.pose.position.x, 3), round(msg.odom.pose.pose.position.y, 3), round(yaw_from_q(msg.odom.pose.pose.orientation), 3)]
        self.store_query            = compressed2np(msg.image)
        self.new_query              = True

    def getMatchInd(self, ft_qry: list):
    # top matching reference index for query

        _features   = self.ip.dataset['dataset'][enum_name(self.FEAT_TYPE.get())]
        dvc         = m2m_dist(_features, np.matrix(ft_qry), True)
        mInd        = np.argmin(dvc)
        return mInd, dvc
    
    def getTrueInd(self):
    # Compare measured odometry to reference odometry and find best match
        squares     = np.square(np.array(self.ip.dataset['dataset']['px']) - self.gt_ego[0]) + \
                                np.square(np.array(self.ip.dataset['dataset']['py']) - self.gt_ego[1])  # no point taking the sqrt; proportional
        tInd        = np.argmin(squares)

        return tInd

    def main(self):
        # Main loop process
        self.set_state(NodeState.MAIN)

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

    def perform_vpr(self):
        ft_qry          = self.ip.getFeat(self.store_query, self.FEAT_TYPE.get(), use_tqdm=False, dims=self.IMG_DIMS.get())
        mInd, dvc       = self.getMatchInd(ft_qry) # Find match
        tInd            = self.getTrueInd() # find correct match based on shortest difference to measured odometry
        tolMode         = self.TOL_MODE.get()

        # Determine if we are within tolerance:
        if tolMode == VPR_Tolerance_Mode.METRE_CROW_TRUE:
            tolError = np.sqrt(np.square(self.ip.dataset['dataset']['px'][tInd] - self.gt_ego[0]) + \
                    np.square(self.ip.dataset['dataset']['py'][tInd] - self.gt_ego[1])) 
        elif tolMode == VPR_Tolerance_Mode.METRE_CROW_MATCH:
            tolError = np.sqrt(np.square(self.ip.dataset['dataset']['px'][mInd] - self.gt_ego[0]) + \
                    np.square(self.ip.dataset['dataset']['py'][mInd] - self.gt_ego[1])) 
        elif tolMode == VPR_Tolerance_Mode.METRE_LINE:
            tolError = np.sqrt(np.square(self.ip.dataset['dataset']['px'][tInd] - self.ip.dataset['dataset']['px'][mInd]) + \
                    np.square(self.ip.dataset['dataset']['py'][tInd] - self.ip.dataset['dataset']['py'][mInd])) 
        elif tolMode == VPR_Tolerance_Mode.FRAME:
            tolError = np.abs(mInd - tInd)
        else:
            raise Exception("Error: Unknown tolerance mode.")
        tolState = int(tolError < self.TOL_THRES.get())

        self.vpr_ego = [self.ip.dataset['dataset']['px'][mInd], self.ip.dataset['dataset']['py'][mInd], self.ip.dataset['dataset']['pw'][mInd]]

        return {'ft_qry': ft_qry, 'mInd': mInd, 'tInd': tInd, 'dvc': dvc, 'tolError': tolError, 'tolState': tolState}
    
    def perform_svm(self, vpr_results):
        # Predict model information, but have a try-except to catch if model is transitioning state
        self.state_hist         = np.roll(self.state_hist, 1, 0)
        self.state_hist[0,:]    = [self.vpr_ego[0], self.vpr_ego[1], self.vpr_ego[2]]
        predict_success         = False
        while not predict_success:
            if rospy.is_shutdown():
                self.exit()
            try:
                rXY = np.stack([self.ip.dataset['dataset']['px'], self.ip.dataset['dataset']['py']], 1)
                (pred, zvalues, factors, prob) = self.svm.predict(vpr_results['dvc'], vpr_results['mInd'], rXY, init_pos=self.state_hist[1, 0:2])
                predict_success = True
            except:
                self.print("Predict failed. Trying again ...", LogType.WARN, throttle=1)
                self.print(formatException(), LogType.DEBUG, throttle=1)
                rospy.sleep(0.005)
        return {'pred': pred, 'zvalues': zvalues, 'factors': factors, 'prob': prob}

    def loop_contents(self):

        if not (self.main_ready and self.parameters_ready): # denest
            self.print("Waiting.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return

        self.rate_obj.sleep()

        if self.new_query:
            vpr_result = self.perform_vpr()
            svm_result = self.perform_svm(vpr_result)
            self.new_query  = False

def do_args():
    parser = ap.ArgumentParser(prog="vpr_follower.py", 
                            description="VPR Follower",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='vpr_follower', rate=10.0)
    
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