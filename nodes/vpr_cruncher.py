#!/usr/bin/env python3

import rospy

from fastdist import fastdist
import numpy as np
import argparse as ap
import sys

from rospy_message_converter import message_converter
from aarapsi_robot_pack.srv import DoExtraction, DoExtractionRequest
from aarapsi_robot_pack.msg import RequestDataset, ResponseDataset, xyw, Label

from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

from pyaarapsi.core.enum_tools              import enum_name
from pyaarapsi.core.argparse_tools          import check_bounded_float, check_positive_float, check_positive_int, check_enum, check_string, check_bool
from pyaarapsi.core.ros_tools               import yaw_from_q, q_from_yaw, roslogger, LogType, NodeState, compressed2np
from pyaarapsi.core.helper_tools            import formatException, uint8_list_to_np_ndarray
from pyaarapsi.vpr_simple.vpr_helpers       import VPR_Tolerance_Mode, FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_classes.base             import Base_ROS_Class, base_optional_args

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars(kwargs['use_gpu'])
        self.init_rospy()
        
        self.node_ready(kwargs['order_id'])

    def init_params(self, rate_num: float, log_level: float, reset: bool):
        super().init_params(rate_num, log_level, reset)
        self.TOL_MODE        = self.params.add(self.namespace + "/tolerance/mode",      None,                   lambda x: check_enum(x, VPR_Tolerance_Mode),    force=False)
        self.TOL_THRES       = self.params.add(self.namespace + "/tolerance/threshold", None,                   check_positive_float,                           force=False)
        self.VPR_ODOM_TOPIC  = self.params.add(self.namespace + "/vpr_odom_topic",      None,                   check_string,                                   force=False)
        self.EXTRACT_SRV     = self.params.add(self.nodespace + "/service_extraction",  True,                   check_bool,                                     force=reset)
        self.TIME_HIST_LEN   = self.params.add(self.nodespace + "/time_history_length", max(1,int(5*rate_num)), check_positive_int,                             force=reset)
        self.DVC_WEIGHT      = self.params.add(self.nodespace + "/dvc_weight",          1,                      lambda x: check_bounded_float(x, 0, 1, 'both'), force=reset)
        
    def init_vars(self, use_gpu):
        super().init_vars()

        self.vpr_ego                = [0.0, 0.0, 0.0] # our estimate of robot position

        # flags to denest main loop:
        self.new_label              = False # new label
        self.main_ready             = False # make sure everything commences together, safely
        self.ego_known              = False # whether or not an initial position has been found
        self.dataset_swap_pending   = False # whether we are waiting on a vpr dataset to be built

        self.dataset_requests       = []
        self.time_history           = []

        # Process reference data
        try:
            dataset_params = self.make_dataset_dict()
            if self.EXTRACT_SRV.get():
                self.ip             = VPRDatasetProcessor(dataset_params=dataset_params, try_gen=False, ros=True, use_tqdm=False)
            else:
                self.ip             = VPRDatasetProcessor(None, try_gen=True, ros=True, use_tqdm=True, \
                                                          cuda=use_gpu, autosave=True)
                if use_gpu:
                    self.ip.init_nns()
                    
                self.ip.load_dataset(dataset_params=dataset_params, try_gen=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

    def init_rospy(self):
        super().init_rospy()
        
        self.last_time              = rospy.Time.now()

        self.odom_estimate_pub      = self.add_pub(         self.namespace + self.VPR_ODOM_TOPIC.get(),     Odometry,           queue_size=1)
        self.vpr_label_pub          = self.add_pub(         self.namespace + "/label",                      Label,              queue_size=1)
        self.dataset_request_pub    = self.add_pub(         self.namespace + "/requests/dataset/request",   RequestDataset,     queue_size=1)
        self.dataset_request_sub    = rospy.Subscriber(     self.namespace + '/requests/dataset/ready',     ResponseDataset,    self.dataset_request_callback,  queue_size=1)
        self.data_sub               = rospy.Subscriber(     self.namespace + "/img_odom",                   Label,              self.data_callback,             queue_size=1)
        self.srv_extraction         = rospy.ServiceProxy(   self.namespace + '/do_extraction',              DoExtraction)

    def dataset_request_callback(self, msg: ResponseDataset):
        pass # TODO

    def update_VPR(self):
        dataset_dict = self.make_dataset_dict()
        if not self.ip.swap(dataset_dict, generate=False, allow_false=True):
            self.print("VPR reference data swap failed. Previous set will be retained (changed ROS parameter will revert)", LogType.WARN)
            self.dataset_swap_pending = True
            self.dataset_requests.append(dataset_dict)
            dataset_msg = message_converter.convert_dictionary_to_ros_message('aarapsi_robot_pack/RequestDataset', dataset_dict)
            self.dataset_request_pub.publish(dataset_msg)
            return False
        else:
            self.print("VPR reference data swapped.", LogType.INFO)
            self.dataset_swap_pending = False
            return True

    def param_helper(self, msg: String):
        ref_data_comp   = [i == msg.data for i in self.REF_DATA_NAMES]
        try:
            param = np.array(self.REF_DATA_PARAMS)[ref_data_comp][0]
            self.print("Change to VPR reference data parameters detected.", LogType.WARN)
            if not self.update_VPR():
                param.revert()
        except IndexError:
            pass
        except:
            self.print(formatException(), LogType.ERROR)

    def data_callback(self, msg: Label):

        self.label                  = msg
        self.new_label              = True
        
    def getMatchInd(self, ft_qry: list):
    # top matching reference index for query

        _features   = self.ip.dataset['dataset'][enum_name(self.FEAT_TYPE.get())]
        _px         = self.ip.dataset['dataset']['px']
        _py         = self.ip.dataset['dataset']['py']
        _pw         = self.ip.dataset['dataset']['py']

        dvc = fastdist.matrix_to_matrix_distance(_features, np.matrix(ft_qry), fastdist.euclidean, "euclidean").flatten() # metric: 'euclidean' or 'cosine'

        if self.ego_known and self.DVC_WEIGHT.get() < 1: # then perform biasing via distance:
            spd = fastdist.matrix_to_matrix_distance(np.transpose(np.matrix([_px, _py])), np.matrix([self.vpr_ego[0], self.vpr_ego[1]]), fastdist.euclidean, "euclidean")
            spd_norm = spd/np.max(spd)
            dvc_norm = dvc/np.max(dvc)
            spd_x_dvc = ((1-self.DVC_WEIGHT.get())*spd_norm**2 + (self.DVC_WEIGHT.get())*dvc_norm) # TODO: vary bias with velocity, weighted sum
            mInd = int(np.argmin(spd_x_dvc))
            return mInd, spd_x_dvc
        else:
            mInd = int(np.argmin(dvc))
            return mInd, dvc
    
    def getTrueInd(self):
    # Compare measured odometry to reference odometry and find best match
        squares = np.square(np.array(self.ip.dataset['dataset']['px']) - self.label.gt_ego.x) + \
                            np.square(np.array(self.ip.dataset['dataset']['py']) - self.label.gt_ego.y)  # no point taking the sqrt; proportional
        trueInd = int(np.argmin(squares))

        return trueInd

    def publish_ros_info(self, tInd: int, mInd: int, dvc: list, gt_class: bool, gt_error: float, gt_mode: VPR_Tolerance_Mode):
    # Publish label and/or image feed

        time                        = rospy.Time.now()
        self.label.header.stamp     = time
        self.label.stamps.append(time) #type: ignore
        self.label.step             = self.label.CRUNCHER

        self.label.vpr_ego          = xyw(*self.vpr_ego)
        self.label.match_index      = mInd
        self.label.truth_index      = tInd

        self.label.gt_class         = gt_class
        self.label.gt_error         = gt_error
        self.label.gt_mode          = enum_name(gt_mode)

        self.label.distance_vector  = dvc

        odom_to_pub                         = Odometry()
        odom_to_pub.pose.pose.position.x    = self.vpr_ego[0]
        odom_to_pub.pose.pose.position.y    = self.vpr_ego[1]
        odom_to_pub.pose.pose.orientation   = q_from_yaw(self.vpr_ego[2])
        odom_to_pub.header.stamp            = time
        odom_to_pub.header.frame_id         = 'map'
        
        self.odom_estimate_pub.publish(odom_to_pub)
        self.vpr_label_pub.publish(self.label) # label publisher

    def extract(self, query: CompressedImage, feat_type: FeatureType, img_dims: list):
        if self.EXTRACT_SRV.get():
            requ            = DoExtractionRequest()
            requ.feat_type  = enum_name(feat_type)
            requ.img_dims   = list(img_dims)
            requ.input      = query
            resp            = self.srv_extraction(requ)

            if resp.success == False:
                raise Exception('[extract] Service executed, success=False!')
            out = uint8_list_to_np_ndarray(resp.output)
        else:
            ft_qry          = self.ip.getFeat(compressed2np(query), feat_type, use_tqdm=False, dims=img_dims)
            out             = ft_qry
        return out

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

    def loop_contents(self):

        if not (self.new_label and self.main_ready and self.parameters_ready): # denest
            self.print("Waiting.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return

        self.rate_obj.sleep()
        self.new_label  = False

        try:
            ft_qry      = self.extract(self.label.query_image, self.FEAT_TYPE.get(), self.IMG_DIMS.get())
        except rospy.service.ServiceException:
            self.print('Service not reachable; is dataset_trainer overloaded? Skipping this iteration...', LogType.WARN, throttle=2)
            return
        
        matchInd, dvc   = self.getMatchInd(ft_qry) #type: ignore # Find match
        self.vpr_ego    = [self.ip.dataset['dataset']['px'][matchInd], self.ip.dataset['dataset']['py'][matchInd], self.ip.dataset['dataset']['pw'][matchInd]]
        self.ego_known  = True
        trueInd         = self.getTrueInd() # find correct match based on shortest difference to measured odometry
        tolMode         = self.TOL_MODE.get()

        # Determine if we are within tolerance:
        if tolMode == VPR_Tolerance_Mode.METRE_CROW_TRUE:
            tolError = np.sqrt(np.square(self.ip.dataset['dataset']['px'][trueInd] - self.label.gt_ego.x) + \
                    np.square(self.ip.dataset['dataset']['py'][trueInd] - self.label.gt_ego.y)) 
        elif tolMode == VPR_Tolerance_Mode.METRE_CROW_MATCH:
            tolError = np.sqrt(np.square(self.ip.dataset['dataset']['px'][matchInd] - self.label.gt_ego.x) + \
                    np.square(self.ip.dataset['dataset']['py'][matchInd] - self.label.gt_ego.y)) 
        elif tolMode == VPR_Tolerance_Mode.METRE_LINE:
            tolError = np.sqrt(np.square(self.ip.dataset['dataset']['px'][trueInd] - self.ip.dataset['dataset']['px'][matchInd]) + \
                    np.square(self.ip.dataset['dataset']['py'][trueInd] - self.ip.dataset['dataset']['py'][matchInd])) 
        elif tolMode == VPR_Tolerance_Mode.FRAME:
            tolError = np.abs(matchInd - trueInd)
        else:
            raise Exception("Error: Unknown tolerance mode.")
        
        if tolError < self.TOL_THRES.get():
            tolState = True
        else:
            tolState = False
        
        # Make ROS messages
        self.publish_ros_info(trueInd, matchInd, dvc, tolState, tolError, tolMode)

def do_args():
    '''
    Helper to handle argparse items
    '''
    parser = ap.ArgumentParser(prog="vpr_cruncher",
                                description="ROS implementation of QVPR's VPR Primer",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    # Optional Arguments:except
    parser = base_optional_args(parser, node_name='vpr_cruncher')
    parser.add_argument('--use-gpu', '-G', type=check_bool, default=True,
                        help='Specify whether to use GPU (default: %(default)s).')
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
        # ros=False as rosnode likely terminated
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False)
    finally:
        # ros=False as rosnode likely terminated
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False)
        roslogger(formatException(), LogType.ERROR, ros=False)