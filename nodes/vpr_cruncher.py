#!/usr/bin/env python3

import rospy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String, Header

from fastdist import fastdist
import cv2
import numpy as np
import rospkg
import argparse as ap
import os
import sys

from rospy_message_converter import message_converter
from aarapsi_robot_pack.srv import GenerateObj, GenerateObjResponse, DoExtraction, DoExtractionRequest
from aarapsi_robot_pack.msg import RequestDataset, ResponseDataset, xyw, ImageOdom, ImageLabelDetails

from pyaarapsi.vpr_simple.vpr_helpers            import Tolerance_Mode, FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool       import VPRDatasetProcessor

from pyaarapsi.core.enum_tools                   import enum_name
from pyaarapsi.core.argparse_tools               import check_bounded_float, check_positive_float, check_positive_two_int_list, check_positive_int, check_bool, check_enum, check_string
from pyaarapsi.core.ros_tools                    import yaw_from_q, q_from_yaw, roslogger, set_rospy_log_lvl, init_node, LogType, NodeState
from pyaarapsi.core.helper_tools                 import formatException, np_ndarray_to_uint8_list, uint8_list_to_np_ndarray, vis_dict

class mrc: # main ROS class
    def __init__(self, do_groundtruth, rate_num, namespace, node_name, anon, log_level, reset, order_id=0):
        
        if not init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30):
            sys.exit()

        self.init_params(rate_num, log_level, do_groundtruth, reset)
        self.init_vars()
        self.init_rospy()

        self.main_ready      = True
        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_params(self, rate_num, log_level, do_groundtruth, reset):
        
        self.FEAT_TYPE       = self.ROS_HOME.params.add(self.namespace + "/feature_type",        None,                   lambda x: check_enum(x, FeatureType),           force=False)
        self.IMG_DIMS        = self.ROS_HOME.params.add(self.namespace + "/img_dims",            None,                   check_positive_two_int_list,                    force=False)
        self.NPZ_DBP         = self.ROS_HOME.params.add(self.namespace + "/npz_dbp",             None,                   check_string,                                   force=False)
        self.BAG_DBP         = self.ROS_HOME.params.add(self.namespace + "/bag_dbp",             None,                   check_string,                                   force=False)
        self.IMG_TOPIC       = self.ROS_HOME.params.add(self.namespace + "/img_topic",           None,                   check_string,                                   force=False)
        self.ODOM_TOPIC      = self.ROS_HOME.params.add(self.namespace + "/odom_topic",          None,                   check_string,                                   force=False)
        
        self.REF_BAG_NAME    = self.ROS_HOME.params.add(self.namespace + "/ref/bag_name",        None,                   check_string,                                   force=False)
        self.REF_FILTERS     = self.ROS_HOME.params.add(self.namespace + "/ref/filters",         None,                   check_string,                                   force=False)
        self.REF_SAMPLE_RATE = self.ROS_HOME.params.add(self.namespace + "/ref/sample_rate",     None,                   check_positive_float,                           force=False) # Hz
        
        self.TOL_MODE        = self.ROS_HOME.params.add(self.namespace + "/tolerance/mode",      None,                   lambda x: check_enum(x, Tolerance_Mode),        force=False)
        self.TOL_THRES       = self.ROS_HOME.params.add(self.namespace + "/tolerance/threshold", None,                   check_positive_float,                           force=False)
        
        self.GROUND_TRUTH    = self.ROS_HOME.params.add(self.namespace + "/groundtruth",         do_groundtruth,         check_bool,                                     force=False)
        
        self.TIME_HIST_LEN   = self.ROS_HOME.params.add(self.nodespace + "/time_history_length", max(1,int(5*rate_num)), check_positive_int,                             force=reset)
        self.LOG_LEVEL       = self.ROS_HOME.params.add(self.nodespace + "/log_level",           log_level,              check_positive_int,                             force=reset)
        self.RATE_NUM        = self.ROS_HOME.params.add(self.nodespace + "/rate",                rate_num,               check_positive_float,                           force=reset) # Hz
        self.DVC_WEIGHT      = self.ROS_HOME.params.add(self.nodespace + "/dvc_weight",          1,                      lambda x: check_bounded_float(x, 0, 1, 'both'), force=reset) # Hz
        
        self.REF_DATA_PARAMS = [self.NPZ_DBP, self.BAG_DBP, self.REF_BAG_NAME, self.REF_FILTERS, self.REF_SAMPLE_RATE, self.IMG_TOPIC, self.ODOM_TOPIC, self.FEAT_TYPE, self.IMG_DIMS]
        self.REF_DATA_NAMES  = [i.name for i in self.REF_DATA_PARAMS]

    def init_vars(self):
        self.ICON_SIZE              = 50
        self.ICON_DIST              = 20
        self.ICON_PATH              = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/media"
        self.ICON_DICT              = {'size': self.ICON_SIZE, 'dist': self.ICON_DIST, 'icon': [], 'good': [], 'poor': []}

        good_icon                   = cv2.imread(self.ICON_PATH + "/tick.png", cv2.IMREAD_UNCHANGED)
        poor_icon                   = cv2.imread(self.ICON_PATH + "/cross.png", cv2.IMREAD_UNCHANGED)
        self.ICON_DICT['good']      = cv2.resize(good_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)
        self.ICON_DICT['poor']      = cv2.resize(poor_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)

        self.ego                    = [0.0, 0.0, 0.0] # ground truth robot position
        self.vpr_ego                = [0.0, 0.0, 0.0] # our estimate of robot position

        # flags to denest main loop:
        self.new_query              = False # new query odom+image
        self.main_ready             = False # make sure everything commences together, safely
        self.ego_known              = False # whether or not an initial position has been found
        self.dataset_swap_pending   = False # whether we are waiting on a vpr dataset to be built
        self.parameters_ready       = True # Separate main loop errors from those due to parameter updates

        self.dataset_requests       = []
        self.time_history           = []

        # Process reference data
        dataset_dict                = self.make_dataset_dict()
        try:
            self.image_processor    = VPRDatasetProcessor(dataset_dict, try_gen=False, ros=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

        self.generate_path(self.image_processor.dataset['dataset']['px'], self.image_processor.dataset['dataset']['py'], self.image_processor.dataset['dataset']['pw'], self.image_processor.dataset['dataset']['time'])

    def make_dataset_dict(self):
        return dict(bag_name=self.REF_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                    odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.REF_SAMPLE_RATE.get(), \
                    ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')

    def init_rospy(self):
        self.last_time              = rospy.Time.now()
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())

        self.odom_estimate_pub      = self.ROS_HOME.add_pub(self.namespace + "/vpr_odom",                   Odometry,           queue_size=1)
        self.path_pub               = self.ROS_HOME.add_pub(self.namespace + '/path',                       Path,               queue_size=1)
        self.vpr_label_pub          = self.ROS_HOME.add_pub(self.namespace + "/label",                      ImageLabelDetails,  queue_size=1)
        self.dataset_request_pub    = self.ROS_HOME.add_pub(self.namespace + "/requests/dataset/request",   RequestDataset,     queue_size=1)
        self.dataset_request_sub    = rospy.Subscriber(     self.namespace + '/requests/dataset/ready',     ResponseDataset,    self.dataset_request_callback,  queue_size=1)
        self.data_sub               = rospy.Subscriber(     self.namespace + "/img_odom",                   ImageOdom,          self.data_callback,             queue_size=1)
        self.param_checker_sub      = rospy.Subscriber(     self.namespace + "/params_update",              String,             self.param_callback,            queue_size=100)
        self.send_path_plan         = rospy.Service(        self.namespace + '/path',                       GenerateObj,        self.handle_GetPathPlan)
        self.srv_extraction         = rospy.ServiceProxy(   self.namespace + '/do_extraction',              DoExtraction)

    def dataset_request_callback(self, msg):
        pass # TODO

    def debug_cb(self, msg):
        if msg.node_name == self.node_name:
            if msg.instruction == 0:
                self.print(self.make_dataset_dict(), LogType.DEBUG)
            elif msg.instruction == 1:
                self.print(vis_dict(self.image_processor.dataset), LogType.DEBUG)
            else:
                self.print(msg.instruction, LogType.DEBUG)

    def handle_GetPathPlan(self, req):
    # /vpr_nodes/path service
        ans = GenerateObjResponse()
        success = True

        try:
            if req.generate == True:
                self.generate_path(self.image_processor.dataset['dataset']['px'], self.image_processor.dataset['dataset']['py'], self.image_processor.dataset['dataset']['pw'], self.image_processor.dataset['dataset']['time'])
            self.path_pub.publish(self.path_msg)
        except:
            success = False

        ans.success = success
        ans.topic = self.namespace + "/path"
        self.print("Service requested [Gen=%s], Success=%s" % (str(req.generate), str(success)), LogType.DEBUG)
        return ans

    def generate_path(self, px, py, pw, time):
        self.path_msg = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        for (c, (x, y, w, t)) in enumerate(zip(px, py, pw, time)):
            if not c % 3 == 0:
                continue
            new_pose = PoseStamped(header=Header(stamp=rospy.Time.from_sec(t), frame_id="map", seq=c))
            new_pose.pose.position = Point(x=x, y=y, z=0)
            new_pose.pose.orientation = q_from_yaw(w)
            self.path_msg.poses.append(new_pose)
            del new_pose

    def update_VPR(self):
        dataset_dict = self.make_dataset_dict()
        if not self.image_processor.swap(dataset_dict, generate=False, allow_false=True):
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
                        param.revert()
                except IndexError:
                    pass
                except:
                    param.revert()
                    self.print(formatException(), LogType.ERROR)
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)
        self.parameters_ready = True

    def data_callback(self, msg):
    # /data/img_odom (aarapsi_robot_pack/ImageOdom)

        self.ego                    = [round(msg.odom.pose.pose.position.x, 3), round(msg.odom.pose.pose.position.y, 3), round(yaw_from_q(msg.odom.pose.pose.orientation), 3)]
        self.store_query            = uint8_list_to_np_ndarray(msg.image)
        self.new_query              = True
        
    def getMatchInd(self, ft_qry):
    # top matching reference index for query

        dvc = fastdist.matrix_to_matrix_distance(self.image_processor.dataset['dataset'][enum_name(self.FEAT_TYPE.get())], \
                                                 np.matrix(ft_qry),\
                                                 fastdist.euclidean, "euclidean") # metric: 'euclidean' or 'cosine'

        if self.ego_known and self.DVC_WEIGHT.get() < 1: # then perform biasing via distance:
            spd = fastdist.matrix_to_matrix_distance(np.transpose(np.matrix([self.image_processor.dataset['dataset']['px'], self.image_processor.dataset['dataset']['py']])), \
                                                     np.matrix([self.vpr_ego[0], self.vpr_ego[1]]), \
                                                     fastdist.euclidean, "euclidean")
            spd_norm = spd/np.max(spd[:]) 
            dvc_norm = dvc/np.max(dvc[:])
            spd_x_dvc = ((1-self.DVC_WEIGHT.get())*spd_norm**2 + (self.DVC_WEIGHT.get())*dvc_norm) # TODO: vary bias with velocity, weighted sum

            mInd = np.argmin(spd_x_dvc[:])
            return mInd, spd_x_dvc
        else:
            mInd = np.argmin(dvc[:])
            return mInd, dvc
    
    def getTrueInd(self):
    # Compare measured odometry to reference odometry and find best match
        squares = np.square(np.array(self.image_processor.dataset['dataset']['px']) - self.ego[0]) + \
                            np.square(np.array(self.image_processor.dataset['dataset']['py']) - self.ego[1])  # no point taking the sqrt; proportional
        trueInd = np.argmin(squares)

        return trueInd

    def publish_ros_info(self, tInd, mInd, dvc, state):
    # Publish label and/or image feed

        struct_to_pub                   = ImageLabelDetails()

        self.vpr_ego                    = [self.image_processor.dataset['dataset']['px'][mInd], self.image_processor.dataset['dataset']['py'][mInd], self.image_processor.dataset['dataset']['pw'][mInd]]
        self.ego_known                  = True

        struct_to_pub.queryImage        = np_ndarray_to_uint8_list(self.store_query)
        struct_to_pub.data.gt_ego       = xyw(x=self.ego[0], y=self.ego[1], w=self.ego[2])
        struct_to_pub.data.vpr_ego      = xyw(x=self.vpr_ego[0], y=self.vpr_ego[1], w=self.vpr_ego[2])
        struct_to_pub.data.dvc          = dvc
        struct_to_pub.data.matchId      = mInd
        struct_to_pub.data.trueId       = tInd
        struct_to_pub.data.state        = state
        struct_to_pub.header.frame_id   = 'map'
        struct_to_pub.header.stamp      = rospy.Time.now()

        odom_to_pub                         = Odometry()
        odom_to_pub.pose.pose.position.x    = self.image_processor.dataset['dataset']['px'][mInd]
        odom_to_pub.pose.pose.position.y    = self.image_processor.dataset['dataset']['py'][mInd]
        odom_to_pub.pose.pose.orientation   = q_from_yaw(self.image_processor.dataset['dataset']['pw'][mInd])
        odom_to_pub.header.stamp            = rospy.Time.now()
        odom_to_pub.header.frame_id         = 'map'
        
        self.odom_estimate_pub.publish(odom_to_pub)
        self.vpr_label_pub.publish(struct_to_pub) # label publisher

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

    def extract(self, array, feat_type, img_dims):
        if not self.main_ready:
            return
        requ            = DoExtractionRequest()
        requ.feat_type  = enum_name(feat_type)
        requ.img_dims   = list(img_dims)
        requ.input      = np_ndarray_to_uint8_list(array)
        resp = self.srv_extraction(requ)
        if resp.success == False:
            raise Exception('[extract] Service executed, success=False!')
        return uint8_list_to_np_ndarray(resp.output)

    def main(self):
        # Main loop process
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            try:
                self.loop_contents()
            except Exception as e:
                if self.parameters_ready:
                    self.print(vis_dict(self.image_processor.dataset))
                    raise Exception('Critical failure. ' + formatException()) from e
                else:
                    self.print('Main loop exception, attempting to handle; waiting for parameters to update. Details:\n' + formatException(), LogType.DEBUG, throttle=5)
                    rospy.sleep(0.5)

    def loop_contents(self):

        if not (self.new_query and self.main_ready and self.parameters_ready): # denest
            self.print("Waiting.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return
        
        self.rate_obj.sleep()
        self.new_query  = False

        ft_qry          = self.extract(self.store_query, self.FEAT_TYPE.get(), self.IMG_DIMS.get())
        matchInd, dvc   = self.getMatchInd(ft_qry) # Find match
        trueInd         = -1 #default; can't be negative.
        tolState        = 0

        if self.GROUND_TRUTH.get():
            trueInd = self.getTrueInd() # find correct match based on shortest difference to measured odometry
            tolMode = self.TOL_MODE.get()
            # Determine if we are within tolerance:
            if tolMode == Tolerance_Mode.METRE_CROW_TRUE:
                tolError = np.sqrt(np.square(self.image_processor.dataset['dataset']['px'][trueInd] - self.ego[0]) + \
                        np.square(self.image_processor.dataset['dataset']['py'][trueInd] - self.ego[1])) 
            elif tolMode == Tolerance_Mode.METRE_CROW_MATCH:
                tolError = np.sqrt(np.square(self.image_processor.dataset['dataset']['px'][matchInd] - self.ego[0]) + \
                        np.square(self.image_processor.dataset['dataset']['py'][matchInd] - self.ego[1])) 
            elif tolMode == Tolerance_Mode.METRE_LINE:
                tolError = np.sqrt(np.square(self.image_processor.dataset['dataset']['px'][trueInd] - self.image_processor.dataset['dataset']['px'][matchInd]) + \
                        np.square(self.image_processor.dataset['dataset']['py'][trueInd] - self.image_processor.dataset['dataset']['py'][matchInd])) 
            elif tolMode == Tolerance_Mode.FRAME:
                tolError = np.abs(matchInd - trueInd)
            else:
                raise Exception("Error: Unknown tolerance mode.")

            if tolError < self.TOL_THRES.get():
                self.ICON_DICT['icon'] = self.ICON_DICT['good']
                tolState = 2
            else:
                self.ICON_DICT['icon'] = self.ICON_DICT['poor']
                tolState = 1
        
        # Make ROS messages
        self.publish_ros_info(trueInd, matchInd, dvc, tolState)

    def exit(self):
        self.print("Quit received.", LogType.INFO)
        sys.exit()

def do_args():
    parser = ap.ArgumentParser(prog="vpr_cruncher", 
                                description="ROS implementation of QVPR's VPR Primer",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser.add_argument('--groundtruth',      '-G',  type=check_bool,                   default=False,            help='Enable groundtruth inclusion (default: %(default)s).')
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=10.0,             help='Set node rate (default: %(default)s).')
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="vpr_cruncher",   help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=True,             help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset',            '-R',  type=check_bool,                   default=False,            help='Force reset of parameters to specified ones (default: %(default)s).')
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')

    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = mrc(do_groundtruth=args['groundtruth'], rate_num=args['rate'], namespace=args['namespace'], \
                   node_name=args['node_name'], anon=args['anon'], log_level=args['log_level'], reset=args['reset'], order_id=args['order_id']\
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