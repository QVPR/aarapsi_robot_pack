#!/usr/bin/env python3

import rospy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String, Header

from cv_bridge import CvBridge
from fastdist import fastdist
import cv2
import numpy as np
import rospkg
import argparse as ap
import os
import sys

from aarapsi_robot_pack.srv import GenerateObj, GenerateObjResponse

from pyaarapsi.vpr_simple.imageprocessor_helpers import Tolerance_Mode, FeatureType
from pyaarapsi.vpr_simple.new_vpr_feature_tool   import VPRImageProcessor
from pyaarapsi.vpr_simple.vpr_image_methods      import labelImage, makeImage, grey2dToColourMap

from pyaarapsi.core.enum_tools                   import enum_name
from pyaarapsi.core.argparse_tools               import check_bounded_float, check_positive_float, check_positive_two_int_tuple, check_positive_int, check_bool, check_enum, check_string, check_string_list
from pyaarapsi.core.ros_tools                    import q_from_yaw, roslogger, get_ROS_message_types_dict, set_rospy_log_lvl, init_node, LogType, NodeState
from pyaarapsi.core.helper_tools                 import formatException

class mrc: # main ROS class
    def __init__(self, compress_in, compress_out, do_groundtruth, rate_num, namespace, node_name, anon, log_level, reset, order_id=0):
        
        init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30)

        self.init_params(rate_num, log_level, compress_in, compress_out, do_groundtruth, reset)
        self.init_vars()
        self.init_rospy()

        self.main_ready      = True
        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_params(self, rate_num, log_level, compress_in, compress_out, do_groundtruth, reset):
        
        self.FEAT_TYPE       = self.ROS_HOME.params.add(self.namespace + "/feature_type",        None,                   lambda x: check_enum(x, FeatureType),           force=False)
        self.IMG_DIMS        = self.ROS_HOME.params.add(self.namespace + "/img_dims",            None,                   check_positive_two_int_tuple,                   force=False)
        self.NPZ_DBP         = self.ROS_HOME.params.add(self.namespace + "/npz_dbp",             None,                   check_string,                                   force=False)
        self.BAG_DBP         = self.ROS_HOME.params.add(self.namespace + "/bag_dbp",             None,                   check_string,                                   force=False)
        self.IMG_TOPIC       = self.ROS_HOME.params.add(self.namespace + "/img_topic",           None,                   check_string,                                   force=False)
        self.ODOM_TOPIC      = self.ROS_HOME.params.add(self.namespace + "/odom_topic",          None,                   check_string,                                   force=False)
        
        self.REF_BAG_NAME    = self.ROS_HOME.params.add(self.namespace + "/ref/bag_name",        None,                   check_string,                                   force=False)
        self.REF_FILTERS     = self.ROS_HOME.params.add(self.namespace + "/ref/filters",         None,                   check_string,                                   force=False)
        self.REF_SAMPLE_RATE = self.ROS_HOME.params.add(self.namespace + "/ref/sample_rate",     None,                   check_positive_float,                           force=False) # Hz
        
        self.TOL_MODE        = self.ROS_HOME.params.add(self.namespace + "/tolerance/mode",      None,                   lambda x: check_enum(x, Tolerance_Mode),        force=False)
        self.TOL_THRES       = self.ROS_HOME.params.add(self.namespace + "/tolerance/threshold", None,                   check_positive_float,                           force=False)
        
        self.TIME_HIST_LEN   = self.ROS_HOME.params.add(self.nodespace + "/time_history_length", max(1,int(5*rate_num)), check_positive_int,                             force=reset)
        self.LOG_LEVEL       = self.ROS_HOME.params.add(self.nodespace + "/log_level",           log_level,              check_positive_int,                             force=reset)
        self.RATE_NUM        = self.ROS_HOME.params.add(self.nodespace + "/rate",                rate_num,               check_positive_float,                           force=reset) # Hz
        self.COMPRESS_IN     = self.ROS_HOME.params.add(self.nodespace + "/compress/in",         compress_in,            check_bool,                                     force=reset)
        self.COMPRESS_OUT    = self.ROS_HOME.params.add(self.nodespace + "/compress/out",        compress_out,           check_bool,                                     force=reset)
        self.GROUND_TRUTH    = self.ROS_HOME.params.add(self.nodespace + "/method/groundtruth",  do_groundtruth,         check_bool,                                     force=reset)
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

        self.INPUTS                 = get_ROS_message_types_dict(self.COMPRESS_IN.get())
        self.OUTPUTS                = get_ROS_message_types_dict(self.COMPRESS_OUT.get())

        self.bridge                 = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        # flags to denest main loop:
        self.new_query              = False # new query odom+image
        self.main_ready             = False # make sure everything commences together, safely
        self.ego_known              = False # whether or not an initial position has been found
        self.vpr_swap_pending       = False # whether we are waiting on a vpr dataset to be built

        self.time_history           = []

        # Process reference data (only needs to be done once)
        dataset_dict                = self.make_dataset_dict()
        try:
            self.image_processor    = VPRImageProcessor(bag_dbp=self.BAG_DBP.get(), npz_dbp=self.NPZ_DBP.get(), dataset=dataset_dict, try_gen=True, \
                                                        init_hybridnet=True, init_netvlad=True, cuda=True, autosave=True, printer=self.print)
            self.ref_dict           = self.image_processor.dataset['dataset']
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()
        
        self.rolling_mtrx_img       = np.zeros((len(self.ref_dict['px']),)*2) # Make empty similarity matrix figure

        self.generate_path(self.ref_dict['px'], self.ref_dict['py'], self.ref_dict['pw'], self.ref_dict['time'])

    def make_dataset_dict(self):
        return dict(bag_name=self.REF_BAG_NAME.get(), odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], \
                    sample_rate=self.REF_SAMPLE_RATE.get(), ft_types=[self.FEAT_TYPE.get()], img_dims=self.IMG_DIMS.get(), filters={})

    def init_rospy(self):
        self.last_time              = rospy.Time.now()
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())

        self.odom_estimate_pub      = self.ROS_HOME.add_pub(self.namespace + "/vpr_odom", Odometry, queue_size=1)
        self.path_pub               = self.ROS_HOME.add_pub(self.namespace + '/path', Path, queue_size=1)
        self.vpr_feed_pub           = self.ROS_HOME.add_pub(self.namespace + "/image" + self.OUTPUTS['topic'], self.OUTPUTS['image'], queue_size=1)
        self.vpr_label_pub          = self.ROS_HOME.add_pub(self.namespace + "/label" + self.OUTPUTS['topic'], self.OUTPUTS['label'], queue_size=1)
        self.rolling_mtrx           = self.ROS_HOME.add_pub(self.namespace + "/matrices/rolling" + self.OUTPUTS['topic'], self.OUTPUTS['image'], queue_size=1)

        self.data_sub               = rospy.Subscriber(self.namespace + "/img_odom" + self.INPUTS['topic'], self.INPUTS['data'], self.data_callback, queue_size=1)
        self.param_checker_sub      = rospy.Subscriber(self.namespace + "/params_update", String, self.param_callback, queue_size=100)
        self.send_path_plan         = rospy.Service(self.namespace + '/path', GenerateObj, self.handle_GetPathPlan)

    def handle_GetPathPlan(self, req):
    # /vpr_nodes/path service
        ans = GenerateObjResponse()
        success = True

        try:
            if req.generate == True:
                self.generate_path(self.ref_dict['px'], self.ref_dict['py'], self.ref_dict['pw'], self.ref_dict['time'])
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

    def update_VPR(self, param_to_change):
        dataset_dict = self.make_dataset_dict()
        if not self.image_processor.swap(dataset_dict, generate=False):
            self.print("VPR reference data swap failed. Previous set will be retained (changed ROS parameter will revert)", LogType.WARN)
            param_to_change.revert()
            self.print(param_to_change.value)
            self.vpr_swap_pending = True
        else:
            self.print("VPR reference data swapped.", LogType.INFO)
            self.vpr_swap_pending = False

    def param_callback(self, msg):
        if self.ROS_HOME.params.exists(msg.data):
            if not self.ROS_HOME.params.update(msg.data):
                self.print("Change to parameter [%s]; bad value." % msg.data, LogType.DEBUG)
                return
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
                self.update_VPR(param)
            except:
                self.print(formatException(), LogType.ERROR)
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)

    def data_callback(self, msg):
    # /data/img_odom (aarapsi_robot_pack/(Compressed)ImageOdom)

        self.ego                    = [round(msg.odom.pose.pose.position.x, 3), round(msg.odom.pose.pose.position.y, 3), round(msg.odom.pose.pose.position.z, 3)]
        if self.COMPRESS_IN.get():
            self.store_query_raw    = self.bridge.compressed_imgmsg_to_cv2(msg.image, "bgr8")
        else:
            self.store_query_raw    = self.bridge.imgmsg_to_cv2(msg.image, "passthrough")
        self.store_query            = self.store_query_raw
        self.new_query              = True
        
    def getMatchInd(self, ft_qry):
    # top matching reference index for query

        dvc = fastdist.matrix_to_matrix_distance(self.ref_dict[enum_name(self.FEAT_TYPE.get())], \
                                                 np.matrix(ft_qry),\
                                                 fastdist.euclidean, "euclidean") # metric: 'euclidean' or 'cosine'

        if self.ego_known and self.DVC_WEIGHT.get() < 1: # then perform biasing via distance:
            spd = fastdist.matrix_to_matrix_distance(np.transpose(np.matrix([self.ref_dict['px'], self.ref_dict['py']])), \
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
        squares = np.square(np.array(self.ref_dict['px']) - self.ego[0]) + \
                            np.square(np.array(self.ref_dict['py']) - self.ego[1])  # no point taking the sqrt; proportional
        trueInd = np.argmin(squares)

        return trueInd

    def publish_ros_info(self, cv2_img, tInd, mInd, dvc, state):
    # Publish label and/or image feed

        self.rolling_mtrx_img = np.delete(self.rolling_mtrx_img, 0, 1) # delete first column (oldest query)
        self.rolling_mtrx_img = np.concatenate((self.rolling_mtrx_img, dvc), 1)

        mtrx_rgb = grey2dToColourMap(self.rolling_mtrx_img, dims=(500,500), colourmap=cv2.COLORMAP_JET)

        if self.COMPRESS_OUT.get():
            ros_image_to_pub = self.bridge.cv2_to_compressed_imgmsg(cv2_img, "jpeg") # jpeg (png slower)
            ros_matrix_to_pub = self.bridge.cv2_to_compressed_imgmsg(mtrx_rgb, "jpeg") # jpeg (png slower)
        else:
            ros_image_to_pub = self.bridge.cv2_to_imgmsg(cv2_img, "bgr8")
            ros_matrix_to_pub = self.bridge.cv2_to_imgmsg(mtrx_rgb, "bgr8")
        struct_to_pub = self.OUTPUTS['label']()

        ros_image_to_pub.header.stamp = rospy.Time.now()
        ros_image_to_pub.header.frame_id = 'map'

        struct_to_pub.queryImage        = ros_image_to_pub
        struct_to_pub.data.odom.x       = self.ego[0]
        struct_to_pub.data.odom.y       = self.ego[1]
        struct_to_pub.data.odom.z       = self.ego[2]
        struct_to_pub.data.dvc          = dvc
        struct_to_pub.data.matchId      = mInd
        struct_to_pub.data.trueId       = tInd
        struct_to_pub.data.state        = state
        struct_to_pub.header.frame_id   = 'map'
        struct_to_pub.header.stamp      = rospy.Time.now()

        odom_to_pub = Odometry()
        odom_to_pub.pose.pose.position.x = self.ref_dict['px'][mInd]
        odom_to_pub.pose.pose.position.y = self.ref_dict['py'][mInd]
        odom_to_pub.pose.pose.orientation = q_from_yaw(self.ref_dict['pw'][mInd])
        odom_to_pub.header.stamp = rospy.Time.now()
        odom_to_pub.header.frame_id = 'map'

        self.rolling_mtrx.publish(ros_matrix_to_pub)
        self.odom_estimate_pub.publish(odom_to_pub)
        self.vpr_feed_pub.publish(ros_image_to_pub) # image feed publisher
        self.vpr_label_pub.publish(struct_to_pub) # label publisher

        self.vpr_ego = [self.ref_dict['px'][mInd], self.ref_dict['py'][mInd], self.ref_dict['pw'][mInd]]
        self.ego_known = True

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

    def main(self):
        # Main loop process
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.loop_contents()

    def loop_contents(self):

        if not (self.new_query and self.main_ready): # denest
            self.print("Waiting for a new query.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return
        self.rate_obj.sleep()
        self.new_query  = False

        ft_qry          = self.image_processor.getFeat(self.store_query, self.FEAT_TYPE.get(), use_tqdm=False, dims=self.IMG_DIMS.get())
        matchInd, dvc   = self.getMatchInd(ft_qry) # Find match
        ft_ref          = self.ref_dict[enum_name(self.FEAT_TYPE.get())][matchInd]
        trueInd         = -1 #default; can't be negative.
        gt_string       = ""
        tolState        = 0

        if self.GROUND_TRUTH.get():
            trueInd = self.getTrueInd() # find correct match based on shortest difference to measured odometry
            # Determine if we are within tolerance:
            if self.TOL_MODE.get() == Tolerance_Mode.METRE_CROW_TRUE:
                tolError = np.sqrt(np.square(self.ref_dict['px'][trueInd] - self.ego[0]) + \
                        np.square(self.ref_dict['py'][trueInd] - self.ego[1])) 
                tolString = "MCT"
            elif self.TOL_MODE.get() == Tolerance_Mode.METRE_CROW_MATCH:
                tolError = np.sqrt(np.square(self.ref_dict['px'][matchInd] - self.ego[0]) + \
                        np.square(self.ref_dict['py'][matchInd] - self.ego[1])) 
                tolString = "MCM"
            elif self.TOL_MODE.get() == Tolerance_Mode.METRE_LINE:
                tolError = np.sqrt(np.square(self.ref_dict['px'][trueInd] - self.ref_dict['px'][matchInd]) + \
                        np.square(self.ref_dict['py'][trueInd] - self.ref_dict['py'][matchInd])) 
                tolString = "ML"
            elif self.TOL_MODE.get() == Tolerance_Mode.FRAME:
                tolError = np.abs(matchInd - trueInd)
                tolString = "F"
            else:
                raise Exception("Error: Unknown tolerance mode.")

            if tolError < self.TOL_THRES.get():
                self.ICON_DICT['icon'] = self.ICON_DICT['good']
                tolState = 2
            else:
                self.ICON_DICT['icon'] = self.ICON_DICT['poor']
                tolState = 1

            gt_string = ", Error: %2.2f%s" % (tolError, tolString)

        cv2_image_to_pub = makeImage(ft_qry, ft_ref, self.ICON_DICT)
        
        # Measure timing for recalculating average rate:
        this_time = rospy.Time.now()
        time_diff = this_time - self.last_time
        self.last_time = this_time
        self.time_history.append(time_diff.to_sec())
        num_time = len(self.time_history)
        if num_time > self.TIME_HIST_LEN.get():
            self.time_history.pop(0)
        time_average = sum(self.time_history) / num_time

        # add label with feed information to image:
        label_string = ("Index [%04d], %2.2f Hz" % (matchInd, 1/time_average)) + gt_string
        cv2_img_lab = labelImage(cv2_image_to_pub, label_string, (20, cv2_image_to_pub.shape[0] - 40), (100, 255, 100))

        img_to_pub = cv2_img_lab
        
        # Make ROS messages
        self.publish_ros_info(img_to_pub, trueInd, matchInd, dvc, tolState)

    def exit(self):
        self.print("Quit received.", LogType.INFO)
        sys.exit()

def do_args():
    parser = ap.ArgumentParser(prog="vpr_cruncher", 
                                description="ROS implementation of QVPR's VPR Primer",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser.add_argument('--compress-in',      '-Ci', type=check_bool,                   default=False,            help='Enable image compression on input (default: %(default)s).')
    parser.add_argument('--compress-out',     '-Co', type=check_bool,                   default=False,            help='Enable image compression on output (default: %(default)s).')
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
        nmrc = mrc(compress_in=args['compress_in'], compress_out=args['compress_out'], \
                    do_groundtruth=args['groundtruth'], rate_num=args['rate'], namespace=args['namespace'], \
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