#!/usr/bin/env python3

import rospy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Header

from cv_bridge import CvBridge
import cv2
import numpy as np
import rospkg
from matplotlib import pyplot as plt
import argparse as ap
import os
import sys

from scipy.spatial.distance import cdist

from aarapsi_robot_pack.msg import ImageLabelStamped, CompressedImageLabelStamped, ImageOdom, CompressedImageOdom # Our custom msg structures
from aarapsi_robot_pack.srv import GenerateObj, GenerateObjResponse
from pyaarapsi.vpr_simple import VPRImageProcessor, Tolerance_Mode, FeatureType, labelImage, makeImage, grey2dToColourMap, \
                                            doMtrxFig, updateMtrxFig, doDVecFig, updateDVecFig, doOdomFig, updateOdomFig

from pyaarapsi.core.enum_tools import enum_value_options, enum_get, enum_name
from pyaarapsi.core.argparse_tools import check_bounded_float, check_positive_float, check_positive_two_int_tuple, check_positive_int, \
                                            check_bool, check_str_list, check_enum, check_string
from pyaarapsi.core.ros_tools import ROS_Param_Server, q_from_yaw, Heartbeat, NodeState, roslogger, LogType

class mrc: # main ROS class
    def __init__(self, database_path, ref_images_path, ref_odometry_path, data_topic, dataset_name, odom_topic,\
                    compress_in, compress_out, do_plotting, do_image, do_groundtruth, do_label, rate_num, ft_type, \
                    img_dims, icon_settings, tolerance_threshold, tolerance_mode, match_metric, namespace, \
                    time_history_length, frame_id, node_name, anon, log_level, reset):
        
        self.NAMESPACE              = namespace
        self.NODENAME               = node_name
        self.NODESPACE              = self.NAMESPACE + "/" + self.NODENAME
        rospy.init_node(self.NODENAME, anonymous=anon, log_level=log_level)
        self.init_params(rate_num, ft_type, img_dims, database_path, dataset_name, ref_images_path, \
                         ref_odometry_path, tolerance_mode, tolerance_threshold, match_metric, time_history_length, \
                         frame_id, compress_in, compress_out, do_plotting, do_image, do_groundtruth, do_label, reset)
        self.init_vars(icon_settings)
        roslogger('Starting %s node.' % (node_name), LogType.INFO, ros=True)
        self.heartbeat              = Heartbeat(self.NODENAME, self.NAMESPACE, NodeState.INIT, self.RATE_NUM.get())
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())

        # Process reference data (only needs to be done once)
        self.image_processor        = VPRImageProcessor(ros=True, init_hybridnet=True, init_netvlad=True, cuda=True, dims=self.IMG_DIMS.get())
        try:
            self.ref_dict           = self.image_processor.npzDatabaseLoadSave(self.DATABASE_PATH.get(), self.REF_DATA_NAME.get(), \
                                                                                self.REF_IMG_PATH.get(), self.REF_ODOM_PATH.get(), \
                                                                                self.FEAT_TYPE.get(), self.IMG_DIMS.get(), do_save=False)
        except:
            self.exit()

        self.generate_path(self.ref_dict['odom']['position'], self.ref_dict['times'])
        self.init_rospy(data_topic, odom_topic)

    def init_rospy(self, data_topic, odom_topic):
        self.param_checker_sub      = rospy.Subscriber(self.NAMESPACE + "/params_update", String, self.param_callback, queue_size=100)
        self.odom_estimate_pub      = rospy.Publisher(self.NAMESPACE + "/vpr_odom", Odometry, queue_size=1)

        self.send_path_plan         = rospy.Service(self.NAMESPACE + '/path', GenerateObj, self.handle_GetPathPlan)
        self.path_pub               = rospy.Publisher(self.NAMESPACE + '/path', Path, queue_size=1)

        if self.MAKE_IMAGE.get():
            self.vpr_feed_pub       = rospy.Publisher(self.NAMESPACE + "/image" + self.OUTPUTS['topic'], self.OUTPUTS['image'], queue_size=1)
            good_icon = cv2.imread(self.ICON_PATH + "/tick.png", cv2.IMREAD_UNCHANGED)
            poor_icon = cv2.imread(self.ICON_PATH + "/cross.png", cv2.IMREAD_UNCHANGED)
            self.ICON_DICT['good']  = cv2.resize(good_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)
            self.ICON_DICT['poor']  = cv2.resize(poor_icon, (self.ICON_SIZE, self.ICON_SIZE), interpolation = cv2.INTER_AREA)

        if self.MAKE_LABEL.get():
            self.FEED_TOPIC         = data_topic
            self.ODOM_TOPIC         = odom_topic
            if (self.ODOM_TOPIC is None) or (self.ODOM_TOPIC == ''):
                self.data_sub       = rospy.Subscriber(self.FEED_TOPIC + self.INPUTS['topic'], self.INPUTS['data'], self.data_callback, queue_size=1) 
            else:
                self.img_sub        = rospy.Subscriber(self.FEED_TOPIC + self.INPUTS['topic'], self.INPUTS['image'], self.img_callback, queue_size=1) 
                self.odom_sub       = rospy.Subscriber(self.ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)
            
            self.vpr_label_pub      = rospy.Publisher(self.NAMESPACE + "/label" + self.OUTPUTS['topic'], self.OUTPUTS['label'], queue_size=1)
            self.rolling_mtrx       = rospy.Publisher(self.NAMESPACE + "/matrices/rolling" + self.OUTPUTS['topic'], self.OUTPUTS['image'], queue_size=1)
            self.rolling_mtrx_img   = np.zeros((len(self.ref_dict['odom']['position']['x']), len(self.ref_dict['odom']['position']['x']))) # Make similarity matrix figure
        else:
            self.vpr_label_sub      = rospy.Subscriber(self.NAMESPACE + "/label" + self.INPUTS['topic'], self.INPUTS['label'], self.label_callback, queue_size=1)

        if self.DO_PLOTTING.get():
            self.fig, self.axes     = plt.subplots(1, 3, figsize=(15,4))
            self.timer_plot         = rospy.Timer(rospy.Duration(0.1), self.timer_plot_callback) # 10 Hz; Plot rate limiter
            # Prepare figures:
            self.fig.suptitle("Odometry Visualised")
            self.fig_mtrx_handles   = doMtrxFig(self.axes[0], self.ref_dict['odom']) # Make simularity matrix figure
            self.fig_dvec_handles   = doDVecFig(self.axes[1], self.ref_dict['odom']) # Make distance vector figure
            self.fig_odom_handles   = doOdomFig(self.axes[2], self.ref_dict['odom']) # Make odometry figure
            self.fig.show()

        # Last item as it sets a flag that enables main loop execution.
        self.main_timer             = rospy.Timer(rospy.Duration(1/self.RATE_NUM.get()), self.main_cb) # Main loop rate limiter

    def init_vars(self, icon_settings):
        self.ICON_SIZE              = icon_settings[0]
        self.ICON_DIST              = icon_settings[1]
        self.ICON_PATH              = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/media"
        self.ICON_DICT              = {'size': self.ICON_SIZE, 'dist': self.ICON_DIST, 'icon': [], 'good': [], 'poor': []}

        self.ego                    = [0.0, 0.0, 0.0] # ground truth robot position
        self.vpr_ego                = [0.0, 0.0, 0.0] # our estimate of robot position

        self.IMG_FOLDER             = 'forward'

        self._compress_on           = {'topic': "/compressed", 'image': CompressedImage, 'label': CompressedImageLabelStamped, 'data': CompressedImageOdom}
        self._compress_off          = {'topic': "", 'image': Image, 'label': ImageLabelStamped, 'data': ImageOdom}

        # Handle ROS details for input topics:
        if self.COMPRESS_IN.get():
            self.INPUTS             = self._compress_on
        else:
            self.INPUTS             = self._compress_off
        # Handle ROS details for output topics:
        if self.COMPRESS_OUT.get():
            self.OUTPUTS            = self._compress_on
        else:
            self.OUTPUTS            = self._compress_off

        self.bridge                 = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        # flags to denest main loop:
        self.new_query              = False # new query image (MAKE_LABEL.get()==True) or new label received (MAKE_LABEL.get()==False)
        self.new_odom               = False # new odometry set
        self.main_ready             = False # rate limiter via timer
        self.do_show                = False # plot rate limiter  via timer
        self.ego_known              = False # whether or not an initial position has been found
        self.vpr_swap_pending       = False # whether we are waiting on a vpr dataset to be built

        self.last_time              = rospy.Time.now()
        self.time_history           = []

    def init_params(self, rate_num, ft_type, img_dims, database_path, dataset_name, ref_images_path, \
                    ref_odometry_path, tolerance_mode, tolerance_threshold, match_metric, time_history_length, \
                    frame_id, compress_in, compress_out, do_plotting, do_image, do_groundtruth, do_label, reset):
        self.PARAM_SERVER           = ROS_Param_Server()
        self.RATE_NUM               = self.PARAM_SERVER.add(self.NODESPACE + "/rate",                rate_num,                  check_positive_float,                                           force=reset) # Hz

        self.FEAT_TYPE              = self.PARAM_SERVER.add(self.NAMESPACE + "/feature_type",        enum_name(ft_type),        lambda x: check_enum(x, FeatureType, skip=[FeatureType.NONE]),  force=reset)
        self.IMG_DIMS               = self.PARAM_SERVER.add(self.NAMESPACE + "/img_dims",            img_dims,                  check_positive_two_int_tuple,                                   force=reset)

        self.DATABASE_PATH          = self.PARAM_SERVER.add(self.NAMESPACE + "/database_path",       database_path,             check_string,                                                   force=reset)
        self.REF_DATA_NAME          = self.PARAM_SERVER.add(self.NODESPACE + "/ref/data_name",       dataset_name,              check_string,                                                   force=reset)
        self.REF_IMG_PATH           = self.PARAM_SERVER.add(self.NODESPACE + "/ref/images_path",     ref_images_path,           check_string,                                                   force=reset)
        self.REF_ODOM_PATH          = self.PARAM_SERVER.add(self.NODESPACE + "/ref/odometry_path",   ref_odometry_path,         check_string,                                                   force=reset)

        self.TOL_MODE               = self.PARAM_SERVER.add(self.NAMESPACE + "/tolerance/mode",      enum_name(tolerance_mode), lambda x: check_enum(x, Tolerance_Mode),                        force=reset)
        self.TOL_THRES              = self.PARAM_SERVER.add(self.NAMESPACE + "/tolerance/threshold", tolerance_threshold,       check_positive_float,                                           force=reset)
        self.MATCH_METRIC           = self.PARAM_SERVER.add(self.NAMESPACE + "/match_metric",        match_metric,              check_string,                                                   force=reset)
        self.TIME_HIST_LEN          = self.PARAM_SERVER.add(self.NODESPACE + "/time_history_length", time_history_length,       check_positive_int,                                             force=reset)
        self.FRAME_ID               = self.PARAM_SERVER.add(self.NAMESPACE + "/frame_id",            frame_id,                  check_string,                                                   force=reset)
        #!# Enable/Disable Features (Label topic will always be generated):
        self.COMPRESS_IN            = self.PARAM_SERVER.add(self.NODESPACE + "/compress/in",         compress_in,               check_bool,                                                     force=reset)
        self.COMPRESS_OUT           = self.PARAM_SERVER.add(self.NODESPACE + "/compress/out",        compress_out,              check_bool,                                                     force=reset)
        self.DO_PLOTTING            = self.PARAM_SERVER.add(self.NODESPACE + "/method/plotting",     do_plotting,               check_bool,                                                     force=reset)
        self.MAKE_IMAGE             = self.PARAM_SERVER.add(self.NODESPACE + "/method/images",       do_image,                  check_bool,                                                     force=reset)
        self.GROUND_TRUTH           = self.PARAM_SERVER.add(self.NODESPACE + "/method/groundtruth",  do_groundtruth,            check_bool,                                                     force=reset)
        self.MAKE_LABEL             = self.PARAM_SERVER.add(self.NODESPACE + "/method/label",        do_label,                  check_bool,                                                     force=reset)

        self.DVC_WEIGHT             = self.PARAM_SERVER.add(self.NODESPACE + "/dvc_weight",          1,                         lambda x: check_bounded_float(x, 0, 1, 'both'),                 force=reset) # Hz

    def handle_GetPathPlan(self, req):
    # /vpr_nodes/path service
        ans = GenerateObjResponse()
        success = True

        try:
            if req.generate == True:
                self.generate_path()
            self.path_pub.publish(self.path_msg)
        except:
            success = False

        ans.success = success
        ans.topic = self.NAMESPACE + "/path"
        rospy.logdebug("Service requested [Gen=%s], Success=%s" % (str(req.generate), str(success)))
        return ans

    def generate_path(self, position_dict, times):
        self.path_msg = Path(header=Header(stamp=rospy.Time.now(), frame_id="map"))
        for (c, (x, y, w, t)) in enumerate(zip(position_dict['x'], position_dict['y'], position_dict['yaw'], times)):
            if not c % 3 == 0:
                continue
            new_pose = PoseStamped(header=Header(stamp=rospy.Time.from_sec(t), frame_id="map", seq=c))
            new_pose.pose.position = Point(x=x, y=y, z=0)
            new_pose.pose.orientation = q_from_yaw(w)
            self.path_msg.poses.append(new_pose)
            del new_pose

    def update_VPR(self):
        self.vpr_data_params       = dict(database_path=self.DATABASE_PATH.get(), name=self.REF_DATA_NAME.get(), img_path=self.REF_IMG_PATH.get(), \
                                           odom_path=self.REF_ODOM_PATH.get(), ft_type=self.FEAT_TYPE.get(), img_dims=self.IMG_DIMS.get())
        if not self.image_processor.swap(self.vpr_data_params):
            roslogger("VPR reference data swap failed. Previous set will be retained (ROS parameters won't be updated!)", LogType.ERROR, ros=True)
            self.vpr_swap_pending = True
        else:
            roslogger("VPR reference data swapped.", LogType.INFO, ros=True)
            self.vpr_swap_pending = False

    def param_callback(self, msg):
        if self.PARAM_SERVER.exists(msg.data):
            roslogger("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG, ros=True)
            self.PARAM_SERVER.update(msg.data)

            if msg.data in [self.DATABASE_PATH.name, self.REF_DATA_NAME.name, self.REF_IMG_PATH.name, \
                            self.REF_ODOM_PATH.name, self.FEAT_TYPE.name, self.IMG_DIMS.name]:
                roslogger("Change to VPR reference data parameters detected.", LogType.WARN, ros=True)
                self.update_VPR()
        else:
            roslogger("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG, ros=True)

    def main_cb(self, event):
    # Toggle flag to let main loop continue execution
    # This is currently bottlenecked by node performance and a rate limiter regardless, but I have kept it for future work

        self.main_ready = True

    def timer_plot_callback(self, event):
    # Toggle flag so that visualisation is performed at a lower rate than main loop processing (10Hz instead of >>10Hz)

        self.do_show            = True

    def label_callback(self, msg):
    # /vpr_nodes/label(/compressed) (aarapsi_robot_pack/(Compressed)ImageLabelStamped)
    # Store new label message and act as drop-in replacement for odom_callback + img_callback

        self.request            = msg

        if self.request.data.trueId < 0:
            self.GROUND_TRUTH   = False

        self.ego                = [msg.data.odom.x, msg.data.odom.y, msg.data.odom.z]

        if self.COMPRESS_IN.get():
            self.store_query    = self.bridge.compressed_imgmsg_to_cv2(self.request.queryImage, "passthrough")
        else:
            self.store_query    = self.bridge.imgmsg_to_cv2(self.request.queryImage, "passthrough")

        self.new_query          = True

    def data_callback(self, msg):
    # /data/img_odom (aarapsi_robot_pack/(Compressed)ImageOdom)

        self.odom_callback(msg.odom)
        self.img_callback(msg.image)

    def odom_callback(self, msg):
    # /odometry/filtered (nav_msgs/Odometry)
    # Store new robot position

        self.ego                = [round(msg.pose.pose.position.x, 3), round(msg.pose.pose.position.y, 3), round(msg.pose.pose.position.z, 3)]
        self.new_odom           = True

    def img_callback(self, msg):
    # /ros_indigosdk_occam/image0(/compressed) (sensor_msgs/(Compressed)Image)
    # Store newest image received

        if self.COMPRESS_IN.get():
            self.store_query_raw    = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        else:
            self.store_query_raw    = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.store_query            = self.store_query_raw#[10:-1,200:-50]
        self.new_query              = True
        
    def getMatchInd(self, ft_qry, metric='euclidean'):
    # top matching reference index for query

        dvc = cdist(self.ref_dict['img_feats'][enum_name(self.FEAT_TYPE.get())][self.IMG_FOLDER], np.matrix(ft_qry), metric) # metric: 'euclidean' or 'cosine'
        if self.ego_known and self.DVC_WEIGHT.get() < 1: # then perform biasing via distance:
            spd = cdist(np.transpose(np.matrix([self.ref_dict['odom']['position']['x'], self.ref_dict['odom']['position']['y']])), \
                np.matrix([self.vpr_ego[0], self.vpr_ego[1]]))
            spd_max_val = np.max(spd[:])
            dvc_max_val = np.max(dvc[:])
            spd_norm = spd/spd_max_val 
            dvc_norm = dvc/dvc_max_val
            spd_x_dvc = ((1-self.DVC_WEIGHT.get())*spd_norm**2 + (self.DVC_WEIGHT.get())*dvc_norm) # TODO: vary bias with velocity, weighted sum
            mInd = np.argmin(spd_x_dvc[:])
            return mInd, spd_x_dvc
        else:
            mInd = np.argmin(dvc[:])
            return mInd, dvc
    
    def getTrueInd(self):
    # Compare measured odometry to reference odometry and find best match
        squares = np.square(np.array(self.ref_dict['odom']['position']['x']) - self.ego[0]) + \
                            np.square(np.array(self.ref_dict['odom']['position']['y']) - self.ego[1])  # no point taking the sqrt; proportional
        trueInd = np.argmin(squares)

        return trueInd

    def publish_ros_info(self, cv2_img, fid, tInd, mInd, dvc, mPath, state):
    # Publish label and/or image feed

        self.rolling_mtrx_img = np.delete(self.rolling_mtrx_img, 0, 1) # delete first column (oldest query)
        self.rolling_mtrx_img = np.concatenate((self.rolling_mtrx_img, dvc), 1)
        
        mtrx_rgb = grey2dToColourMap(self.rolling_mtrx_img, dims=(500,500), colourmap=cv2.COLORMAP_JET)

        if self.COMPRESS_OUT.get():
            ros_image_to_pub = self.bridge.cv2_to_compressed_imgmsg(cv2_img, "jpeg") # jpeg (png slower)
            ros_matrix_to_pub = nmrc.bridge.cv2_to_compressed_imgmsg(mtrx_rgb, "jpeg") # jpeg (png slower)
        else:
            ros_image_to_pub = self.bridge.cv2_to_imgmsg(cv2_img, "bgr8")
            ros_matrix_to_pub = nmrc.bridge.cv2_to_imgmsg(mtrx_rgb, "bgr8")
        struct_to_pub = self.OUTPUTS['label']()
            
        ros_image_to_pub.header.stamp = rospy.Time.now()
        ros_image_to_pub.header.frame_id = fid

        struct_to_pub.queryImage        = ros_image_to_pub
        struct_to_pub.data.odom.x       = self.ego[0]
        struct_to_pub.data.odom.y       = self.ego[1]
        struct_to_pub.data.odom.z       = self.ego[2]
        struct_to_pub.data.dvc          = dvc
        struct_to_pub.data.matchId      = mInd
        struct_to_pub.data.trueId       = tInd
        struct_to_pub.data.state        = state
        struct_to_pub.data.compressed   = self.COMPRESS_OUT.get()
        struct_to_pub.data.matchPath    = mPath
        struct_to_pub.header.frame_id   = fid
        struct_to_pub.header.stamp      = rospy.Time.now()

        odom_to_pub = Odometry()
        odom_to_pub.pose.pose.position.x = self.ref_dict['odom']['position']['x'][mInd]
        odom_to_pub.pose.pose.position.y = self.ref_dict['odom']['position']['y'][mInd]
        odom_to_pub.pose.pose.orientation = q_from_yaw(self.ref_dict['odom']['position']['yaw'][mInd])
        odom_to_pub.header.stamp = rospy.Time.now()
        odom_to_pub.header.frame_id = fid

        self.rolling_mtrx.publish(ros_matrix_to_pub)
        self.odom_estimate_pub.publish(odom_to_pub)
        if self.MAKE_IMAGE.get():
            self.vpr_feed_pub.publish(ros_image_to_pub) # image feed publisher
        if self.MAKE_LABEL.get():
            self.vpr_label_pub.publish(struct_to_pub) # label publisher

        self.vpr_ego = [self.ref_dict['odom']['position']['x'][mInd], self.ref_dict['odom']['position']['y'][mInd], self.ref_dict['odom']['position']['yaw'][mInd]]
        self.ego_known = True

    def exit(self):
        rospy.loginfo("Quit received.")
        sys.exit()

def main_loop(nmrc):
# Main loop process
    nmrc.heartbeat.set_state(NodeState.MAIN)

    if nmrc.do_show and nmrc.DO_PLOTTING.get(): # set by timer callback and node input
        nmrc.fig.canvas.draw() # update all fig subplots
        plt.pause(0.001)
        nmrc.do_show = False # clear flag

    if not (nmrc.new_query and (nmrc.new_odom or not nmrc.MAKE_LABEL.get()) and nmrc.main_ready): # denest
        rospy.loginfo_throttle(60, "Waiting for a new query.") # print every 60 seconds
        return

    if (not nmrc.MAKE_LABEL.get()): # use label subscriber feed instead
        dvc             = np.transpose(np.matrix(nmrc.request.data.dvc))
        matchInd        = nmrc.request.data.matchId
        trueInd         = nmrc.request.data.trueId
    else:
        ft_qry          = nmrc.image_processor.getFeat(nmrc.store_query, nmrc.FEAT_TYPE.get(), use_tqdm=False)
        matchInd, dvc   = nmrc.getMatchInd(ft_qry, nmrc.MATCH_METRIC.get()) # Find match
        trueInd         = -1 #default; can't be negative.

    # Clear flags:
    nmrc.new_query      = False
    nmrc.new_odom       = False
    nmrc.main_ready     = False

    if nmrc.GROUND_TRUTH.get():
        trueInd = nmrc.getTrueInd() # find correct match based on shortest difference to measured odometry
    else:
        nmrc.ICON_DICT['size'] = -1

    ground_truth_string = ""
    tolState = 0
    if nmrc.GROUND_TRUTH.get(): # set by node inputs
        # Determine if we are within tolerance:
        nmrc.ICON_DICT['icon'] = nmrc.ICON_DICT['poor']
        if nmrc.TOL_MODE.get() == Tolerance_Mode.METRE_CROW_TRUE:
            tolError = np.sqrt(np.square(nmrc.ref_dict['odom']['position']['x'][trueInd] - nmrc.ego[0]) + \
                    np.square(nmrc.ref_dict['odom']['position']['y'][trueInd] - nmrc.ego[1])) 
            tolString = "MCT"
        elif nmrc.TOL_MODE.get() == Tolerance_Mode.METRE_CROW_MATCH:
            tolError = np.sqrt(np.square(nmrc.ref_dict['odom']['position']['x'][matchInd] - nmrc.ego[0]) + \
                    np.square(nmrc.ref_dict['odom']['position']['y'][matchInd] - nmrc.ego[1])) 
            tolString = "MCM"
        elif nmrc.TOL_MODE.get() == Tolerance_Mode.METRE_LINE:
            tolError = np.sqrt(np.square(nmrc.ref_dict['odom']['position']['x'][trueInd] - nmrc.ref_dict['odom']['position']['x'][matchInd]) + \
                    np.square(nmrc.ref_dict['odom']['position']['y'][trueInd] - nmrc.ref_dict['odom']['position']['y'][matchInd])) 
            tolString = "ML"
        elif nmrc.TOL_MODE.get() == Tolerance_Mode.FRAME:
            tolError = np.abs(matchInd - trueInd)
            tolString = "F"
        else:
            raise Exception("Error: Unknown tolerance mode.")

        if tolError < nmrc.TOL_THRES.get():
            nmrc.ICON_DICT['icon'] = nmrc.ICON_DICT['good']
            tolState = 2
        else:
            tolState = 1

        ground_truth_string = ", Error: %2.2f%s" % (tolError, tolString)

    if nmrc.MAKE_IMAGE.get(): # set by node input
        # make labelled match+query (processed) images and add icon for groundtruthing (if enabled):
        ft_ref = nmrc.ref_dict['img_feats'][enum_name(nmrc.FEAT_TYPE.get())][nmrc.IMG_FOLDER][matchInd]
        if nmrc.FEAT_TYPE.get() in [FeatureType.NETVLAD, FeatureType.HYBRIDNET]:
            reshape_dims = (64, 64)
        else:
            reshape_dims = nmrc.IMG_DIMS.get()
        cv2_image_to_pub = makeImage(ft_qry, ft_ref, reshape_dims, nmrc.ICON_DICT)
        
        # Measure timing for recalculating average rate:
        this_time = rospy.Time.now()
        time_diff = this_time - nmrc.last_time
        nmrc.last_time = this_time
        nmrc.time_history.append(time_diff.to_sec())
        num_time = len(nmrc.time_history)
        if num_time > nmrc.TIME_HIST_LEN.get():
            nmrc.time_history.pop(0)
        time_average = sum(nmrc.time_history) / num_time

        # add label with feed information to image:
        label_string = ("Index [%04d], %2.2f Hz" % (matchInd, 1/time_average)) + ground_truth_string
        cv2_img_lab = labelImage(cv2_image_to_pub, label_string, (20, cv2_image_to_pub.shape[0] - 40), (100, 255, 100))

        img_to_pub = cv2_img_lab
    else:
        img_to_pub = nmrc.store_query

    if nmrc.DO_PLOTTING.get(): # set by node input
        # Update odometry visualisation:
        updateMtrxFig(matchInd, trueInd, dvc, nmrc.ref_dict['odom'], nmrc.fig_mtrx_handles)
        updateDVecFig(matchInd, trueInd, dvc, nmrc.ref_dict['odom'], nmrc.fig_dvec_handles)
        updateOdomFig(matchInd, trueInd, dvc, nmrc.ref_dict['odom'], nmrc.fig_odom_handles)
    
    # Make ROS messages
    nmrc.publish_ros_info(img_to_pub, nmrc.FRAME_ID.get(), trueInd, matchInd, dvc, str(nmrc.ref_dict['image_paths']).replace('\'',''), tolState)

def do_args():
    parser = ap.ArgumentParser(prog="vpr_all_in_one", 
                                description="ROS implementation of QVPR's VPR Primer",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Positional Arguments:
    parser.add_argument('dataset-name',             type=check_string,                                              help='Specify name of dataset (for fast loading; matches are made on names starting with provided string).')
    parser.add_argument('database-path',            type=check_string,                                              help="Specify path to where compressed databases exist (for fast loading).")
    parser.add_argument('ref-imgs-path',            type=check_str_list,                                            help="Specify path to reference images (for slow loading).")
    parser.add_argument('ref-odom-path',            type=check_string,                                              help="Specify path to reference odometry (for slow loading).")
    parser.add_argument('data-topic-in',            type=check_string,                                              help="Specify input data topic (exclude /compressed). If --odom-topic-in is left unspecified, must be [Compressed]ImageOdom messages. Otherwise, [Compressed]Image messages.")

    # Optional Arguments:
    ft_options, ft_options_text             = enum_value_options(FeatureType, skip=FeatureType.NONE)
    tolmode_options, tolmode_options_text   = enum_value_options(Tolerance_Mode)

    parser.add_argument('--odom-topic-in',  '-o',   type=check_string,                  default=None,               help="Specify input odometry topic (default: %(default)s).")
    parser.add_argument('--compress-in',    '-Ci',  type=check_bool,                    default=False,              help='Enable image compression on input (default: %(default)s).')
    parser.add_argument('--compress-out',   '-Co',  type=check_bool,                    default=False,              help='Enable image compression on output (default: %(default)s).')
    parser.add_argument('--do-plotting',    '-P',   type=check_bool,                    default=False,              help='Enable matplotlib visualisations (default: %(default)s).')
    parser.add_argument('--make-images',    '-I',   type=check_bool,                    default=False,              help='Enable image topic generation (default: %(default)s).')
    parser.add_argument('--groundtruth',    '-G',   type=check_bool,                    default=False,              help='Enable groundtruth inclusion (default: %(default)s).')
    parser.add_argument('--make-labels',    '-L',   type=check_bool,                    default=True,               help='Enable label topic generation; false enables a subscriber instead (default: %(default)s).')
    parser.add_argument('--rate', '-r',             type=check_positive_float,          default=10.0,               help='Set node rate (default: %(default)s).')
    parser.add_argument('--time-hist',      '-l',   type=check_positive_int,            default=10,                 help='Set keep history size for logging true rate (default: %(default)s).')
    parser.add_argument('--img-dims',       '-i',   type=check_positive_two_int_tuple,  default=(64,64),            help='Set image dimensions (default: %(default)s).')
    parser.add_argument('--ft-type',        '-F',   type=int, choices=ft_options,       default=ft_options[0],      help='Choose feature type for extraction, types: %s (default: %s).' % (ft_options_text, '%(default)s'))
    parser.add_argument('--tol-mode',       '-t',   type=int, choices=tolmode_options,  default=tolmode_options[0], help='Choose tolerance mode for ground truth, types: %s (default: %s).' % (tolmode_options_text, '%(default)s'))
    parser.add_argument('--tol-thresh',     '-T',   type=check_positive_float,          default=1.0,                help='Set tolerance threshold for ground truth (default: %(default)s).')
    parser.add_argument('--icon-info',      '-p',   type=check_positive_two_int_tuple,  default=(50,20),            help='Set icon (size, distance) (default: %(default)s).')
    parser.add_argument('--node-name',      '-N',   type=check_string,                  default="vpr_all_in_one",   help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon',           '-a',   type=check_bool,                    default=True,               help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',      '-n',   type=check_string,                  default="/vpr_nodes",       help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--frame-id',       '-f',   type=check_string,                  default="odom",             help="Specify frame_id for messages (default: %(default)s).")
    parser.add_argument('--log-level',      '-V',   type=int, choices=[1,2,4,8,16],     default=2,                  help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset',          '-R',   type=check_bool,                    default=False,              help='Force reset of parameters to specified ones (default: %(default)s).')

    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        
        # Hand to class ...
        nmrc = mrc(args['database-path'], args['ref-imgs-path'], args['ref-odom-path'], args['data-topic-in'], args['dataset-name'], \
                    odom_topic=args['odom_topic_in'], compress_in=args['compress_in'], compress_out=args['compress_out'], do_plotting=args['do_plotting'], do_image=args['make_images'], \
                    do_groundtruth=args['groundtruth'], do_label=args['make_labels'], rate_num=args['rate'], ft_type=enum_get(args['ft_type'], FeatureType), \
                    img_dims=args['img_dims'], icon_settings=args['icon_info'], tolerance_threshold=args['tol_thresh'], \
                    tolerance_mode=enum_get(args['tol_mode'], Tolerance_Mode), match_metric='euclidean', namespace=args['namespace'], \
                    time_history_length=args['time_hist'], frame_id=args['frame_id'], \
                    node_name=args['node_name'], anon=args['anon'], log_level=args['log_level'], reset=args['reset']\
                )

        rospy.loginfo("Initialisation complete. Listening for queries...")    
        
        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            main_loop(nmrc)
            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass

        