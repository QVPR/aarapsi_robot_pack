#!/usr/bin/env python3

import rospy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Header

from cv_bridge import CvBridge
from fastdist import fastdist
import cv2
import numpy as np
import rospkg
from matplotlib import pyplot as plt
import argparse as ap
import os
import sys

from aarapsi_robot_pack.msg import ImageLabelStamped, CompressedImageLabelStamped, ImageOdom, CompressedImageOdom # Our custom msg structures
from aarapsi_robot_pack.srv import GenerateObj, GenerateObjResponse

from pyaarapsi.vpr_simple.imageprocessor_helpers import Tolerance_Mode, FeatureType
from pyaarapsi.vpr_simple.vpr_feature_tool       import VPRImageProcessor
from pyaarapsi.vpr_simple.vpr_image_methods      import labelImage, makeImage, grey2dToColourMap
from pyaarapsi.vpr_simple.vpr_plots              import doMtrxFig, updateMtrxFig, doDVecFig, updateDVecFig, doOdomFig, updateOdomFig

from pyaarapsi.core.enum_tools      import enum_value_options, enum_get, enum_name
from pyaarapsi.core.argparse_tools  import check_bounded_float, check_positive_float, check_positive_two_int_tuple, check_positive_int, \
                                           check_bool, check_str_list, check_enum, check_string
from pyaarapsi.core.ros_tools       import q_from_yaw, roslogger, LogType, NodeState, ROS_Home
from pyaarapsi.core.helper_tools    import formatException, Timer

class mrc: # main ROS class
    def __init__(self, database_path, ref_images_path, ref_odometry_path, data_topic, dataset_name, odom_topic,\
                    compress_in, compress_out, do_plotting, do_image, do_groundtruth, do_label, rate_num, ft_type, \
                    img_dims, icon_settings, tolerance_threshold, tolerance_mode, match_metric, namespace, \
                    time_history_length, frame_id, node_name, anon, log_level, reset):
        
        self.NAMESPACE              = namespace
        self.NODENAME               = node_name
        self.NODESPACE              = self.NAMESPACE + "/" + self.NODENAME

        rospy.init_node(self.NODENAME, anonymous=anon, log_level=log_level)
        self.ROS_HOME               = ROS_Home(self.NODENAME, self.NAMESPACE, rate_num)
        self.print('Starting %s node.' % (node_name), LogType.INFO)

        self.init_params(rate_num, ft_type, img_dims, database_path, dataset_name, ref_images_path, \
                         ref_odometry_path, tolerance_mode, tolerance_threshold, match_metric, time_history_length, \
                         frame_id, compress_in, compress_out, do_plotting, do_image, do_groundtruth, do_label, reset)
        self.init_vars(icon_settings)
        self.init_rospy(data_topic, odom_topic)

    def init_rospy(self, data_topic, odom_topic):
        self.last_time              = rospy.Time.now()
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())

        self.param_checker_sub      = rospy.Subscriber(self.NAMESPACE + "/params_update", String, self.param_callback, queue_size=100)
        self.odom_estimate_pub      = self.ROS_HOME.add_pub(self.NAMESPACE + "/vpr_odom", Odometry, queue_size=1)

        self.send_path_plan         = rospy.Service(self.NAMESPACE + '/path', GenerateObj, self.handle_GetPathPlan)
        self.path_pub               = self.ROS_HOME.add_pub(self.NAMESPACE + '/path', Path, queue_size=1)

        if self.MAKE_IMAGE.get():
            self.vpr_feed_pub       = self.ROS_HOME.add_pub(self.NAMESPACE + "/image" + self.OUTPUTS['topic'], self.OUTPUTS['image'], queue_size=1)
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
            
            self.vpr_label_pub      = self.ROS_HOME.add_pub(self.NAMESPACE + "/label" + self.OUTPUTS['topic'], self.OUTPUTS['label'], queue_size=1)
            self.rolling_mtrx       = self.ROS_HOME.add_pub(self.NAMESPACE + "/matrices/rolling" + self.OUTPUTS['topic'], self.OUTPUTS['image'], queue_size=1)
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

        self.time_history           = []

        # Process reference data (only needs to be done once)
        self.image_processor        = VPRImageProcessor(ros=True, init_hybridnet=True, init_netvlad=True, cuda=True, dims=self.IMG_DIMS.get())
        try:
            self.ref_dict           = self.image_processor.npzDatabaseLoadSave(self.DATABASE_PATH.get(), self.REF_DATA_NAME.get(), \
                                                                                self.REF_IMG_PATH.get(), self.REF_ODOM_PATH.get(), \
                                                                                self.FEAT_TYPE.get(), self.IMG_DIMS.get(), do_save=False)
        except:
            self.exit()

        self.generate_path(self.ref_dict['odom']['position'], self.ref_dict['times'])

    def init_params(self, rate_num, ft_type, img_dims, database_path, dataset_name, ref_images_path, \
                    ref_odometry_path, tolerance_mode, tolerance_threshold, match_metric, time_history_length, \
                    frame_id, compress_in, compress_out, do_plotting, do_image, do_groundtruth, do_label, reset):
        
        self.FEAT_TYPE              = self.ROS_HOME.params.add(self.NAMESPACE + "/feature_type",        enum_name(ft_type),        lambda x: check_enum(x, FeatureType, skip=[FeatureType.NONE]),  force=reset)
        self.IMG_DIMS               = self.ROS_HOME.params.add(self.NAMESPACE + "/img_dims",            img_dims,                  check_positive_two_int_tuple,                                   force=reset)

        self.DATABASE_PATH          = self.ROS_HOME.params.add(self.NAMESPACE + "/database_path",       database_path,             check_string,                                                   force=reset)
        self.REF_DATA_NAME          = self.ROS_HOME.params.add(self.NAMESPACE + "/ref/data_name",       dataset_name,              check_string,                                                   force=reset)
        self.REF_IMG_PATH           = self.ROS_HOME.params.add(self.NAMESPACE + "/ref/images_path",     ref_images_path,           check_string,                                                   force=reset)
        self.REF_ODOM_PATH          = self.ROS_HOME.params.add(self.NAMESPACE + "/ref/odometry_path",   ref_odometry_path,         check_string,                                                   force=reset)
        self.IMG_FOLDER             = self.ROS_HOME.params.add(self.NAMESPACE + "/img_folder",         'forward',                  check_string,                                                   force=reset)

        self.TOL_MODE               = self.ROS_HOME.params.add(self.NAMESPACE + "/tolerance/mode",      enum_name(tolerance_mode), lambda x: check_enum(x, Tolerance_Mode),                        force=reset)
        self.TOL_THRES              = self.ROS_HOME.params.add(self.NAMESPACE + "/tolerance/threshold", tolerance_threshold,       check_positive_float,                                           force=reset)
        self.MATCH_METRIC           = self.ROS_HOME.params.add(self.NAMESPACE + "/match_metric",        match_metric,              check_string,                                                   force=reset)
        self.FRAME_ID               = self.ROS_HOME.params.add(self.NAMESPACE + "/frame_id",            frame_id,                  check_string,                                                   force=reset)

        self.RATE_NUM               = self.ROS_HOME.params.add(self.NODESPACE + "/rate",                rate_num,                  check_positive_float,                                           force=reset) # Hz
        self.TIME_HIST_LEN          = self.ROS_HOME.params.add(self.NODESPACE + "/time_history_length", time_history_length,       check_positive_int,                                             force=reset)
        self.COMPRESS_IN            = self.ROS_HOME.params.add(self.NODESPACE + "/compress/in",         compress_in,               check_bool,                                                     force=reset)
        self.COMPRESS_OUT           = self.ROS_HOME.params.add(self.NODESPACE + "/compress/out",        compress_out,              check_bool,                                                     force=reset)
        self.DO_PLOTTING            = self.ROS_HOME.params.add(self.NODESPACE + "/method/plotting",     do_plotting,               check_bool,                                                     force=reset)
        self.MAKE_IMAGE             = self.ROS_HOME.params.add(self.NODESPACE + "/method/images",       do_image,                  check_bool,                                                     force=reset)
        self.GROUND_TRUTH           = self.ROS_HOME.params.add(self.NODESPACE + "/method/groundtruth",  do_groundtruth,            check_bool,                                                     force=reset)
        self.MAKE_LABEL             = self.ROS_HOME.params.add(self.NODESPACE + "/method/label",        do_label,                  check_bool,                                                     force=reset)
        self.DVC_WEIGHT             = self.ROS_HOME.params.add(self.NODESPACE + "/dvc_weight",          1,                         lambda x: check_bounded_float(x, 0, 1, 'both'),                 force=reset) # Hz

        self.REF_DATA_PARAMS = [self.DATABASE_PATH, self.REF_DATA_NAME, self.REF_IMG_PATH, self.REF_ODOM_PATH, self.FEAT_TYPE, self.IMG_DIMS]
        self.REF_DATA_NAMES  = [i.name for i in self.REF_DATA_PARAMS]

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
        self.print("Service requested [Gen=%s], Success=%s" % (str(req.generate), str(success)), LogType.DEBUG)
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

    def update_VPR(self, param_to_change):
        self.vpr_data_params       = dict(database_path=self.DATABASE_PATH.get(), name=self.REF_DATA_NAME.get(), img_path=self.REF_IMG_PATH.get(), \
                                           odom_path=self.REF_ODOM_PATH.get(), ft_type=self.FEAT_TYPE.get(), img_dims=self.IMG_DIMS.get())
        if not self.image_processor.swap(self.vpr_data_params):
            self.print("VPR reference data swap failed. Previous set will be retained (changed ROS parameter will revert)", LogType.WARN)
            self.vpr_swap_pending = True
        else:
            self.print("VPR reference data swapped.", LogType.INFO)
            self.vpr_swap_pending = False

    def param_callback(self, msg):
        if self.ROS_HOME.params.exists(msg.data):
            self.print("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG)
            self.ROS_HOME.params.update(msg.data)
            return #TODO
            ref_data_comp   = [i == msg.data for i in self.REF_DATA_NAMES]
            try:
                param = np.array(self.REF_DATA_PARAMS)[ref_data_comp][0]
                self.print("Change to VPR reference data parameters detected.", LogType.WARN)
                self.update_VPR(param)
            except:
                pass # if error, not relevant to VPR ref dataset.
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)

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
        timer = Timer(rospy_on=True)
        timer.add()
        dvc = fastdist.matrix_to_matrix_distance(self.ref_dict['img_feats'][enum_name(self.FEAT_TYPE.get())][self.IMG_FOLDER.get()], np.matrix(ft_qry), fastdist.euclidean, "euclidean") # metric: 'euclidean' or 'cosine'
        timer.add()
        if self.ego_known and self.DVC_WEIGHT.get() < 1: # then perform biasing via distance:
            spd = fastdist.matrix_to_matrix_distance(np.transpose(np.matrix([self.ref_dict['odom']['position']['x'], self.ref_dict['odom']['position']['y']])), \
                np.matrix([self.vpr_ego[0], self.vpr_ego[1]]), fastdist.euclidean, "euclidean")
            spd_max_val = np.max(spd[:])
            dvc_max_val = np.max(dvc[:])
            spd_norm = spd/spd_max_val 
            dvc_norm = dvc/dvc_max_val
            spd_x_dvc = ((1-self.DVC_WEIGHT.get())*spd_norm**2 + (self.DVC_WEIGHT.get())*dvc_norm) # TODO: vary bias with velocity, weighted sum
            timer.add()
            mInd = np.argmin(spd_x_dvc[:])
            timer.add()
            timer.addb()
            timer.show()
            return mInd, spd_x_dvc
        else:
            timer.add()
            mInd = np.argmin(dvc[:])
            timer.add()
            timer.addb()
            timer.show()
            return mInd, dvc
    
    def getTrueInd(self):
    # Compare measured odometry to reference odometry and find best match
        squares = np.square(np.array(self.ref_dict['odom']['position']['x']) - self.ego[0]) + \
                            np.square(np.array(self.ref_dict['odom']['position']['y']) - self.ego[1])  # no point taking the sqrt; proportional
        trueInd = np.argmin(squares)

        return trueInd

    def publish_ros_info(self, cv2_img, fid, tInd, mInd, dvc, mPath, state):
    # Publish label and/or image feed
        timer = Timer()

        timer.add()
        self.rolling_mtrx_img = np.delete(self.rolling_mtrx_img, 0, 1) # delete first column (oldest query)
        self.rolling_mtrx_img = np.concatenate((self.rolling_mtrx_img, dvc), 1)
        timer.add()
        mtrx_rgb = grey2dToColourMap(self.rolling_mtrx_img, dims=(500,500), colourmap=cv2.COLORMAP_JET)
        timer.add()
        if self.COMPRESS_OUT.get():
            ros_image_to_pub = self.bridge.cv2_to_compressed_imgmsg(cv2_img, "jpeg") # jpeg (png slower)
            ros_matrix_to_pub = self.bridge.cv2_to_compressed_imgmsg(mtrx_rgb, "jpeg") # jpeg (png slower)
        else:
            ros_image_to_pub = self.bridge.cv2_to_imgmsg(cv2_img, "bgr8")
            ros_matrix_to_pub = self.bridge.cv2_to_imgmsg(mtrx_rgb, "bgr8")
        struct_to_pub = self.OUTPUTS['label']()
        timer.add()
        ros_image_to_pub.header.stamp = rospy.Time.now()
        ros_image_to_pub.header.frame_id = fid
        timer.add()
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
        timer.add()
        odom_to_pub = Odometry()
        odom_to_pub.pose.pose.position.x = self.ref_dict['odom']['position']['x'][mInd]
        odom_to_pub.pose.pose.position.y = self.ref_dict['odom']['position']['y'][mInd]
        odom_to_pub.pose.pose.orientation = q_from_yaw(self.ref_dict['odom']['position']['yaw'][mInd])
        odom_to_pub.header.stamp = rospy.Time.now()
        odom_to_pub.header.frame_id = fid
        timer.add()
        self.rolling_mtrx.publish(ros_matrix_to_pub)
        self.odom_estimate_pub.publish(odom_to_pub)
        if self.MAKE_IMAGE.get():
            self.vpr_feed_pub.publish(ros_image_to_pub) # image feed publisher
        if self.MAKE_LABEL.get():
            self.vpr_label_pub.publish(struct_to_pub) # label publisher
        timer.add()
        self.vpr_ego = [self.ref_dict['odom']['position']['x'][mInd], self.ref_dict['odom']['position']['y'][mInd], self.ref_dict['odom']['position']['yaw'][mInd]]
        self.ego_known = True
        timer.add()
        timer.addb()
        #timer.show()

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
        timer = Timer(rospy_on=True)
        timer.add()
        if self.do_show and self.DO_PLOTTING.get(): # set by timer callback and node input
            self.fig.canvas.draw() # update all fig subplots
            plt.pause(0.001)
            self.do_show = False # clear flag
        if not (self.new_query and (self.new_odom or not self.MAKE_LABEL.get()) and self.main_ready): # denest
            self.print("Waiting for a new query.", throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return
        self.rate_obj.sleep()

        if (not self.MAKE_LABEL.get()): # use label subscriber feed instead
            dvc             = np.transpose(np.matrix(self.request.data.dvc))
            matchInd        = self.request.data.matchId
            trueInd         = self.request.data.trueId
        else:
            ft_qry          = self.image_processor.getFeat(self.store_query, self.FEAT_TYPE.get(), use_tqdm=False)
            timer.add()
            matchInd, dvc   = self.getMatchInd(ft_qry, self.MATCH_METRIC.get()) # Find match
            timer.add()
            trueInd         = -1 #default; can't be negative.
        # Clear flags:
        self.new_query      = False
        self.new_odom       = False
        self.main_ready     = False

        if self.GROUND_TRUTH.get():
            trueInd = self.getTrueInd() # find correct match based on shortest difference to measured odometry
        else:
            self.ICON_DICT['size'] = -1
        timer.add()
        ground_truth_string = ""
        tolState = 0
        if self.GROUND_TRUTH.get(): # set by node inputs
            # Determine if we are within tolerance:
            self.ICON_DICT['icon'] = self.ICON_DICT['poor']
            if self.TOL_MODE.get() == Tolerance_Mode.METRE_CROW_TRUE:
                tolError = np.sqrt(np.square(self.ref_dict['odom']['position']['x'][trueInd] - self.ego[0]) + \
                        np.square(self.ref_dict['odom']['position']['y'][trueInd] - self.ego[1])) 
                tolString = "MCT"
            elif self.TOL_MODE.get() == Tolerance_Mode.METRE_CROW_MATCH:
                tolError = np.sqrt(np.square(self.ref_dict['odom']['position']['x'][matchInd] - self.ego[0]) + \
                        np.square(self.ref_dict['odom']['position']['y'][matchInd] - self.ego[1])) 
                tolString = "MCM"
            elif self.TOL_MODE.get() == Tolerance_Mode.METRE_LINE:
                tolError = np.sqrt(np.square(self.ref_dict['odom']['position']['x'][trueInd] - self.ref_dict['odom']['position']['x'][matchInd]) + \
                        np.square(self.ref_dict['odom']['position']['y'][trueInd] - self.ref_dict['odom']['position']['y'][matchInd])) 
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
                tolState = 1

            ground_truth_string = ", Error: %2.2f%s" % (tolError, tolString)
        timer.add()
        if self.MAKE_IMAGE.get(): # set by node input
            # make labelled match+query (processed) images and add icon for groundtruthing (if enabled):
            ft_ref = self.ref_dict['img_feats'][enum_name(self.FEAT_TYPE.get())][self.IMG_FOLDER.get()][matchInd]
            if self.FEAT_TYPE.get() in [FeatureType.NETVLAD, FeatureType.HYBRIDNET]:
                reshape_dims = (64, 64)
            else:
                reshape_dims = self.IMG_DIMS.get()
            cv2_image_to_pub = makeImage(ft_qry, ft_ref, reshape_dims, self.ICON_DICT)
            
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
            label_string = ("Index [%04d], %2.2f Hz" % (matchInd, 1/time_average)) + ground_truth_string
            cv2_img_lab = labelImage(cv2_image_to_pub, label_string, (20, cv2_image_to_pub.shape[0] - 40), (100, 255, 100))

            img_to_pub = cv2_img_lab
        else:
            img_to_pub = self.store_query
        timer.add()
        if self.DO_PLOTTING.get(): # set by node input
            # Update odometry visualisation:
            updateMtrxFig(matchInd, trueInd, dvc, self.ref_dict['odom'], self.fig_mtrx_handles)
            updateDVecFig(matchInd, trueInd, dvc, self.ref_dict['odom'], self.fig_dvec_handles)
            updateOdomFig(matchInd, trueInd, dvc, self.ref_dict['odom'], self.fig_odom_handles)
        timer.add()
        # Make ROS messages
        self.publish_ros_info(img_to_pub, self.FRAME_ID.get(), trueInd, matchInd, dvc, str(self.ref_dict['image_paths']).replace('\'',''), tolState)
        timer.add()
        timer.addb()
        #timer.show()

    def exit(self):
        self.print("Quit received.", LogType.INFO)
        sys.exit()

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

        