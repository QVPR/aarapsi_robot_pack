#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import rospkg
import argparse as ap
import os
import sys
import copy

from aarapsi_robot_pack.msg import ImageLabelStamped, CompressedImageLabelStamped, \
    ImageDetails, CompressedImageDetails, MonitorDetails, CompressedMonitorDetails # Our custom msg structures
from aarapsi_robot_pack.srv import GenerateObj, GenerateObjRequest
from pyaarapsi.vpr_simple import VPRImageProcessor, FeatureType, \
                                            doDVecFigBokeh, doOdomFigBokeh, doFDVCFigBokeh, doCntrFigBokeh, doSVMMFigBokeh, doXYWVFigBokeh, \
                                            updateDVecFigBokeh, updateOdomFigBokeh, updateFDVCFigBokeh, updateCntrFigBokeh, updateXYWVFigBokeh, updateSVMMFigBokeh

from pyaarapsi.core.enum_tools import enum_value_options, enum_get, enum_name
from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_positive_two_int_tuple, check_positive_int, check_valid_ip, check_enum, check_string
from pyaarapsi.core.helper_tools import formatException, getArrayDetails
from pyaarapsi.core.ros_tools import roslogger, NodeState, LogType, ROS_Home

from functools import partial
from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from bokeh.server.server import Server
from bokeh.themes import Theme

import logging

class mrc: # main ROS class
    def __init__(self, database_path, dataset_name, ft_type=FeatureType.RAW, compress_in=True, rate_num=20.0, \
                 img_dims=(64,64), namespace='/vpr_nodes', node_name='vpr_all_in_one', \
                 anon=True, log_level=2, reset=False):
        
        self.NAMESPACE              = namespace
        self.NODENAME               = node_name
        self.NODESPACE              = self.NAMESPACE + "/" + self.NODENAME

        rospy.init_node(self.NODENAME, anonymous=anon, log_level=log_level)
        self.ROS_HOME               = ROS_Home(self.NODENAME, self.NAMESPACE, rate_num)
        self.print('Starting %s node.' % (node_name), LogType.INFO)

        self.init_params(rate_num, ft_type, img_dims, database_path, dataset_name, compress_in, reset)
        self.init_vars()
        self.init_rospy()

        # Prepare figures:
        iframe_start          = """<iframe src="http://131.181.33.60:8080/stream?topic="""
        iframe_end_rect       = """&type=ros_compressed" width=2000 height=1000 style="border: 0; transform: scale(0.5); transform-origin: 0 0;"/>"""
        iframe_end_even       = """&type=ros_compressed" width=510 height=510 style="border: 0; transform: scale(0.49); transform-origin: 0 0;"/>"""
        self.fig_iframe_feed_ = Div(text=iframe_start + self.NAMESPACE + "/image" + iframe_end_rect, width=500, height=250)
        self.fig_iframe_mtrx_ = Div(text=iframe_start + self.NAMESPACE + "/matrices/rolling" + iframe_end_even, width=250, height=250)
        self.rolling_mtrx_img = np.zeros((len(self.ref_dict['odom']['position']['x']), len(self.ref_dict['odom']['position']['x']))) # Make similarity matrix figure
        
        self.print('[Bokeh Server] Waiting for services ...')
        rospy.wait_for_service(self.NAMESPACE + '/GetSVMField')
        self.print('[Bokeh Server] Services ready.')
        
        try:
            self.fig_dvec_handles = doDVecFigBokeh(self, self.ref_dict['odom']) # Make distance vector figure
            self.fig_odom_handles = doOdomFigBokeh(self, self.ref_dict['odom']) # Make odometry figure
            self.fig_fdvc_handles = doFDVCFigBokeh(self, self.ref_dict['odom']) # Make filtered dvc figure
            self.fig_cntr_handles = doCntrFigBokeh(self, self.ref_dict['odom']) # Make contour figure
            self.fig_svmm_handles = doSVMMFigBokeh(self, self.ref_dict['odom']) # Make SVM metrics figure
            self.fig_xywv_handles = doXYWVFigBokeh(self, self.ref_dict['odom']) # Make linear&angular metrics figure
        except:
            self.print(formatException(), LogType.ERROR)

        # Last item as it sets a flag that enables main loop execution.
        self.main_ready = True

    def init_rospy(self):
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())

        self.param_checker_sub      = rospy.Subscriber(self.NAMESPACE + "/params_update", String, self.param_callback, queue_size=100)
        self.svm_state_sub          = rospy.Subscriber(self.NAMESPACE + "/monitor/state" + self.INPUTS['topic'], self.INPUTS['mon_dets'], self.state_callback, queue_size=1)
        self.svm_field_sub          = rospy.Subscriber(self.NAMESPACE + "/monitor/field" + self.INPUTS['topic'], self.INPUTS['img_dets'], self.field_callback, queue_size=1)
        self.srv_GetSVMField        = rospy.ServiceProxy(self.NAMESPACE + '/GetSVMField', GenerateObj)

    def init_vars(self):
        # flags to denest main loop:
        self.new_state              = False # new SVM state message received
        self.new_field              = False # new SVM field message received
        self.field_exists           = False # first SVM field message hasn't been received
        self.main_ready             = False
        self.srv_GetSVMField_once   = False

        self.bridge                 = CvBridge() # to convert sensor_msgs/(Compressed)Image to cv2.

        self._compress_on           = {'topic': "/compressed", 'image': CompressedImage, 'label': CompressedImageLabelStamped, 
                                       'img_dets': CompressedImageDetails, 'mon_dets': CompressedMonitorDetails}
        self._compress_off          = {'topic': "", 'image': Image, 'label': ImageLabelStamped, 
                                       'img_dets': ImageDetails, 'mon_dets': MonitorDetails}
        
        # Handle ROS details for input topics:
        if self.COMPRESS_IN.get():
            self.INPUTS             = self._compress_on
        else:
            self.INPUTS             = self._compress_off
        
        # Process reference data (only needs to be done once)
        self.image_processor        = VPRImageProcessor(ros=True, init_hybridnet=False, init_netvlad=False, cuda=False, dims=self.IMG_DIMS.get())
        if not self.image_processor.npzLoader(self.DATABASE_PATH.get(), self.REF_DATA_NAME.get()):
            self.exit()
        self.ref_dict               = copy.deepcopy(self.image_processor.SET_DICT) # store reference data
        self.image_processor.destroy() # destroy to save client memory

    def init_params(self, rate_num, ft_type, img_dims, database_path, dataset_name, compress_in, reset):
        self.RATE_NUM               = self.ROS_HOME.params.add(self.NODESPACE + "/rate", rate_num, check_positive_float, force=reset) # Hz
        
        self.FEAT_TYPE              = self.ROS_HOME.params.add(self.NAMESPACE + "/feature_type", enum_name(ft_type), lambda x: check_enum(x, FeatureType, skip=[FeatureType.NONE]), force=reset)
        self.IMG_DIMS               = self.ROS_HOME.params.add(self.NAMESPACE + "/img_dims", img_dims, check_positive_two_int_tuple, force=reset)

        self.DATABASE_PATH          = self.ROS_HOME.params.add(self.NAMESPACE + "/database_path", database_path, check_string, force=reset)
        self.REF_DATA_NAME          = self.ROS_HOME.params.add(self.NAMESPACE + "/ref/data_name", dataset_name, check_string, force=reset)

        #!# Enable/Disable Features:
        self.COMPRESS_IN            = self.ROS_HOME.params.add(self.NODESPACE + "/compress/in", compress_in, check_bool, force=reset)

    def param_callback(self, msg):
        try:
            if (msg.data in self.RATE_NUM.updates_possible) and not (msg.data in self.RATE_NUM.updates_queued):
                self.RATE_NUM.updates_queued.append(msg.data)
        except:
            self.print(formatException(), LogType.ERROR)
    
    def _timer_srv_GetSVMField(self, generate=False):
        try:
            if not self.main_ready:
                return
            requ = GenerateObjRequest()
            requ.generate = generate
            resp = self.srv_GetSVMField(requ)
            if resp.success == False:
                self.print('[timer_srv_GetSVMField] Service executed, success=False!', LogType.ERROR)
            self.srv_GetSVMField_once = True
        except:
            self.print(formatException(), LogType.ERROR)

    def timer_srv_GetSVMField(self):
        self._timer_srv_GetSVMField(generate=True)

    def field_callback(self, msg):
    # /vpr_nodes/monitor/field(/compressed) (aarapsi_robot_pack/(Compressed)ImageDetails)
    # Store new SVM field
        try:
            self.svm_field_msg  = msg
            self.new_field      = True
            self.field_exists   = True
        except:
            self.print(formatException(), LogType.ERROR)

    def state_callback(self, msg):
    # /vpr_nodes/monitor/state(/compressed) (aarapsi_robot_pack/(Compressed)MonitorDetails)
    # Store new label message and act as drop-in replacement for odom_callback + img_callback
        try:
            self.state              = msg
            self.new_state          = True
        except:
            self.print(formatException(), LogType.ERROR)

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

    def exit(self):
        global server
        server.io_loop.stop()
        server.stop()
        msg = 'Exit state reached.'
        if not rospy.is_shutdown():
            self.print(msg)
        else:
            print(msg)
        sys.exit()

def main_loop(nmrc):
# Main loop process

    if not nmrc.srv_GetSVMField_once:
        nmrc._timer_srv_GetSVMField(generate=True)

    if not (nmrc.new_state and nmrc.main_ready): # denest
        return

    # Clear flags:
    nmrc.new_state  = False

    dvc             = np.transpose(np.matrix(nmrc.state.data.dvc))
    matchInd        = nmrc.state.data.matchId
    trueInd         = nmrc.state.data.trueId

    # Update odometry visualisation:
    updateDVecFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])
    updateOdomFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])
    updateFDVCFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])
    updateXYWVFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])
    if nmrc.field_exists:
        updateCntrFigBokeh(nmrc, matchInd, trueInd, dvc, nmrc.ref_dict['odom'])
        updateSVMMFigBokeh(nmrc, nmrc.state, nmrc.ref_dict['odom'])

def ros_spin(nmrc):
    try:
        nmrc.rate_obj.sleep()
        main_loop(nmrc)

        if rospy.is_shutdown():
            nmrc.exit()
    except:
        nmrc.print(formatException(), LogType.ERROR)
        nmrc.exit()

def do_args():
    parser = ap.ArgumentParser(prog="vpr_all_in_one", 
                                description="ROS implementation of QVPR's VPR Primer",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    # Positional Arguments:
    parser.add_argument('dataset-name', help='Specify name of dataset (for fast loading; matches are made on names starting with provided string).')
    parser.add_argument('database-path', help="Specify path to where compressed databases exist (for fast loading).")

    # Optional Arguments:
    parser.add_argument('--compress-in', '-Ci', type=check_bool, default=False, help='Enable image compression on input (default: %(default)s)')
    parser.add_argument('--rate', '-r', type=check_positive_float, default=10.0, help='Set node rate (default: %(default)s).')
    parser.add_argument('--node-name', '-N', default="vpr_all_in_one", help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon', '-a', type=check_bool, default=True, help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace', '-n', default="/vpr_nodes", help="Specify namespace for topics (default: %(default)s).")
    parser.add_argument('--img-dims', '-i', type=check_positive_two_int_tuple, default=(64,64), help='Set image dimensions (default: %(default)s).')
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2, help="Specify ROS log level (default: %(default)s).")
    ft_options, ft_options_text = enum_value_options(FeatureType, skip=FeatureType.NONE)
    parser.add_argument('--ft-type', '-F', type=int, choices=ft_options, default=ft_options[0], \
                        help='Choose feature type for extraction, types: %s (default: %s).' % (ft_options_text, '%(default)s'))
    parser.add_argument('--reset', '-R', type=check_bool, default=False, help='Force reset of parameters to specified ones (default: %(default)s)')
    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])
    
def main(doc):
    try:
        # Parse args...
        args = do_args()

        # Hand to class ...
        nmrc = mrc(args['database-path'], args['dataset-name'], enum_get(args['ft_type'], FeatureType), compress_in=args['compress_in'], \
                   rate_num=args['rate'], namespace=args['namespace'], img_dims=args['img_dims'], \
                    node_name=args['node_name'], anon=args['anon'], log_level=args['log_level'], reset=args['reset'])

        doc.add_root(row(   column(nmrc.fig_iframe_feed_, row(nmrc.fig_iframe_mtrx_)), \
                            column(nmrc.fig_dvec_handles['fig'], nmrc.fig_odom_handles['fig'], nmrc.fig_fdvc_handles['fig']), \
                            column(nmrc.fig_cntr_handles['fig'], nmrc.fig_svmm_handles['fig'], nmrc.fig_xywv_handles['fig']) \
                        ))

        root = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/scripts/vpr_simple/"
        doc.theme = Theme(filename=root + "theme.yaml")

        doc.add_periodic_callback(partial(ros_spin, nmrc=nmrc), int(1000 * (1/nmrc.RATE_NUM.get())))
        #doc.add_periodic_callback(nmrc.timer_srv_GetSVMField, 1000)
        nmrc.ROS_HOME.set_state(NodeState.MAIN)
        nmrc.print("[Bokeh Server] Initialisation complete. Listening for queries...")
    except Exception:
        nmrc.print(formatException(), LogType.ERROR)
        nmrc.exit()

if __name__ == '__main__':
    os.environ['BOKEH_ALLOW_WS_ORIGIN'] = '0.0.0.0,131.181.33.60'
    logging.getLogger('bokeh').setLevel(logging.CRITICAL) # hide bokeh superfluous messages

    parser_server = ap.ArgumentParser()
    parser_server.add_argument('--port', '-P', type=check_positive_int, default=5006, help='Set bokeh server port  (default: %(default)s).')
    parser_server.add_argument('--address', '-A', type=check_valid_ip, default='0.0.0.0', help='Set bokeh server address (default: %(default)s).')
    server_vars = vars(parser_server.parse_known_args()[0])

    port=5006
    server = Server({'/': main}, num_procs=1, address=server_vars['address'], port=server_vars['port'])
    server.start()

    #print('Opening Bokeh application on http://localhost' + str(port))
    #server.show("/")
    server.io_loop.start()

## Useful
# rospy.loginfo([server._tornado._applications['/']._session_contexts[key].destroyed for key in server._tornado._applications['/']._session_contexts.keys()])

    