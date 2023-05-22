#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import numpy as np
import rospkg
import argparse as ap
import os
import sys

from aarapsi_robot_pack.srv import GenerateImageDetails, GenerateImageDetailsRequest
from aarapsi_robot_pack.msg import MonitorDetails, ImageDetails

from pyaarapsi.vpr_simple.vpr_dataset_tool       import VPRDatasetProcessor
from pyaarapsi.vpr_simple.vpr_helpers            import FeatureType
from pyaarapsi.vpr_simple.vpr_plots              import doDVecFigBokeh, doOdomFigBokeh, doFDVCFigBokeh, doCntrFigBokeh, doSVMMFigBokeh, doXYWVFigBokeh, \
                                                        updateDVecFigBokeh, updateOdomFigBokeh, updateFDVCFigBokeh, updateCntrFigBokeh, updateXYWVFigBokeh, updateSVMMFigBokeh

from pyaarapsi.core.argparse_tools  import check_positive_float, check_bool, check_positive_two_int_list, check_positive_int, check_valid_ip, check_enum, check_string
from pyaarapsi.core.helper_tools    import formatException, vis_dict
from pyaarapsi.core.ros_tools       import roslogger, set_rospy_log_lvl, init_node, NodeState, LogType
from pyaarapsi.core.enum_tools      import enum_name

from functools import partial
from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from bokeh.server.server import Server
from bokeh.themes import Theme

import logging

class mrc: # main ROS class
    def __init__(self, compress_in, rate_num, namespace, node_name, anon, log_level, reset, order_id=0):
        
        if not init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30):
            sys.exit()

        self.init_params(rate_num, log_level, compress_in, reset)
        self.init_vars()
        self.init_rospy()

        # Last item as it sets a flag that enables main loop execution.
        self.main_ready = True
        rospy.set_param(self.namespace + '/launch_step', order_id + 1)
        self.ROS_HOME.set_state(NodeState.MAIN)

    def init_params(self, rate_num, log_level, compress_in, reset):
        self.FEAT_TYPE       = self.ROS_HOME.params.add(self.namespace + "/feature_type",        None,                   lambda x: check_enum(x, FeatureType), force=False)
        self.IMG_DIMS        = self.ROS_HOME.params.add(self.namespace + "/img_dims",            None,                   check_positive_two_int_list,          force=False)
        self.NPZ_DBP         = self.ROS_HOME.params.add(self.namespace + "/npz_dbp",             None,                   check_string,                         force=False)
        self.BAG_DBP         = self.ROS_HOME.params.add(self.namespace + "/bag_dbp",             None,                   check_string,                         force=False)
        self.IMG_TOPIC       = self.ROS_HOME.params.add(self.namespace + "/img_topic",           None,                   check_string,                         force=False)
        self.ODOM_TOPIC      = self.ROS_HOME.params.add(self.namespace + "/odom_topic",          None,                   check_string,                         force=False)
        
        self.REF_BAG_NAME    = self.ROS_HOME.params.add(self.namespace + "/ref/bag_name",        None,                   check_string,                         force=False)
        self.REF_FILTERS     = self.ROS_HOME.params.add(self.namespace + "/ref/filters",         None,                   check_string,                         force=False)
        self.REF_SAMPLE_RATE = self.ROS_HOME.params.add(self.namespace + "/ref/sample_rate",     None,                   check_positive_float,                 force=False) # Hz
        
        self.COMPRESS_IN     = self.ROS_HOME.params.add(self.nodespace + "/compress/in",         compress_in,            check_bool,                           force=reset)
        self.RATE_NUM        = self.ROS_HOME.params.add(self.nodespace + "/rate",                rate_num,               check_positive_float,                 force=reset) # Hz
        self.LOG_LEVEL       = self.ROS_HOME.params.add(self.nodespace + "/log_level",           log_level,              check_positive_int,                   force=reset)
        
        self.REF_DATA_PARAMS = [self.NPZ_DBP, self.BAG_DBP, self.REF_BAG_NAME, self.REF_FILTERS, self.REF_SAMPLE_RATE, self.IMG_TOPIC, self.ODOM_TOPIC, self.FEAT_TYPE, self.IMG_DIMS]
        self.REF_DATA_NAMES  = [i.name for i in self.REF_DATA_PARAMS]

    def init_vars(self):
        # flags to denest main loop:
        self.new_state              = False # new SVM state message received
        self.main_ready             = False
        self.parameters_ready       = True
        self.svm_field_msg          = None
        self.update_contour         = False
        
        try:
            # Process reference data
            dataset_dict            = self.make_dataset_dict()
            self.image_processor    = VPRDatasetProcessor(dataset_dict, try_gen=False, ros=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

    def init_rospy(self):
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())

        self.param_checker_sub      = rospy.Subscriber(self.namespace + "/params_update",        String,         self.param_callback,        queue_size=100)
        self.svm_state_sub          = rospy.Subscriber(self.namespace + "/state",                MonitorDetails, self.state_callback,        queue_size=1)
        self.field_sub              = rospy.Subscriber(self.namespace + "/field",                ImageDetails,   self.field_callback,        queue_size=1)
        self.timer_check_if_dead    = rospy.Timer(rospy.Duration(secs=2),                                        self.timer_check_if_dead)

    def timer_check_if_dead(self, event):
        if rospy.is_shutdown():
            self.exit()

    def update_VPR(self):
        dataset_dict = self.make_dataset_dict()
        if not self.image_processor.swap(dataset_dict, generate=False, allow_false=True):
            self.print("VPR reference data swap failed. Previous set will be retained (changed ROS parameter will revert)", LogType.WARN)
            return False
        else:
            self.print("VPR reference data swapped.", LogType.INFO)
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

    def make_dataset_dict(self):
        return dict(bag_name=self.REF_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                    odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.REF_SAMPLE_RATE.get(), \
                    ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')

    def field_callback(self, msg):
    # /vpr_nodes/field (aarapsi_robot_pack/ImageDetails)
    # Store new SVM field
        self.svm_field_msg  = msg
        self.update_contour = self.svm_field_msg.data.update

    def state_callback(self, msg):
    # /vpr_nodes/state (aarapsi_robot_pack/MonitorDetails)
    # Store new label message and act as drop-in replacement for odom_callback + img_callback
        self.state              = msg
        self.new_state          = True

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
        try:
            server.io_loop.stop()
        except:
            print('Server IO Loop not accessible for forced shutdown, ignoring.')
        try:
            server.stop()
        except:
            print('Server not accessible for forced shutdown, ignoring.')
        print('Exit state reached.')
        sys.exit()

class Doc_Frame:
    def __init__(self, nmrc, printer=print):
        # Prepare figures:
        #iframe_start          = """<iframe src="http://131.181.33.60:8080/stream?topic="""
        #iframe_end_rect       = """&type=ros_compressed" width=2000 height=1000 style="border: 0; transform: scale(0.5); transform-origin: 0 0;"/>"""
        #iframe_end_even       = """&type=ros_compressed" width=510 height=510 style="border: 0; transform: scale(0.49); transform-origin: 0 0;"/>"""
        #self.fig_iframe_feed_ = Div(text=iframe_start + nmrc.namespace + "/image" + iframe_end_rect, width=500, height=250)
        #self.fig_iframe_mtrx_ = Div(text=iframe_start + nmrc.namespace + "/similiarity_matrix" + iframe_end_even, width=250, height=250)
        self.num_points       = len(nmrc.image_processor.dataset['dataset']['px'])
        self.sim_mtrx_img     = np.zeros((self.num_points, self.num_points)) # Make similarity matrix figure
        
        self.fig_cntr_handles = doCntrFigBokeh() # Make contour figure
        self.fig_xywv_handles = doXYWVFigBokeh(self.num_points) # Make linear&angular metrics figure
        self.fig_dvec_handles = doDVecFigBokeh(self.num_points) # Make distance vector figure
        self.fig_fdvc_handles = doFDVCFigBokeh(self.num_points) # Make filtered dvc figure
        self.fig_odom_handles = doOdomFigBokeh(nmrc.image_processor.dataset['dataset']['px'], nmrc.image_processor.dataset['dataset']['py']) # Make odometry figure
        self.fig_svmm_handles = doSVMMFigBokeh(self.num_points) # Make SVM metrics figure
        

def main_loop(nmrc, doc_frame):
    # Main loop process

    if not (nmrc.new_state and nmrc.main_ready): # denest
        return

    # Clear flags:
    nmrc.new_state  = False

    dvc             = np.transpose(np.matrix(nmrc.state.data.dvc))
    matchInd        = nmrc.state.data.matchId
    trueInd         = nmrc.state.data.trueId

    # Update odometry visualisation:
    updateXYWVFigBokeh(doc_frame, matchInd, nmrc.image_processor.dataset['dataset']['px'], nmrc.image_processor.dataset['dataset']['py'], nmrc.image_processor.dataset['dataset']['pw'])
    updateDVecFigBokeh(doc_frame, matchInd, trueInd, dvc, nmrc.image_processor.dataset['dataset']['px'], nmrc.image_processor.dataset['dataset']['py'])
    updateFDVCFigBokeh(doc_frame, matchInd, trueInd, dvc, nmrc.image_processor.dataset['dataset']['px'], nmrc.image_processor.dataset['dataset']['py'])
    updateOdomFigBokeh(doc_frame, matchInd, trueInd, nmrc.image_processor.dataset['dataset']['px'], nmrc.image_processor.dataset['dataset']['py'])
    if (nmrc.svm_field_msg is None):
        return
    updateCntrFigBokeh(doc_frame, nmrc.svm_field_msg, nmrc.state, nmrc.update_contour)
    updateSVMMFigBokeh(doc_frame, nmrc.state)
    nmrc.update_contour = False

def ros_spin(nmrc, doc_frame):
    try:
        nmrc.rate_obj.sleep()
        try:
            main_loop(nmrc, doc_frame)
        except Exception as e:
            if nmrc.parameters_ready:
                nmrc.print(vis_dict(nmrc.image_processor.dataset))
                raise Exception('Critical failure. ' + formatException()) from e
            else:
                nmrc.print('Main loop exception, attempting to handle; waiting for parameters to update. Details:\n' + formatException(), LogType.DEBUG, throttle=5)
                rospy.sleep(0.5)

            if rospy.is_shutdown():
                nmrc.exit()
    except:
        nmrc.print(formatException(), LogType.ERROR)
        nmrc.exit()

def main(doc, nmrc):
    try:
        doc_frame = Doc_Frame(nmrc)
        doc.add_root(row(   #column(doc_frame.fig_iframe_feed_, row(doc_frame.fig_iframe_mtrx_)), \
                            column(doc_frame.fig_dvec_handles['fig'], doc_frame.fig_odom_handles['fig'], doc_frame.fig_fdvc_handles['fig']), \
                            column(doc_frame.fig_cntr_handles['fig'], doc_frame.fig_svmm_handles['fig'], doc_frame.fig_xywv_handles['fig']) \
                        ))

        doc.theme = Theme(filename=rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/cfg/theme.yaml")

        doc.add_periodic_callback(partial(ros_spin, nmrc=nmrc, doc_frame=doc_frame), int(1000 * (1/nmrc.RATE_NUM.get())))
        nmrc.print("[Bokeh Server] Ready.")
    except Exception:
        nmrc.print(formatException(), LogType.ERROR)
        nmrc.exit()

def do_args():
    parser = ap.ArgumentParser(prog="vpr_plotter", 
                               description="ROS implementation of QVPR's VPR Primer: Plotting Extension",
                               epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--port',             '-P',  type=check_positive_int,        default=5006,             help='Set bokeh server port  (default: %(default)s).')
    parser.add_argument('--address',          '-A',  type=check_valid_ip,            default='0.0.0.0',        help='Set bokeh server address (default: %(default)s).')
    parser.add_argument('--compress-in',      '-Ci', type=check_bool,                default=False,            help='Enable image compression on input (default: %(default)s)')
    parser.add_argument('--rate',             '-r',  type=check_positive_float,      default=10.0,             help='Set node rate (default: %(default)s).')
    parser.add_argument('--node-name',        '-N',  type=check_string,              default="vpr_plotter",    help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon',             '-a',  type=check_bool,                default=True,             help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,              default="/vpr_nodes",     help="Specify namespace for topics (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16], default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--reset',            '-R',  type=check_bool,                default=False,            help='Force reset of parameters to specified ones (default: %(default)s)')
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')
    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    os.environ['BOKEH_ALLOW_WS_ORIGIN'] = '0.0.0.0,131.181.33.60'
    logging.getLogger('bokeh').setLevel(logging.CRITICAL) # hide bokeh superfluous messages

    args = do_args()
    nmrc = mrc(compress_in=args['compress_in'], rate_num=args['rate'], namespace=args['namespace'], \
                    node_name=args['node_name'], anon=args['anon'], log_level=args['log_level'], reset=args['reset'], order_id=args['order_id'])
    nmrc.print("[ROS Base] Ready.")
    server = Server({'/': lambda doc: main(doc, nmrc)}, num_procs=1, address=args['address'], port=args['port'])
    server.start()
    server.io_loop.start()

## Useful
# rospy.loginfo([server._tornado._applications['/']._session_contexts[key].destroyed for key in server._tornado._applications['/']._session_contexts.keys()])

    