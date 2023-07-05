#!/usr/bin/env python3

import rospy
import numpy as np
import rospkg
import argparse as ap
import os

from aarapsi_robot_pack.msg import MonitorDetails, ImageDetails
from std_msgs.msg import String

from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.vpr_plots         import doDVecFigBokeh, doOdomFigBokeh, doFDVCFigBokeh, doCntrFigBokeh, doSVMMFigBokeh, doXYWVFigBokeh, \
                                                    updateDVecFigBokeh, updateOdomFigBokeh, updateFDVCFigBokeh, updateCntrFigBokeh, updateXYWVFigBokeh, updateSVMMFigBokeh

from pyaarapsi.core.argparse_tools          import check_positive_int, check_valid_ip
from pyaarapsi.core.helper_tools            import formatException, vis_dict
from pyaarapsi.core.ros_tools               import LogType
from pyaarapsi.vpr_classes.base             import Base_ROS_Class, base_optional_args

from functools import partial
from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from bokeh.server.server import Server
from bokeh.themes import Theme

import logging

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()
        
        self.node_ready(kwargs['order_id'])

    def init_vars(self):
        super().init_vars()

        # flags to denest main loop:
        self.new_state              = False # new SVM state message received
        self.main_ready             = False
        self.svm_field_msg          = None
        self.update_contour         = True
        
        try:
            # Process reference data
            dataset_dict            = self.make_dataset_dict()
            self.ip                 = VPRDatasetProcessor(dataset_dict, try_gen=False, ros=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

    def init_rospy(self):
        super().init_rospy()
        
        self.svm_state_sub          = rospy.Subscriber(self.namespace + "/state",                MonitorDetails, self.state_callback,        queue_size=1)
        self.field_sub              = rospy.Subscriber(self.namespace + "/field",                ImageDetails,   self.field_callback,        queue_size=1)

    def update_VPR(self):
        dataset_dict = self.make_dataset_dict()
        if not self.ip.swap(dataset_dict, generate=False, allow_false=True):
            self.print("VPR reference data swap failed. Previous set will be retained (changed ROS parameter will revert)", LogType.WARN)
            return False
        else:
            self.print("VPR reference data swapped.", LogType.INFO)
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
            param.revert()
            self.print(formatException(), LogType.ERROR)

    def field_callback(self, msg: ImageDetails):
    # Store new SVM field
        self.svm_field_msg  = msg
        self.update_contour = self.svm_field_msg.data.update

    def state_callback(self, msg: MonitorDetails):
    # Store new label message and act as drop-in replacement for odom_callback + img_callback
        self.state              = msg
        self.new_state          = True

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
        super().exit()

class Doc_Frame:
    def __init__(self, nmrc: Main_ROS_Class, printer=print):
        # Prepare figures:
        self.num_points       = len(nmrc.ip.dataset['dataset']['px'])
        self.sim_mtrx_img     = np.zeros((self.num_points, self.num_points)) # Make similarity matrix figure
        
        self.fig_cntr_handles = doCntrFigBokeh() # Make contour figure
        self.fig_xywv_handles = doXYWVFigBokeh(self.num_points) # Make linear&angular metrics figure
        self.fig_dvec_handles = doDVecFigBokeh(self.num_points) # Make distance vector figure
        self.fig_fdvc_handles = doFDVCFigBokeh(self.num_points) # Make filtered dvc figure
        self.fig_odom_handles = doOdomFigBokeh(nmrc.ip.dataset['dataset']['px'], nmrc.ip.dataset['dataset']['py']) # Make odometry figure
        self.fig_svmm_handles = doSVMMFigBokeh(self.num_points) # Make SVM metrics figure
        

def main_loop(nmrc: Main_ROS_Class, doc_frame: Doc_Frame):
    # Main loop process

    if not (nmrc.new_state and nmrc.main_ready): # denest
        return

    # Clear flags:
    nmrc.new_state  = False

    dvc             = np.transpose(np.matrix(nmrc.state.data.dvc))
    matchInd        = nmrc.state.data.matchId
    trueInd         = nmrc.state.data.trueId

    # Update odometry visualisation:
    updateXYWVFigBokeh(doc_frame, matchInd, nmrc.ip.dataset['dataset']['px'], nmrc.ip.dataset['dataset']['py'], nmrc.ip.dataset['dataset']['pw'])
    updateDVecFigBokeh(doc_frame, matchInd, trueInd, dvc, nmrc.ip.dataset['dataset']['px'], nmrc.ip.dataset['dataset']['py'])
    updateFDVCFigBokeh(doc_frame, matchInd, trueInd, dvc, nmrc.ip.dataset['dataset']['px'], nmrc.ip.dataset['dataset']['py'])
    updateOdomFigBokeh(doc_frame, matchInd, trueInd, nmrc.ip.dataset['dataset']['px'], nmrc.ip.dataset['dataset']['py'])
    if (nmrc.svm_field_msg is None):
        return
    if updateCntrFigBokeh(doc_frame, nmrc.svm_field_msg, nmrc.state, nmrc.update_contour):
        nmrc.update_contour = False
    updateSVMMFigBokeh(doc_frame, nmrc.state)

def ros_spin(nmrc: Main_ROS_Class, doc_frame: Doc_Frame):
    try:
        nmrc.rate_obj.sleep()
        try:
            main_loop(nmrc, doc_frame)
        except Exception as e:
            if nmrc.parameters_ready:
                nmrc.print(vis_dict(nmrc.ip.dataset), LogType.DEBUG)
                raise Exception('Critical failure. ' + formatException()) from e
            else:
                nmrc.print('Main loop exception, attempting to handle; waiting for parameters to update. Details:\n' + formatException(), LogType.DEBUG, throttle=5)
                rospy.sleep(0.5)

            if rospy.is_shutdown():
                nmrc.exit()
    except:
        nmrc.print(formatException(), LogType.ERROR)
        nmrc.exit()

def main(doc, nmrc: Main_ROS_Class):
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
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='vpr_plotter')
    parser.add_argument('--port',    '-P',  type=check_positive_int, default=5006,      help='Set bokeh server port  (default: %(default)s).')
    parser.add_argument('--address', '-A',  type=check_valid_ip,     default='0.0.0.0', help='Set bokeh server address (default: %(default)s).')
    
    # Parse args...
    return vars(parser.parse_known_args()[0])

if __name__ == '__main__':
    os.environ['BOKEH_ALLOW_WS_ORIGIN'] = '0.0.0.0,131.181.33.60'
    logging.getLogger('bokeh').setLevel(logging.ERROR) # hide bokeh superfluous messages

    args = do_args()
    nmrc = Main_ROS_Class(**args)
    nmrc.print("[ROS Base] Ready.")
    server = Server({'/': lambda doc: main(doc, nmrc)}, num_procs=1, address=args['address'], port=args['port'])
    server.start()
    server.io_loop.start()

## Useful
# rospy.loginfo([server._tornado._applications['/']._session_contexts[key].destroyed for key in server._tornado._applications['/']._session_contexts.keys()])

    