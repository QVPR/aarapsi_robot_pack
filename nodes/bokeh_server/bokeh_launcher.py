#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import numpy as np
import rospkg
import argparse as ap
import os
import sys
import time

from aarapsi_robot_pack.msg import MonitorDetails, ImageDetails

from pyaarapsi.vpr_simple.vpr_dataset_tool       import VPRDatasetProcessor
from pyaarapsi.vpr_simple.vpr_helpers            import FeatureType
from pyaarapsi.vpr_simple.vpr_plots              import doDVecFigBokeh, doOdomFigBokeh, doFDVCFigBokeh, doCntrFigBokeh, doSVMMFigBokeh, doXYWVFigBokeh, \
                                                        updateDVecFigBokeh, updateOdomFigBokeh, updateFDVCFigBokeh, updateCntrFigBokeh, updateXYWVFigBokeh, updateSVMMFigBokeh

from pyaarapsi.core.argparse_tools  import check_positive_float, check_bool, check_positive_two_int_list, check_positive_int, check_valid_ip, check_enum, check_string
from pyaarapsi.core.helper_tools    import formatException, vis_dict, uint8_list_to_np_ndarray
from pyaarapsi.core.ros_tools       import roslogger, set_rospy_log_lvl, init_node, NodeState, LogType
from pyaarapsi.core.enum_tools      import enum_name
from pyaarapsi.core.ajax_tools      import AJAX_Connection, POST_Method_Types

from functools import partial
from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from bokeh.server.server import Server
from bokeh.themes import Theme

import logging

class mrc: # main ROS class
    def __init__(self, compress_in, rate_num, namespace, node_name, anon, log_level, reset, order_id=0):
        
        if not init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30, disable_signals=True):
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
        self.main_ready             = False
        self.parameters_ready       = True
        self.ajax_ready             = False
        self.ajax                   = AJAX_Connection(name='ROS')
        
        try:
            # Process reference data
            dataset_dict            = self.make_dataset_dict()
            self.ip                 = VPRDatasetProcessor(dataset_dict, try_gen=False, ros=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

    def init_rospy(self):
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())

        self.param_checker_sub      = rospy.Subscriber(self.namespace + "/params_update",        String,         self.param_callback,        queue_size=100)
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

    def state_callback(self, msg):
    # /vpr_nodes/state (aarapsi_robot_pack/MonitorDetails)
    # Send new SVM state to AJAX database
        if not self.ajax_ready:
            return
        data = {
            'time': msg.header.stamp.to_sec(),
            'dvc': msg.data.dvc,
            'gt_ego': [msg.data.gt_ego.x, msg.data.gt_ego.y, msg.data.gt_ego.w],
            'vpr_ego': [msg.data.vpr_ego.x, msg.data.vpr_ego.y, msg.data.vpr_ego.w],
            'match_ind': msg.data.matchId,
            'true_ind': msg.data.trueId,
            'gt_state': msg.data.gt_state,
            'gt_error': msg.data.gt_error,
            'zvalue': msg.mState,
            'prob': msg.prob,
            'pred': msg.mStateBin,
            'factors': msg.factors 
        }
        self.ajax.post('state', data=data, method_type=POST_Method_Types.SET)

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

        while (not rospy.is_shutdown()) and (not self.ajax_ready):
            self.print('Waiting for AJAX database to finish initialisation...', throttle=2)
            self.ajax_ready = self.ajax.check_if_ready()

        self.print("AJAX responsive.")
        self.print("Handling ROS data.")

        # loop forever until signal shutdown
        time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            new_time    = rospy.Time.now().to_sec()
            dt          = np.round(new_time - time,3)
            if dt == 0:
                dt = 0.0001
            self.print((dt, np.round(1/dt,3)), LogType.DEBUG)
            time        = new_time

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

def kill_screen(name):
    '''
    Kill screen

    Inputs:
    - name: str type, corresponding to full screen name as per 'screen -list'
    Returns:
    None
    '''
    os.system("if screen -list | grep -q '{sname}'; then screen -S '{sname}' -X quit; fi;".format(sname=name))

def kill_screens(names):
    '''
    Kill all screens in list of screen names

    Inputs:
    - names: list of str type, elements corresponding to full screen name as per 'screen -list'
    Returns:
    None
    '''
    for name in names:
        kill_screen(name)

def exec_screen(name, cmd):
    os.system("screen -dmS '{sname}' bash -c '{scmd}; exec bash'".format(sname=name, scmd=cmd))

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
    parser.add_argument('--order-id',         '-ID', type=int,                       default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')
    # Parse args...
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':

    time_str        = str(int(time.time()))
    show            = False
    bokeh_sname     = 'bokeh_' + time_str
    bokeh_scmd      = 'cd ../; bokeh serve --check-unused-sessions 100 --unused-session-lifetime 100 ros_test'
    if show: 
        bokeh_scmd += ' --show'

    ajax_sname      = 'ajax_' + time_str
    ajax_scmd       = './ajax_node.py'

    try:
        args = do_args()
        nmrc = mrc(compress_in=args['compress_in'], rate_num=args['rate'], namespace=args['namespace'], \
                        node_name=args['node_name'], anon=args['anon'], log_level=args['log_level'], reset=args['reset'], order_id=args['order_id'])
        nmrc.print("ROS Base ready.")
        
        kill_screens([bokeh_sname, ajax_sname])

        #exec_screen(bokeh_sname, bokeh_scmd)
        exec_screen(ajax_sname, ajax_scmd)
        
        nmrc.main()

    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)

    kill_screens([bokeh_sname, ajax_sname])

