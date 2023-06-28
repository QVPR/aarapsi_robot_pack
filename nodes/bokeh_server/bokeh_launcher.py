#!/usr/bin/env python3

import rospy
import rospkg
from std_msgs.msg import String
import numpy as np
import argparse as ap
import os
import time

from aarapsi_robot_pack.msg import MonitorDetails

from pyaarapsi.vpr_simple.vpr_dataset_tool       import VPRDatasetProcessor

from pyaarapsi.core.argparse_tools  import check_positive_float, check_bool, check_positive_int, check_valid_ip, check_string
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.ros_tools       import Base_ROS_Class, roslogger, set_rospy_log_lvl, NodeState, LogType
from pyaarapsi.core.ajax_tools      import AJAX_Connection, POST_Method_Types

class Main_ROS_Class(Base_ROS_Class): # main ROS class
    def __init__(self, rate_num, namespace, node_name, anon, log_level, reset, order_id=0):
        
        super().__init__(node_name, namespace, rate_num, anon, log_level, order_id=order_id, disable_signals=True)

        self.init_params(rate_num, log_level, reset)
        self.init_vars()
        self.init_rospy()

        # Last item as it sets a flag that enables main loop execution.
        self.main_ready = True
        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

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
        if self.params.exists(msg.data):
            if not self.params.update(msg.data):
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
    
    def main(self):
        self.set_state(NodeState.MAIN)

        while (not rospy.is_shutdown()) and (not self.ajax_ready):
            self.print('Waiting for AJAX database to finish initialisation...', throttle=2)
            self.ajax_ready = self.ajax.check_if_ready()

        self.print("AJAX responsive.")
        self.ajax.post('odom', data={k: self.ip.dataset['dataset'][k].tolist() for k in ['time', 'px', 'py', 'pw', 'vx', 'vy', 'vw']}, method_type=POST_Method_Types.SET)
        self.print("Handling ROS data.")

        # loop forever until signal shutdown
        time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            new_time    = rospy.Time.now().to_sec()
            dt          = np.max([np.round(new_time - time,3), 0.0001])
            self.print((dt, np.round(1/dt,3)), LogType.DEBUG)
            time        = new_time

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
    ajax_sname      = 'ajax_' + time_str

    try:
        args = do_args()

        file_path       = rospkg.RosPack().get_path('aarapsi_robot_pack')+'/nodes/bokeh_server'
        origin          = '--port %s --address %s' % (str(args['port']), args['address'])
        bokeh_scmd      = 'bokeh serve --allow-websocket-origin "*" ' + origin + ' --check-unused-sessions 100 --unused-session-lifetime 100 ' + file_path
        if show: 
            bokeh_scmd += ' --show'

        ajax_scmd       = file_path + '/ajax_node.py'

        nmrc = Main_ROS_Class(rate_num=args['rate'], namespace=args['namespace'], \
                        node_name=args['node_name'], anon=args['anon'], log_level=args['log_level'], reset=args['reset'], order_id=args['order_id'])
        nmrc.print("ROS Base ready.")
        
        kill_screens([bokeh_sname, ajax_sname])

        exec_screen(bokeh_sname, bokeh_scmd)
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

