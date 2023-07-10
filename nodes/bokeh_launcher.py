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

from pyaarapsi.core.argparse_tools  import check_positive_int, check_valid_ip
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.ros_tools       import roslogger, NodeState, LogType
from pyaarapsi.core.ajax_tools      import AJAX_Connection, POST_Method_Types
from pyaarapsi.core.os_tools        import exec_screen, kill_screens
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

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

    def param_helper(self, msg):
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
        # Main loop process
        self.set_state(NodeState.MAIN)

        while (not rospy.is_shutdown()) and (not self.ajax_ready):
            self.print('Waiting for AJAX database to finish initialisation...', throttle=2)
            self.ajax_ready = self.ajax.check_if_ready()

        self.print("AJAX responsive.")
        self.ajax.post('odom', data={k: self.ip.dataset['dataset'][k].tolist() for k in ['time', 'px', 'py', 'pw', 'vx', 'vy', 'vw']}, method_type=POST_Method_Types.SET)
        self.print("Handling ROS data.")

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
        self.rate_obj.sleep()

def do_args():
    parser = ap.ArgumentParser(prog="vpr_plotter", 
                               description="ROS implementation of QVPR's VPR Primer: Plotting Extension",
                               epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='vpr_plotter')
    parser.add_argument('--port',             '-P',  type=check_positive_int, default=5006,      help='Set bokeh server port  (default: %(default)s).')
    parser.add_argument('--address',          '-A',  type=check_valid_ip,     default='0.0.0.0', help='Set bokeh server address (default: %(default)s).')
    
    # Parse args...
    return vars(parser.parse_known_args()[0])

if __name__ == '__main__':

    time_str        = str(int(time.time()))
    show            = False
    bokeh_sname     = 'bokeh_' + time_str
    ajax_sname      = 'ajax_' + time_str

    try:
        args = do_args()

        file_path       = rospkg.RosPack().get_path('aarapsi_robot_pack')+'/servers/bokeh_server'
        origin          = '--port %s --address %s' % (str(args.pop('port')), str(args.pop('address')))
        bokeh_scmd      = 'bokeh serve --allow-websocket-origin "*" ' + origin + ' --check-unused-sessions 100 --unused-session-lifetime 100 ' + file_path
        if show: 
            bokeh_scmd += ' --show'

        ajax_scmd       = file_path + '/ajax_node.py'

        nmrc = Main_ROS_Class(disable_signals=True, **args)
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

