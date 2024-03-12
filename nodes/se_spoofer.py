#!/usr/bin/env python3

import rospy
import argparse as ap
import numpy as np
import sys
import copy
from enum import Enum

from pyaarapsi.core.ros_tools                   import NodeState, roslogger, LogType
from pyaarapsi.core.helper_tools                import formatException, normalize_angle, angle_wrap, m2m_dist
from pyaarapsi.vpr_classes.base                 import base_optional_args
from pyaarapsi.core.argparse_tools              import check_positive_float, check_string
from pyaarapsi.vpr_classes.dataset_loader_base  import Dataset_Loader
from pyaarapsi.vpr_simple.vpr_dataset_tool      import VPRDatasetProcessor, FeatureType
from pyaarapsi.pathing.basic                    import calc_path_stats, make_speed_array

from aarapsi_robot_pack.msg import Label, xyw

'''
State estimate spoofer

Corrupt an input odometry in-line with VPR track requirements.

'''

class Attack_Type(Enum):
    NONE            = 0
    Corrupt         = 1
    Zero            = 2
    Hold            = 3
    BiasForward     = 4
    BiasBackward    = 5
    Target          = 6

class Main_ROS_Class(Dataset_Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_params(self, rate_num, log_level, reset):
        super().init_params(rate_num, log_level, reset)

        self.PATH_SAMPLE_RATE       = self.params.add(self.namespace + "/path/sample_rate",         5.0,                    check_positive_float,                       force=False) # Hz
        self.PATH_FILTERS           = self.params.add(self.namespace + "/path/filters",             "{}",                   check_string,                               force=False)

    def init_vars(self):
        super().init_vars()

        self.attack_type    = Attack_Type.BiasBackward

        self.bias_growth_rate   = 0.3 # m/s; 5 mm per second
        self.current_bias       = 0
        self.start_time         = 0 # seconds

        self.state_msg      = Label()
        self.old_state_msg  = Label()
        self.new_state_msg  = False

        # Initialise dataset processor:
        self.ref_ip             = VPRDatasetProcessor(None, try_gen=False, ros=True, printer=self.print)
        self.path_ip            = VPRDatasetProcessor(None, try_gen=False, ros=True, printer=self.print) 

        self.ref_info                   = self.make_dataset_dict() # Get VPR pipeline's dataset dictionary
        self.ref_info['ft_types']       = [FeatureType.RAW.name] # Ensure same as path_follower
        self.path_info                  = copy.deepcopy(self.ref_info)
        self.path_info['sample_rate']   = self.PATH_SAMPLE_RATE.get()
        self.path_info['filters']       = self.PATH_FILTERS.get()

    def init_rospy(self):
        super().init_rospy()

        self.state_sub  = rospy.Subscriber( self.namespace + '/state', Label, self.state_cb, queue_size=1)
        self.state_pub  = self.add_pub( self.namespace + '/state_corrupted', Label)

    def state_cb(self, msg: Label):
        self.old_state_msg  = self.state_msg
        self.state_msg      = msg
        self.new_state_msg  = True

    def make_path(self):
        '''
        Generate:
        - Downsampled list of path points
        '''
        assert not self.path_ip.dataset is None
        # generate an n-row, 4 column array (x, y, yaw, speed) corresponding to each path node / reference image (same index)
        self.path_xyws  = np.transpose(np.stack([self.path_ip.dataset['dataset']['px'].flatten(), 
                                                 self.path_ip.dataset['dataset']['py'].flatten(),
                                                 self.path_ip.dataset['dataset']['pw'].flatten(),
                                                 make_speed_array(self.path_ip.dataset['dataset']['pw'].flatten())]))
        self.ref_xyws  = np.transpose(np.stack([self.ref_ip.dataset['dataset']['px'].flatten(), 
                                                 self.ref_ip.dataset['dataset']['py'].flatten(),
                                                 self.ref_ip.dataset['dataset']['pw'].flatten(),
                                                 make_speed_array(self.ref_ip.dataset['dataset']['pw'].flatten())]))
        
        # generate path / ref stats:
        self.path_sum, self.path_len     = calc_path_stats(self.path_xyws)
        self.ref_sum, self.ref_len       = calc_path_stats(self.ref_xyws)

    def ref2path(self, ref_index):
        if not hasattr(self, 'ref2path_matrix'):
            _ref2path_matrix = m2m_dist(self.ref_xyws[:,0:2],self.path_xyws[:,0:2])
            self.ref2path_matrix = np.argmin(_ref2path_matrix, axis=1)
        return self.ref2path_matrix[ref_index]

    def path2ref(self, path_index):
        if not hasattr(self, 'path2ref_matrix'):
            _path2ref_matrix = m2m_dist(self.path_xyws[:,0:2], self.ref_xyws[:,0:2])
            self.path2ref_matrix = np.argmin(_path2ref_matrix, axis=1)
        return self.path2ref_matrix[path_index]

    def main(self):
        # Main loop process
        self.set_state(NodeState.MAIN)

        self.print('Loading reference data...')
        self.load_dataset(self.ref_ip, self.ref_info) # Load reference data
        self.print('Loading path data...')
        self.load_dataset(self.path_ip, self.path_info) # Load path data

        # Generate path details:
        self.make_path()

        self.start_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            try:
                if not (self.new_state_msg):
                    self.print("Waiting for new data...", LogType.DEBUG, throttle=10)
                    rospy.sleep(0.005)
                    continue
                
                self.rate_obj.sleep()
                self.new_state_msg = False
                self.loop_contents()
            except rospy.exceptions.ROSInterruptException as e:
                pass
            except Exception as e:
                if self.parameters_ready:
                    raise Exception('Critical failure. ' + formatException()) from e
                else:
                    self.print('Main loop exception, attempting to handle; waiting for parameters to update. Details:\n' + formatException(), LogType.DEBUG, throttle=5)
                    rospy.sleep(0.5)

    def attack_numeric(self, numeric: float, perc: float, minval: float) -> float:
        random_sign = 1 if np.random.rand() < 0.5 else -1
        adj_minval  = (minval * (np.random.rand() * perc) * random_sign) + minval
        return numeric + adj_minval
    
    def attack_yaw(self, numeric: float, perc: float, minval: float) -> float:
        attacked_yaw = self.attack_numeric(numeric=numeric, perc=perc, minval=minval)
        return normalize_angle(attacked_yaw)
    
    def get_curr_path_ind(self, curr_xyw: xyw) -> int:
        return int(np.argmin(m2m_dist(self.path_xyws[:,0:2], np.array([curr_xyw.x, curr_xyw.y]))))

    def generate_bias(self, msg: xyw, _sign: int) -> xyw:
        time_now            = rospy.Time.now().to_sec()
        bias_growth_amount  = _sign * self.bias_growth_rate * (time_now - self.start_time) # amount to bias in direction
        path_ind            = self.get_curr_path_ind(curr_xyw=msg) # closest path index to true position
        biased_dist         = (self.path_sum[path_ind] + bias_growth_amount) %  self.path_len# along-track biased position
        biased_path_mid_ind = int(np.argmin( (self.path_sum - biased_dist) % self.path_len )) # along-track biased index
        # Find percentage position between indices:
        _K                  =  1 - ((biased_dist - self.path_sum[biased_path_mid_ind-1]) / (self.path_sum[biased_path_mid_ind+1] - self.path_sum[biased_path_mid_ind-1]))
        # Use percentage position to perform fractional average:
        _xy                 = self.path_xyws[biased_path_mid_ind-1,0:2] * _K + self.path_xyws[biased_path_mid_ind+1,0:2] * (1 - _K)
        _yaw                = angle_wrap(self.path_xyws[biased_path_mid_ind-1,2] + # take mean of two angles
                                         (1-_K) * angle_wrap(self.path_xyws[biased_path_mid_ind+1,2] -
                                                          self.path_xyws[biased_path_mid_ind-1,2], mode='RAD'), mode='RAD')
        return xyw(x=_xy[0], y=_xy[1], w=_yaw)
    
    def generate_attacked_xyw(self, msg: xyw, old_msg: xyw, curr_index: int) -> xyw:
        if self.attack_type == Attack_Type.NONE:
            pass
        elif self.attack_type == Attack_Type.Corrupt:
            msg.x = self.attack_numeric(numeric=msg.x, perc=0.2, minval=0.05)
            msg.y = self.attack_numeric(numeric=msg.y, perc=0.2, minval=0.05)
            msg.w = self.attack_yaw(numeric=msg.w, perc=0.2, minval=5*np.pi/180)
        elif self.attack_type == Attack_Type.Zero:
            msg = xyw(x=0, y=0, w=0)
        elif self.attack_type == Attack_Type.Hold:
            msg = old_msg
        elif self.attack_type == Attack_Type.BiasForward:
            msg = self.generate_bias(msg, 1)
        elif self.attack_type == Attack_Type.BiasBackward:
            msg = self.generate_bias(msg, -1)
        elif self.attack_type == Attack_Type.Target:
            pass
        return msg

    def loop_contents(self):

        self.state_msg.gt_ego = self.generate_attacked_xyw(msg=self.state_msg.gt_ego, 
                                                           old_msg=self.old_state_msg.gt_ego,
                                                           curr_index=self.state_msg.truth_index)
        
        self.state_pub.publish(self.state_msg)

def do_args():
    parser = ap.ArgumentParser(prog="se_spoofer.py", 
                            description="State Estimate Spoofer",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='se_spoofer', rate=20.0)
    
    # Parse args...
    return vars(parser.parse_known_args()[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = Main_ROS_Class(**args)
        nmrc.print("Initialisation complete.", LogType.INFO)
        nmrc.main()
        nmrc.print("Operation complete.", LogType.INFO, ros=False) # False as rosnode likely terminated
        sys.exit()
    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)