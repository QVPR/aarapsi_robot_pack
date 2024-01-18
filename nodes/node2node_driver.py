#!/usr/bin/env python3

import rospy
import copy
import numpy as np
import argparse as ap
import sys
from enum import Enum

import scipy.stats as st

from pyaarapsi.core.ros_tools                   import NodeState, roslogger, LogType
from pyaarapsi.core.helper_tools                import formatException, p2p_dist_2d, plt_pause, m2m_dist
from pyaarapsi.core.vars                        import C_I_GREEN, C_I_YELLOW, C_I_RED, C_I_BLUE, C_RESET
from pyaarapsi.core.argparse_tools              import check_positive_float, check_string
from pyaarapsi.vpr_classes.base                 import Base_ROS_Class, base_optional_args
from pyaarapsi.vpr_classes.dataset_loader_base  import Dataset_Loader
from pyaarapsi.vpr_simple.vpr_dataset_tool      import VPRDatasetProcessor, FeatureType
from pyaarapsi.pathing.basic                    import calc_path_stats, make_speed_array
from aarapsi_robot_pack.msg                     import Label, ControllerStateInfo, SpeedCommand

import matplotlib.pyplot as plt

'''
Node2Node Driver

Control path_follower to drive between nodes.

'''

class Vehicle_Mode(Enum):
    REVERSE = -1
    STOPPED = 0
    FORWARD = 1

class Stage(Enum):
    INIT    = -1
    S0      = 0
    S1      = 1
    S2      = 2
    S3      = 3
    S4      = 4
    S5      = 5
    END     = 10

def shortest_index_distance(_start, _end, _length):
    _options = [((_end-_start)-_length), ((_end-_start)-_length)%_length]
    return _options[np.argmin(np.abs(_options))]

def find_percentile_conf(length: int, percentile: float, confidence: float):
    assert (length > 0) and isinstance(length, int) 
    assert (percentile) <= 1 and (percentile >= 0)
    assert (confidence) <= 1 and (confidence >= 0)
    # set up initial values:
    _mid    = np.max([0, np.min([np.ceil(percentile * length).astype(int), length - 1])])
    _conf   = 0.0
    _lower, _upper = _mid, _mid
    # search:
    while _conf < confidence:
        _lower = np.max([0, _lower - 1])
        _upper = np.min([length - 1, _upper + 1])
        _conf = np.sum(st.binom.pmf(np.arange(_lower,_upper), length, percentile))
        # check if we've exceeded our range:
        if _lower == 0 and _upper == length - 1:
            break
    return ((_lower, _upper), _conf, _mid-_lower==_upper-_mid)

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

        self.ready      = False

        self.label      = Label()
        self.info       = ControllerStateInfo()
        self.command    = SpeedCommand()
        self.new_label  = False
        self.new_info   = False

        self.mode       = Vehicle_Mode.STOPPED
        self.stage      = Stage.INIT

        self.odometer   = 0.0
        self.recent_odo = 0.0
        self.stage_odo  = 0.0
        self.last_pose  = [0.0,0.0,0.0]
        self.mdist_hist = []
        self.mdist_len  = 0

        self.___means   = []
        self.___dists   = []
        self.___cibbs   = []
        self.___dynth   = []

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

        self.state_sub   = rospy.Subscriber( self.namespace + '/state',                 Label,               self.state_cb, queue_size=1)
        self.info_sub    = rospy.Subscriber( self.namespace + '/path_follower/info',    ControllerStateInfo, self.info_cb,  queue_size=1)
        self.command_pub = self.add_pub(     self.namespace + '/path_follower/command', SpeedCommand,                       queue_size=1)

    def state_cb(self, msg: Label):
        '''
        Callback to handle new labels from the VPR pipeline
        '''
        if not self.ready:
            return
        
        self.label     = msg
        self.new_label = True

    def info_cb(self, msg: ControllerStateInfo):
        '''
        Callback to handle new ControllerStateInfo messages from the VPR pipeline
        '''
        if not self.ready:
            return
        
        self.info      = msg
        self.new_info  = True

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

    def main(self):
        # Main loop process
        self.set_state(NodeState.MAIN)

        self.print('Loading reference data...')
        self.load_dataset(self.ref_ip, self.ref_info) # Load reference data
        self.print('Loading path data...')
        self.load_dataset(self.path_ip, self.path_info) # Load path data

        # Generate path details:
        self.make_path()

        self.ready = True

        self.fig,self.ax = plt.subplots(1,1)
        plt.show(block=False)

        while not rospy.is_shutdown():
            try:
                # Denest main loop; wait for new messages:
                if not (self.new_label and self.new_info):
                    self.print("Waiting for new data...", LogType.DEBUG, throttle=10)
                    rospy.sleep(0.005)
                    continue
                
                self.rate_obj.sleep()
                self.new_label = False
                self.new_info  = False
                if self.stage == Stage.INIT:
                    self.init_robot()
                else:
                    self.loop_contents()

            except rospy.exceptions.ROSInterruptException as e:
                pass
            except Exception as e:
                if self.parameters_ready:
                    raise Exception('Critical failure. ' + formatException()) from e
                else:
                    self.print('Main loop exception, attempting to handle; waiting for parameters to update. Details:\n' + formatException(), LogType.DEBUG, throttle=5)
                    rospy.sleep(0.5)

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

    def init_robot(self):

        if not hasattr(self, '_first_robot_print'):
            self.print(C_I_BLUE + 'Initialising: Heading to Start Position.' + C_RESET, LogType.DEBUG)
            self._first_robot_print = True

        self.command.enabled    = True
        self.command.mode       = self.command.OVERRIDE

        _end        = self.ref2path(0) + 5
        _start      = self.info.group.current_ind
        _length     = self.path_xyws.shape[0]
        _options    = [((_end-_start)%_length), ((_end-_start)%_length)-_length]
        _distance   = _options[np.argmin(np.abs(_options))]

        self.command.speed      = [0.5 if np.abs(_distance) < (0.9 * self.path_xyws.shape[0]) else 0.2]
        self.command.reverse    = _distance < 0

        if np.abs(_distance) <= 1:
            self.stage = Stage.S0
            self.mode = Vehicle_Mode.STOPPED
            self.command.reverse = False
            self.command.speed = [0.0]
            self.print(C_I_BLUE + 'Ready: Commencing Mission Operation.' + C_RESET, LogType.DEBUG)

        self.command_pub.publish(self.command)


    def loop_contents(self):
        
        if self.mode == Vehicle_Mode.STOPPED:

            self.command.enabled = True
            self.command.mode    = self.command.OVERRIDE
            self.command.speed   = [0.2]

            self.mode   = Vehicle_Mode.FORWARD
            self.stage  = Stage.S0

            self.last_pose = [getattr(self.label.robot_ego, i) for i in 'xyw']

            return

        curr_pose           = [getattr(self.label.robot_ego, i) for i in 'xyw']
        odo_step            = p2p_dist_2d(self.last_pose, curr_pose)
        self.recent_odo    += odo_step
        self.stage_odo     += odo_step
        self.odometer      += odo_step * np.sign(self.mode.value)
        self.last_pose      = curr_pose

        curr_dist           = self.label.distance_vector[self.label.match_index]
    
        # if self.recent_odo > 0.01:
        #     self.recent_odo = 0.0

        self.mdist_hist.append(curr_dist)
        self.mdist_len     += 1
        if self.mdist_len > 30:
            self.mdist_hist.pop(0)
            self.mdist_len -= 1
        
        if not hasattr(self, 'q'):
            self.q = 0


        if self.mdist_len > 10:
            #ci_lower, ci_upper = st.t.interval(alpha=0.975, df=self.mdist_len-1, loc=np.mean(self.mdist_hist), scale=st.sem(self.mdist_hist))
            #print('%0.2f,%0.2f [%0.2f]' % (ci_lower, ci_upper, curr_dist))
            #print(np.sort(self.mdist_hist)[lower], curr_dist)
            _omean, _ostd = np.mean(self.mdist_hist), np.std(self.mdist_hist)

            _samples = np.random.normal(loc=0.0, scale=_ostd, size=np.min([self.mdist_len, 30]))
            _zsamples = _samples - np.mean(_samples) # remove mean,
            _asamples = _zsamples * (_ostd / np.std(_zsamples)) # fix std,
            _fsamples = _asamples + _omean # set mean,
            #print([np.mean(_fsamples), np.std(_fsamples)], [_omean, _ostd], self.mdist_len) # check!
            self.mdist_hist = list(_fsamples)

            (lower,upper), _conf, _sym = find_percentile_conf(self.mdist_len, 0.25, 0.995)

            self.___means.append(np.mean(_fsamples))
            self.___dists.append(curr_dist)
            self.___cibbs.append(np.sort(self.mdist_hist)[lower])
            self.___dynth.append(np.mean(self.___cibbs[max([-10,-len(self.___cibbs)]):]))

            if len(self.___means) > 200:
                self.___means.pop(0)
                self.___dists.pop(0)
                self.___cibbs.pop(0)
                self.___dynth.pop(0)

            self.ax.clear()
            self.ax.plot(self.___means, 'r.')
            self.ax.plot(self.___dists, 'g+')
            self.ax.plot(self.___cibbs, 'b+', alpha=0.5)
            self.ax.plot(self.___dynth, 'b.')
            self.q += 1
            plt_pause(0.001, self.fig)

        if True:
            if not hasattr(self, 'bmd'):
                self.bmd        = 100000
            self.bmd            = np.min([curr_dist, self.bmd])

            if self.stage == Stage.S0:
                if self.mdist_len > 11:

                    self.stage = Stage.S1
                    self.stage_odo = 0.0
                    self.command.speed = [0.2]
                    self.best_s1 = [None, 100000, None]
                    self.s1_flag = [False, 0]
                    self.better_flag = False

                    self.print('> Commencing Stage 1: Looking for an OK point ...', LogType.DEBUG)
                
            elif self.stage == Stage.S1:
                if (curr_dist < self.best_s1[1]):
                    self.best_s1 = [self.odometer, curr_dist, self.label.match_index]
                    if curr_dist < self.___cibbs[-1]:
                        if not self.s1_flag[0]:
                            self.print('\tFound OK point: Overshooting by 0.5m ...', LogType.DEBUG)
                            self.s1_flag = [True, self.odometer]
                        else:
                            if not self.better_flag:
                                self.print('\tFound better point: Overshooting by 0.5m ...', LogType.DEBUG)
                                self.better_flag = True
                            self.s1_flag[1] = self.odometer

                if (self.s1_flag[0]) and (np.abs(self.odometer - self.s1_flag[1]) > 0.5):
                    self.best_s2 = [None, 100000, None, False]
                    self.stage = Stage.S2
                    self.command.reverse = True
                    self.mode = Vehicle_Mode.REVERSE
                    self.stage_odo = 0.0
                    self.print('> Commencing Stage 2: Reversing 1m ...', LogType.DEBUG)

            elif self.stage == Stage.S2:
                if self.stage_odo > 1.0:
                    self.command.reverse = False
                    self.mode       = Vehicle_Mode.FORWARD
                    self.stage_odo  = 0.0
                    #if self.best_s2[3]:
                    if True:
                        self.stage      = Stage.S3
                        self.best_odo   = (np.prod(self.best_s1[0:2]) + np.prod(self.best_s2[0:2])) / (self.best_s1[1]+self.best_s2[1]) # weighted mean
                        self.print('> Commencing Stage 3: Driving to target ... [%0.2f {%0.2f,%0.2f}]' % (self.best_odo, self.best_s1[0], self.best_s2[0]), LogType.DEBUG)
                    else:
                        self.print(C_I_YELLOW + '\tFailed to verify, skipping to Stage 5 ...' + C_RESET, LogType.DEBUG)
                        self.stage      = Stage.S4
                        self.time_now   = rospy.Time.now().to_sec() - 5
                else:
                    if curr_dist < self.best_s2[1]:
                        self.best_s2 = [self.odometer, curr_dist, self.label.match_index, self.best_s2[3]]

                    if (curr_dist < self.___cibbs[-1]):
                        self.best_s2[3] = True

            elif self.stage == Stage.S3:
                if self.odometer < self.best_odo:
                    self.mode = Vehicle_Mode.FORWARD
                    self.command.reverse = False
                elif self.odometer > self.best_odo:
                    self.mode = Vehicle_Mode.REVERSE
                    self.command.reverse = True

                if np.abs(self.odometer - self.best_odo) < 0.01:
                    self.stage = Stage.S4
                    self.time_now = rospy.Time.now().to_sec()
                    _truth_dist = p2p_dist_2d(np.array([
                                self.ref_ip.dataset['dataset']['px'][self.label.truth_index], 
                                self.ref_ip.dataset['dataset']['py'][self.label.truth_index]    
                            ]), np.array([getattr(self.label.gt_ego, i) for i in 'xy']) )
                    _match_dist = p2p_dist_2d(np.array([
                                self.ref_ip.dataset['dataset']['px'][self.label.truth_index], 
                                self.ref_ip.dataset['dataset']['py'][self.label.truth_index]    
                            ]), np.array([
                                self.ref_ip.dataset['dataset']['px'][self.label.match_index], 
                                self.ref_ip.dataset['dataset']['py'][self.label.match_index]
                            ]))

                    _truth_sign = np.sign(shortest_index_distance(self.info.group.current_ind, self.ref2path(self.label.truth_index), self.path_xyws.shape[0]))
                    _match_sign = np.sign(shortest_index_distance(self.info.group.current_ind, self.ref2path(self.label.match_index), self.path_xyws.shape[0]))

                    _truth_colour = C_I_GREEN if _truth_dist < 0.15 else C_I_YELLOW if _truth_dist < 0.5 else C_I_RED
                    _match_colour = C_I_GREEN if _match_dist < 0.15 else C_I_YELLOW if _match_dist < 0.5 else C_I_RED
                    self.print('\tTruth Err: %s%0.2f%s m | Match Err: %s%0.2f%s m' % \
                        (_truth_colour, _truth_dist * _truth_sign, C_RESET, _match_colour, _match_dist * _match_sign, C_RESET), LogType.INFO)
                    self.print('\tMatches: ' + str([self.best_s1[2], self.best_s2[2], self.label.match_index]), LogType.DEBUG)
                    self.print('> Commencing Stage 4: Pausing for a second ...', LogType.DEBUG)
            
            elif self.stage == Stage.S4:
                self.command.speed = [0.0]
                if rospy.Time.now().to_sec() - self.time_now > 1.0:
                    self.stage = Stage.S5
                    self.mode = Vehicle_Mode.FORWARD
                    self.command.reverse = False
                    self.command.speed = [0.5]
                    self.stage_odo = 0.0
                    self.print('> Commencing Stage 5: Driving for 0.5m ...', LogType.DEBUG)

            elif self.stage == Stage.S5:
                if self.stage_odo > 0.5:
                    self.print(C_I_GREEN + 'Mission Complete: Resetting to Stage 0.' + C_RESET + '\n', LogType.DEBUG)
                    self.stage = Stage.S0

            else:
                self.print('Complete.', LogType.DEBUG)
                sys.exit()

        self.command_pub.publish(self.command)

        # if not hasattr(self, 'datime'):
        #     self.datime = rospy.Time.now().to_sec()
        # if rospy.Time.now().to_sec() - self.datime > (60*5):
        #     sys.exit()

def do_args():
    parser = ap.ArgumentParser(prog="node2node_driver.py", 
                            description="Node2Node Driver",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='node2node_driver', rate=10.0)
    
    # Parse args...
    return vars(parser.parse_known_args()[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = Main_ROS_Class(**args)
        nmrc.print("Initialisation complete.", LogType.INFO)
        nmrc.main()
        nmrc.print("Operation complete.", LogType.INFO, ros=False) # False as rosnode likely terminated
        print(rospy.is_shutdown())
        sys.exit()
    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)