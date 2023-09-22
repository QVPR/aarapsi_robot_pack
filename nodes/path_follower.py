#!/usr/bin/env python3

import rospy
import argparse as ap
import numpy as np
import sys
import copy

from pyaarapsi.core.ros_tools               import NodeState, roslogger, LogType
from pyaarapsi.core.helper_tools            import formatException, angle_wrap, normalize_angle, p2p_dist_2d, r2d
from pyaarapsi.core.enum_tools              import enum_name
from pyaarapsi.core.transforms              import apply_homogeneous_transform, Transform_Builder
from pyaarapsi.vpr_classes.base             import base_optional_args

# Import break-out libraries (they exist to help keep me sane and make this file readable):
from pyaarapsi.pathing.enums                import * # Enumerations
from pyaarapsi.pathing.basic                import * # Helper functions
from pyaarapsi.pathing.base                 import Main_ROS_Class

from pyaarapsi.core.vars                    import C_CLEAR

from aarapsi_robot_pack.msg                 import GoalExpResults, xyw

'''
Path Follower

Node description.

'''

class Follower_Class(Main_ROS_Class):
    def __init__(self, **kwargs):
        '''

        Node Initialisation

        '''
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def main(self):
        '''

        Main function

        Handles:
        - Pre-loop duties
        - Keeping main loop alive
        - Exit

        '''
        self.set_state(NodeState.MAIN)

        self.load_dataset() # Load reference data
        self.make_path()    # Generate reference path

        # Publish path, speed, zones for RViz:
        self.path_pub.publish(self.viz_path)
        self.speed_pub.publish(self.viz_speeds)
        self.zones_pub.publish(self.viz_zones)

        self.ready = True

        # Commence main loop; do forever:
        self.print('Entering main loop.')
        while not rospy.is_shutdown():
            try:
                # Denest main loop; wait for new messages:
                if not (self.new_label):# and self.new_robot_odom):
                    self.print("Waiting for new position information...", LogType.DEBUG, throttle=10)
                    rospy.sleep(0.005)
                    continue
                
                self.rate_obj.sleep()
                self.new_label          = False
                #self.new_robot_odom     = False

                self.loop_contents()

            except rospy.exceptions.ROSInterruptException as e:
                pass
            except Exception as e:
                if self.parameters_ready:
                    raise Exception('Critical failure. ' + formatException()) from e
                else:
                    self.print('Main loop exception, attempting to handle; waiting for parameters to update. Details:\n' + formatException(), LogType.DEBUG, throttle=5)
                    rospy.sleep(0.5)

    def path_follow(self, ego, current_ind):
        '''
        
        Follow a pre-defined path
        
        '''

        # Ensure robot is within path limits
        if not self.check_for_safety_stop():
            return
        
        # Calculate heading, cross-track, velocity errors and the target index:
        self.target_ind, self.adjusted_lookahead = calc_target(current_ind, self.lookahead, self.lookahead_mode, self.path_xyws)

        # Publish a pose to visualise the target:
        publish_xyw_pose(self.path_xyws[self.target_ind], self.goal_pub)

        # Calculate control signal for angular velocity:
        if self.command_mode == Command_Mode.VPR:
            error_ang = angle_wrap(self.path_xyws[self.target_ind, 2] - ego[2], 'RAD')
        else:
            error_ang = calc_yaw_error(ego, self.path_xyws[self.target_ind])

        # Calculate control signal for linear velocity
        error_lin = self.path_xyws[current_ind, 3]

        # Send a new command to drive the robot based on the errors:
        self.try_send_command(error_lin, error_ang)

    def zone_return(self, ego, target, ignore_heading=False):
        '''

        Handle an autonomous movement to a target
        - Picks nearest target
        - Drives to target
        - Turns on-spot to face correctly

        Stages:
        Return_STAGE.UNSET: Stage 0 - New request: identify zone target.
        Return_STAGE.DIST:  Stage 1 - Distance from target exceeds 5cm: head towards target.
        Return_STAGE.TURN:  Stage 2 - Heading error exceeds 1 degree: turn on-the-spot towards target heading.
        Return_STAGE.DONE:  Stage 3 - FINISHED.

        '''

        publish_xyw_pose(target, self.goal_pub)

        if self.return_stage == Return_Stage.DONE:
            return True

        yaw_err     = calc_yaw_error(ego, target)
        ego_cor     = self.update_COR(ego) # must improve accuracy of centre-of-rotation as we do on-the-spot turns
        dist        = np.sqrt(np.square(ego_cor[0] - target[0]) + np.square(ego_cor[1] - target[1]))
        head_err    = angle_wrap(target[2] - ego[2], 'RAD')

        # If stage 1: calculate distance to target (lin_err) and heading error (ang_err)
        if self.return_stage == Return_Stage.DIST:
            if abs(yaw_err) < np.pi/18:
                ang_err = np.sign(yaw_err) * np.min([np.max([0.1, -0.19*abs(yaw_err)**2 + 0.4*abs(yaw_err) - 0.007]),1])
                lin_err = np.max([0.1, 2 * np.log10(dist + 0.7)])
            else:
                lin_err = 0
                ang_err = np.sign(yaw_err) * 0.2

            # If we're within 5 cm of the target, stop and move to stage 2
            if dist < 0.05:
                self.return_stage = Return_Stage.TURN

        # If stage 2: calculate heading error and turn on-the-spot
        elif self.return_stage == Return_Stage.TURN:
            lin_err = 0
            # If heading error is less than 1 degree, stop and move to stage 3
            if (abs(head_err) < np.pi/180) or ignore_heading:
                self.set_command_mode(Command_Mode.STOP)
                self.return_stage = Return_Stage.DONE
                ang_err = 0

                return True
            else:
                ang_err = np.sign(head_err) * np.max([0.1, abs(head_err)])
        else:
            raise Exception('Bad return stage [%s].' % str(self.return_stage))

        self.try_send_command(lin_err,ang_err)
        return False
    
    def point_and_shoot(self, start, ego, point, shoot):
        '''

        Handle an autonomous movement to a target
        - Aims at target
        - Drives to target

        Stages:
        Point_Shoot_Stage.INIT:  Stage 0 - New request: start. 
        Point_Shoot_Stage.POINT: Stage 1 - Use point amount to correct within 2 degrees of error
        Point_Shoot_Stage.SHOOT: Stage 2 - Use shoot amount to correct witihn 10cm of error
        Point_Shoot_Stage.DONE:  Stage 3 - FINISHED.

        '''

        if self.point_shoot_stage == Point_Shoot_Stage.DONE:
            return True
        
        elif self.point_shoot_stage == Point_Shoot_Stage.INIT:
            self.point_shoot_stage = Point_Shoot_Stage.POINT

        elif self.point_shoot_stage == Point_Shoot_Stage.POINT:
            point_err = normalize_angle(point - (ego[2] - start[2]))
            if np.abs(point_err) > np.pi/90:
                self.try_send_command(0, np.sign(point_err) * 0.3)
            else:
                self.point_shoot_stage = Point_Shoot_Stage.SHOOT

        elif self.point_shoot_stage == Point_Shoot_Stage.SHOOT:
            shoot_err = shoot - p2p_dist_2d(start, ego)
            if np.abs(shoot_err) > 0.1:
                self.try_send_command(0.3,0)
            else:
                self.point_shoot_stage = Point_Shoot_Stage.DONE

        return False
    
    def experiment(self):

        if self.new_goal: # If 2D Nav Goal is used to request a goal
            _closest_goal_ind   = calc_current_ind(self.goal_pose, self.path_xyws)
            _stopping           = 0.5 # provide 50cm for vehicle to come to a halt
            self.exp_stop_SLAM  = np.argmin(np.abs(self.path_sum - (self.path_sum[_closest_goal_ind] - _stopping))) 
            # Some magic numbers, 0.10m and 0.05m, to ensure the historical data gets cleaned out between experiments
            self.exp_start_SLAM = np.argmin(np.abs(self.path_sum - (self.path_sum[self.exp_stop_SLAM] - (self.SLICE_LENGTH.get()+0.10)))) 
            self.exp_dist       = self.path_sum[self.exp_stop_SLAM] - self.path_sum[self.exp_start_SLAM]
            if self.exp_dist < (self.SLICE_LENGTH.get()+0.05): # Warn if the historical data may not clear
                self.print('Proposed experiment length: %0.2f [m]: Historical data may be retained!' % self.exp_dist, LogType.WARN)
            else:
                self.print('Proposed experiment length: %0.2f [m].' % self.exp_dist)
            self.new_goal       = False
            self.EXPERIMENT_MODE.set(Experiment_Mode.INIT)
            self.print('[Experiment] Initialisation phase.')

            publish_xyzrpy_pose([self.path_xyws[self.exp_start_SLAM,0], self.path_xyws[self.exp_start_SLAM,1], -0.5, 0, -np.pi/2, 0], self.experi_start_pub)
            publish_xyzrpy_pose([self.path_xyws[self.exp_stop_SLAM,0], self.path_xyws[self.exp_stop_SLAM,1], -0.5, 0, -np.pi/2, 0], self.experi_finish_pub)

        elif self.exp_dist is None:
            self.print('Experiment pending 2D Nav Goal ...', LogType.WARN, throttle=30)
            return
        
        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.INIT:
            self.exp_results    = GoalExpResults()
            self.exp_results.id = self.exp_count
            self.exp_count      = self.exp_count + 1
            self.exp_results.path_start_pos     = xyw(*self.path_xyws[self.exp_start_SLAM,0:3])
            self.exp_results.path_finish_pos    = xyw(*self.path_xyws[self.exp_stop_SLAM,0:3])
            self.exp_results.mode               = enum_name(self.TECHNIQUE.get())
            self.return_stage   = Return_Stage.DIST
            self.EXPERIMENT_MODE.set(Experiment_Mode.ALIGN)
            self.print('[Experiment] Align phase.')
        
        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.ALIGN:
            if self.zone_return(self.slam_ego, self.path_xyws[self.exp_start_SLAM]):
                self.EXPERIMENT_MODE.set(Experiment_Mode.DRIVE_PATH)
                self.print('[Experiment] Driving along path.')
                return

        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.DRIVE_PATH:
            if (self.path_sum[self.slam_current_ind] - self.path_sum[self.exp_start_SLAM]) < self.exp_dist:
                self.path_follow(self.slam_ego, self.slam_current_ind)
            else:
                self.EXPERIMENT_MODE.set(Experiment_Mode.HALT1)
                self.print('[Experiment] Halting ... (1)')
                return

        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.HALT1:
            if len(self.robot_velocities):
                if np.sum(self.robot_velocities) < 0.05:
                    self.EXPERIMENT_MODE.set(Experiment_Mode.DRIVE_GOAL)
                    self.print('[Experiment] Driving to goal.')
                    self.point_shoot_stage                  = Point_Shoot_Stage.INIT
                    self.point_shoot_point                  = calc_yaw_error(self.current_hist_pos, self.goal_pose)
                    self.point_shoot_shoot                  = p2p_dist_2d(self.current_hist_pos, self.goal_pose)
                    self.point_shoot_start                  = self.robot_ego
                    self.exp_results.point                  = self.point_shoot_point
                    self.exp_results.shoot                  = self.point_shoot_shoot
                    self.exp_results.localisation           = copy.deepcopy(self.current_results)
                    self.exp_results.robot_goal_start_pos   = xyw(*self.robot_ego)
                    self.exp_results.slam_goal_start_pos    = xyw(*self.slam_ego)
                    return
            self.try_send_command(0,0)
        
        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.DRIVE_GOAL:
            if self.point_and_shoot(start=self.point_shoot_start, ego=self.robot_ego, point=self.point_shoot_point, shoot=self.point_shoot_shoot):
                self.EXPERIMENT_MODE.set(Experiment_Mode.HALT2)
                self.print('[Experiment] Halting ... (2)')
                return

        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.HALT2:
            if len(self.robot_velocities):
                if np.sum(self.robot_velocities) < 0.05:
                    self.exp_results.robot_goal_finish_pos   = xyw(*self.robot_ego)
                    self.exp_results.slam_goal_finish_pos    = xyw(*self.slam_ego)
                    self.experi_results_pub.publish(self.exp_results)
                    self.EXPERIMENT_MODE.set(Experiment_Mode.DONE)
                    return
            self.try_send_command(0,0)

        elif self.EXPERIMENT_MODE.get() == Experiment_Mode.DONE:
            self.print('[Experiment] Complete.', throttle=30)

    def loop_contents(self):
        '''
        
        Main Loop

        '''

        # Calculate current SLAM position and zone:
        self.slam_current_ind       = calc_current_ind(self.slam_ego, self.path_xyws)
        self.slam_zone              = calc_current_zone(self.slam_current_ind, self.num_zones, self.zone_indices)

        self.update_zone_target()

        # Calculate/estimate current ego:
        if self.command_mode == Command_Mode.VPR: # If we are estimating pose, calculate via VPR:
            self.est_current_ind    = self.label.match_index
            self.heading_fixed      = normalize_angle(angle_wrap(self.vpr_ego[2] + self.roll_match(self.est_current_ind), 'RAD'))
            self.ego                = [self.vpr_ego[0], self.vpr_ego[1], self.heading_fixed]

        else: # If we are not estimating pose, use everything from the ground truth:
            self.est_current_ind    = self.slam_current_ind
            self.heading_fixed      = normalize_angle(angle_wrap(self.slam_ego[2] + self.roll_match(self.est_current_ind), 'RAD'))
            self.ego                = self.slam_ego
        
        publish_xyw_pose(self.path_xyws[self.slam_current_ind], self.slam_pub) # Visualise SLAM nearest position on path
        self.update_historical_data() # Manage storage of historical data

        # Denest; check if stopped:
        if self.command_mode in [Command_Mode.STOP]:
            pass

        # Else: if the current command mode is a path-following exercise:
        elif self.command_mode in [Command_Mode.VPR, Command_Mode.SLAM]:
            self.path_follow(self.ego, self.est_current_ind)

        # Else: if the current command mode is set to return-to-nearest-zone-boundary:
        elif self.command_mode in [Command_Mode.ZONE_RETURN]:
        
            # If stage 0: determine target and move to stage 1
            if self.return_stage == Return_Stage.UNSET:
                if self.saved_index == -1: # No saved zone
                    self.zone_index  = calc_nearest_zone(self.zone_indices, self.est_current_ind, self.path_xyws.shape[0])
                else:
                    self.zone_index  = self.saved_index
                self.return_stage = Return_Stage.DIST
            
            self.zone_return(self.ego, self.path_xyws[self.zone_index,:])

        # Else: if the current command mode is set to special functions (experiments, testing):
        if not (self.EXPERIMENT_MODE.get() == Experiment_Mode.UNSET):
            self.experiment()

        # Print HMI:
        if self.PRINT_DISPLAY.get():
            self.print_display()
        self.publish_controller_info()

def do_args():
    parser = ap.ArgumentParser(prog="path_follower.py", 
                                description="Path Follower",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='path_follower')

    # Parse args...
    return vars(parser.parse_known_args()[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = Follower_Class(**args)
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