#!/usr/bin/env python3

import rospy
import argparse as ap
import numpy as np
import sys
import cv2

import matplotlib.pyplot as plt

from std_msgs.msg           import Header
from geometry_msgs.msg      import PoseStamped, Point, Twist

from pyaarapsi.core.ros_tools               import NodeState, roslogger, LogType, q_from_yaw, compressed2np, np2compressed, q_from_rpy
from pyaarapsi.core.helper_tools            import formatException, angle_wrap, normalize_angle, m2m_dist, roll
from pyaarapsi.vpr_classes.base             import base_optional_args
from pyaarapsi.core.enum_tools              import enum_value

# Import break-out libraries (they exist to help keep me sane and make this file readable):
from pyaarapsi.pathing.enums                import * # Enumerations
from pyaarapsi.pathing.make_paths           import * # Helper functions
from pyaarapsi.pathing.base                 import Main_ROS_Class # For ROS and data loading/generation related functions

'''
Path Follower

Node description.

'''

class Follower_Class(Main_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def global2local(self, ego):

        Tx  = self.path_xyws[:,0] - ego[0]
        Ty  = self.path_xyws[:,1] - ego[1]
        R   = np.sqrt(np.power(Tx, 2) + np.power(Ty, 2))
        A   = np.arctan2(Ty, Tx) - ego[2]

        return list(np.multiply(np.cos(A), R)), list(np.multiply(np.sin(A), R))
        
    def calc_current_zone(self, ind):
        # The closest zone boundary that is 'behind' the closest index (in the direction of the path):
        zone                = np.max(np.arange(self.num_zones)[np.array(self.zone_indices[0:-1]) <= ind] + 1)
        return zone
    
    def calc_current_ind(self, ego):
        # Closest index based on provided ego:
        ind         = np.argmin(m2m_dist(self.path_xyws[:,0:2], ego[0:2], True), axis=0)
        return ind

    def calc_yaw_y_errors(self, ego, target_ind: int = None):
        rel_x, rel_y    = self.global2local(ego) # Convert to local coordinates
        # Heading error (angular):
        error_yaw       = normalize_angle(np.arctan2(rel_y[target_ind], rel_x[target_ind]))
        # Cross-track error (perpendicular distance):
        error_y         = rel_y[target_ind]
        return {'error_yaw': error_yaw, 'error_y': error_y}
    
    def calc_all_errors(self, ego, current_ind: int):
        adj_lookahead = np.max([self.lookahead * (self.path_xyws[current_ind, 3]/np.max(self.path_xyws[:, 3].flatten())), 0.3])
        if self.lookahead_mode == Lookahead_Mode.INDEX:
            target_ind  = (current_ind + int(np.round(adj_lookahead))) % self.path_xyws.shape[0]
        elif self.lookahead_mode == Lookahead_Mode.DISTANCE:
            target_ind  = current_ind
            # find first index at least lookahead-distance-away from current index:
            dist        = np.sqrt(np.sum(np.square(self.path_xyws[target_ind, 0:2] - self.path_xyws[current_ind, 0:2])))
            while dist < adj_lookahead:
                target_ind  = (target_ind + 1) % self.path_xyws.shape[0] 
                dist        = np.sqrt(np.sum(np.square(self.path_xyws[target_ind, 0:2] - self.path_xyws[current_ind, 0:2])))
        else:
            raise Exception('Unknown lookahead_mode: %s' % str(self.lookahead_mode))
        
        if self.command_mode == Command_Mode.SLAM:
            errors = self.calc_yaw_y_errors(ego, target_ind)
        else:
            errors = {'error_yaw': angle_wrap(self.path_xyws[target_ind, 2] - ego[2], 'RAD'), 'error_y': 0.0}
        errors.update({'error_v': self.path_xyws[current_ind, 3]})
        return errors, target_ind, adj_lookahead
    
    def make_path(self):
        # generate an n row, 4 column array (x, y, yaw, speed) corresponding to each reference image (same index)
        self.path_xyws  = np.transpose(np.stack([self.ip.dataset['dataset']['px'].flatten(), 
                                                 self.ip.dataset['dataset']['py'].flatten(),
                                                 self.ip.dataset['dataset']['pw'].flatten(),
                                                 make_speed_array(self.ip.dataset['dataset']['pw'].flatten())]))
        
        # determine zone number, length, indices:
        path_sum, path_len               = calc_path_stats(self.path_xyws)
        self.zone_length, self.num_zones = calc_zone_stats(path_len, self.ZONE_LENGTH.get(), self.ZONE_NUMBER.get(), )
        _end                             = [self.path_xyws.shape[0] + (int(not self.LOOP_PATH.get()) - 1)]
        self.zone_indices                = [np.argmin(np.abs(path_sum-(self.zone_length*i))) for i in np.arange(self.num_zones)] + _end
        
        # generate stuff for visualisation:
        self.path_indices                = [np.argmin(np.abs(path_sum-(0.2*i))) for i in np.arange(int(5 * path_len))]
        self.viz_path, self.viz_speeds   = make_path_speeds(self.path_xyws, self.path_indices)
        self.viz_zones                   = make_zones(self.path_xyws, self.zone_indices)

    def roll_match(self, ind: int):
        resize          = [int(self.IMG_HFOV.get()), 8]
        img_dims        = self.IMG_DIMS.get()
        query_raw       = cv2.cvtColor(compressed2np(self.state_msg.queryImage), cv2.COLOR_BGR2GRAY)
        img             = cv2.resize(query_raw, resize)
        img_mask        = np.ones(img.shape)

        _b              = int(resize[0] / 2)
        sliding_options = range((-_b) + 1, _b)

        against_image   = cv2.resize(np.reshape(self.ip.dataset['dataset']['RAW'][ind], [img_dims[1], img_dims[0]]), resize)
        options_stacked = np.stack([roll(against_image, i).flatten() for i in sliding_options])
        img_stacked     = np.stack([(roll(img_mask, i)*img).flatten() for i in sliding_options])
        matches         = np.sum(np.square(img_stacked - options_stacked),axis=1)
        yaw_fix_deg     = sliding_options[np.argmin(matches)]
        yaw_fix_rad     = normalize_angle(yaw_fix_deg * np.pi / 180.0)

        if self.PUBLISH_ROLLMATCH.get():
            fig, ax = plt.subplots()
            ax.plot(sliding_options, matches)
            ax.plot(sliding_options[np.argmin(matches)], matches[np.argmin(matches)])
            ax.set_title('%s' % str([sliding_options[np.argmin(matches)], matches[np.argmin(matches)]]))
            fig.canvas.draw()
            img_np_raw_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_np_raw      = img_np_raw_flat.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_np          = np.flip(img_np_raw, axis=2) # to bgr format, for ROS
            plt.close('all') # close matplotlib
            plt.show()
            img_msg = np2compressed(img_np)
            img_msg.header.stamp = rospy.Time.now()
            img_msg.header.frame_id = 'map'
            self.rollmatch_pub.publish(img_msg)

        return yaw_fix_rad

    def publish_pose(self, goal_ind: int, pub) -> None:
        # Update visualisation of current goal/target pose
        goal                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        goal.pose.position      = Point(x=self.path_xyws[goal_ind,0], y=self.path_xyws[goal_ind,1], z=0.0)
        goal.pose.orientation   = q_from_yaw(self.path_xyws[goal_ind,2])
        pub.publish(goal)

    def make_new_command(self, error_v: float, error_y: float, error_yaw: float) -> Twist:
        # Determine maximum linear and angular speeds based on safety state:
        if self.safety_mode == Safety_Mode.SLOW:
            lin_max = self.SLOW_LIN_VEL_MAX.get()
            ang_max = self.SLOW_LIN_VEL_MAX.get()
        elif self.safety_mode == Safety_Mode.FAST:
            lin_max = self.FAST_LIN_VEL_MAX.get()
            ang_max = self.FAST_LIN_VEL_MAX.get()
        else:
            lin_max = 0
            ang_max = 0
        
        # If we're not estimating where we are, we're ignoring the SVM, or we have a good point:
        if not (self.command_mode == Command_Mode.VPR) or self.REJECT_MODE.get() == Reject_Mode.NONE or self.state_msg.mStateBin:
            # ... then calculate based on the controller errors:
            new_linear          = np.sign(error_v)   * np.min([abs(error_v),   lin_max])
            new_angular         = np.sign(error_yaw) * np.min([abs(error_yaw), ang_max])
        
        # otherwise, if we want to reject the point we received:
        else:
            # ... then we must decide on what new command we should send.
            # If we want to stop:
            if self.REJECT_MODE.get() == Reject_Mode.STOP:
                new_linear      = 0.0
                new_angular     = 0.0

            # If we want to resend the last command:
            elif self.REJECT_MODE.get() == Reject_Mode.OLD:
                new_linear      = self.old_linear
                new_angular     = self.old_angular
                
            # If we want to quickly decelerate:
            elif self.REJECT_MODE.get() == Reject_Mode.OLD_50:
                new_linear      = self.old_linear * 0.5
                new_angular     = self.old_angular * 0.5

            # If we want to slowly decelerate:
            elif self.REJECT_MODE.get() == Reject_Mode.OLD_90:
                new_linear      = self.old_linear * 0.9
                new_angular     = self.old_angular * 0.9

            else:
                raise Exception('Unknown rejection mode %s' % str(self.REJECT_MODE.get()))

        # Update the last-sent-command to be the one we just made:
        self.old_linear         = new_linear
        self.old_angular        = new_angular

        # Generate the actual ROS command:
        new_msg                 = Twist()
        new_msg.linear.x        = new_linear
        new_msg.angular.z       = new_angular
        return new_msg
    
    def update_COR(self, ego):
        # Update centre-of-rotation for visualisation and precise alignment:
        COR_x                   = ego[0] + self.COR_OFFSET.get() * np.cos(ego[2])
        COR_y                   = ego[1] + self.COR_OFFSET.get() * np.sin(ego[2])
        pose                    = PoseStamped(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        pose.pose.position      = Point(x=COR_x, y=COR_y)
        pose.pose.orientation   = q_from_rpy(0, -np.pi/2, 0)
        self.COR_pub.publish(pose)
        return [COR_x, COR_y]
    
    def zone_return(self, ego, current_ind):
        '''
        Handle an autonomous return-to-zone:
        - Picks nearest target
        - Drives to target
        - Turns on-spot to face correctly

        Stages:
        Return_STAGE.UNSET: Stage 0 - New request: identify zone target.
        Return_STAGE.DIST:  Stage 1 - Distance from target exceeds 5cm: head towards target.
        Return_STAGE.TURN:  Stage 2 - Heading error exceeds 1 degree: turn on-the-spot towards target heading.
        Return_STAGE.DONE:  Stage 3 - FINISHED.

        '''

        if self.return_stage == Return_Stage.DONE:
            return
        
        # If stage 0: determine target and move to stage 1
        if self.return_stage == Return_Stage.UNSET:
            self.zone_index  = self.zone_indices[np.argmin(m2m_dist(current_ind, np.transpose(np.matrix(self.zone_indices))))] % self.path_xyws.shape[0]
            self.return_stage = Return_Stage.DIST

        self.publish_pose(self.zone_index, self.goal_pub)

        errors      = self.calc_yaw_y_errors(ego, target_ind=self.zone_index)
        ego_cor     = self.update_COR(ego) # must improve accuracy of centre-of-rotation as we do on-the-spot turns
        dist        = np.sqrt(np.square(ego_cor[0]-self.path_xyws[self.zone_index, 0]) + np.square(ego_cor[1]-self.path_xyws[self.zone_index, 1]))
        yaw_err     = errors.pop('error_yaw')
        head_err    = self.path_xyws[self.zone_index, 2] - ego[2]

        # If stage 1: calculate distance to target (lin_err) and heading error (ang_err)
        if self.return_stage == Return_Stage.DIST:
            if abs(yaw_err) < np.pi/6:
                ang_err = np.sign(yaw_err) * np.max([0.1, -0.19*abs(yaw_err)**2 + 0.4*abs(yaw_err) - 0.007])
                lin_err = np.max([0.1, -(1/3)*dist**2 + (19/30)*dist - 0.06])
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
            if abs(head_err) < np.pi/180:
                self.set_command_mode(Command_Mode.STOP)
                self.return_stage = Return_Stage.DONE
                ang_err = 0
                return # DONE! :)
            else:
                ang_err = np.sign(head_err) * np.max([0.1, abs(head_err)])
        else:
            raise Exception('Bad return stage [%s].' % str(self.return_stage))

        new_msg     = self.make_new_command(error_v=lin_err, error_yaw=ang_err, error_y=errors.pop('error_y'))
        self.cmd_pub.publish(new_msg)

    def check_for_safety_stop(self, lin_err, ang_err):
        if self.command_mode in [Command_Mode.VPR, Command_Mode.SLAM]:
            if lin_err > self.LINSTOP_OVERRIDE.get() and self.LINSTOP_OVERRIDE.get() > 0:
                self.set_command_mode(Command_Mode.STOP)
            elif ang_err > self.ANGSTOP_OVERRIDE.get() and self.ANGSTOP_OVERRIDE.get() > 0:
                self.set_command_mode(Command_Mode.STOP)

    def main(self):
        # Main loop process
        self.set_state(NodeState.MAIN)

        self.load_dataset() # Load reference data
        self.make_path()    # Generate reference path

        # Publish path, speed, zones for RViz:
        self.path_pub.publish(self.viz_path)
        self.speed_pub.publish(self.viz_speeds)
        self.zones_pub.publish(self.viz_zones)

        # Wait for initial wheel odometry and VPR position:
        while not (self.new_robot_ego and self.new_vpr_ego):
            self.rate_obj.sleep()
            self.print('Waiting for start position information...', throttle=5)
            if rospy.is_shutdown():
                return
        self.ready          = True
        self.new_vpr_ego    = False
        self.new_robot_ego  = False

        # Commence main loop; do forever:
        self.print('Entering main loop.')
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

        # Denest main loop; wait for new messages:
        if not (self.new_robot_ego and self.new_vpr_ego):
            self.print("Waiting for new position information...", LogType.DEBUG, throttle=10)
            rospy.sleep(0.005)
            return
        self.rate_obj.sleep()
        self.new_vpr_ego    = False
        self.new_robot_ego  = False

        # Calculate current SLAM position and zone:
        t_current_ind           = self.calc_current_ind(self.slam_ego)
        t_zone                  = self.calc_current_zone(t_current_ind)

        if self.command_mode == Command_Mode.VPR: # If we are estimating pose, calculate via VPR:
            current_ind         = self.state_msg.data.matchId
            rm_corr             = self.roll_match(current_ind)
            heading_fixed       = normalize_angle(angle_wrap(self.vpr_ego[2] + rm_corr, 'RAD'))
            ego                 = [self.vpr_ego[0], self.vpr_ego[1], heading_fixed]

        else: # If we are not estimating pose, use everything from the ground truth:
            current_ind         = t_current_ind
            rm_corr             = self.roll_match(t_current_ind)
            heading_fixed       = normalize_angle(angle_wrap(self.slam_ego[2] + rm_corr, 'RAD'))
            ego                 = self.slam_ego

        # Visualise SLAM nearest position on path: 
        self.publish_pose(t_current_ind, self.slam_pub)
        
        # Calculate perpendicular (lin) and angular (ang) path errors:
        lin_err, ang_err = calc_path_errors(self.slam_ego, t_current_ind, self.path_xyws)

        # If path-following, ensure robot is within path limits:
        self.check_for_safety_stop(lin_err, ang_err)

        # Set default values for printing:
        new_command     = Twist()
        errs            = {'error_yaw': 0, 'error_y': 0, 'error_v': 0}
        target_ind      = 0
        adj_lookahead   = self.lookahead

        # If the current command mode is a path-following exercise:
        if self.command_mode in [Command_Mode.VPR, Command_Mode.SLAM]:
            # calculate heading, cross-track, velocity errors and the target index:
            errs, target_ind, adj_lookahead = self.calc_all_errors(ego, current_ind)
            # publish a pose to visualise the target:
            self.publish_pose(target_ind, self.goal_pub)
            # generate a new command to drive the robot based on the errors:
            new_command = self.make_new_command(**errs)

            # If a safety is 'enabled':
            if self.safety_mode in [Safety_Mode.SLOW, Safety_Mode.FAST]:
                # send command:
                self.cmd_pub.publish(new_command)

        # If the current command mode is set to return-to-nearest-zone-boundary:
        elif self.command_mode == Command_Mode.ZONE_RETURN:
            self.zone_return(ego, current_ind)

        # Print diagnostics:
        if self.PRINT_DISPLAY.get():
            self.print_display(new_linear=new_command.linear.x, new_angular=new_command.angular.z, 
                               current_ind=current_ind, zone=t_zone, 
                               lin_path_err=lin_err, ang_path_err=ang_err, **errs)
        self.publish_controller_info(current_ind, target_ind, heading_fixed, t_zone, adj_lookahead)

    def print(self, *args, **kwargs):
        arg_list = list(args) + [kwargs[k] for k in kwargs]
        log_level = enum_value(LogType.INFO)
        for i in arg_list:
            if isinstance(i, LogType):
                log_level = enum_value(i)
                break
        if (enum_value(self.LOG_LEVEL.get()) <= log_level) and super().print(*args, **kwargs):
            self.print_lines += 1

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