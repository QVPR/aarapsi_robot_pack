#!/usr/bin/env python3

import rospy
import argparse as ap
import numpy as np
import sys
import cv2

import matplotlib.pyplot as plt

from pyaarapsi.core.ros_tools               import NodeState, roslogger, LogType, compressed2np, np2compressed
from pyaarapsi.core.helper_tools            import formatException, angle_wrap, normalize_angle, roll
from pyaarapsi.vpr_classes.base             import base_optional_args

# Import break-out libraries (they exist to help keep me sane and make this file readable):
from pyaarapsi.pathing.enums                import * # Enumerations
from pyaarapsi.pathing.basic                import * # Helper functions
from pyaarapsi.pathing.base                 import Zone_Return_Class

'''
Path Follower

Node description.

'''

class Follower_Class(Zone_Return_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def roll_match(self, ind: int):
        resize          = [int(self.IMG_HFOV.get()), 8]
        img_dims        = self.IMG_DIMS.get()
        query_raw       = cv2.cvtColor(compressed2np(self.label.query_image), cv2.COLOR_BGR2GRAY)
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
        t_current_ind           = calc_current_ind(self.slam_ego, self.path_xyws)
        t_zone                  = calc_current_zone(t_current_ind, self.num_zones, self.zone_indices)

        if self.command_mode == Command_Mode.VPR: # If we are estimating pose, calculate via VPR:
            current_ind         = self.label.match_index
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
        speed           = 0.0
        error_yaw       = 0.0
        target_ind      = 0
        adj_lookahead   = self.lookahead
        new_lin         = 0.0
        new_ang         = 0.0

        # If the current command mode is a path-following exercise:
        if self.command_mode in [Command_Mode.VPR, Command_Mode.SLAM]:
            # calculate heading, cross-track, velocity errors and the target index:
            target_ind, adj_lookahead = calc_target(current_ind, self.lookahead, self.lookahead_mode, self.path_xyws)
            if self.command_mode == Command_Mode.SLAM:
                error_yaw = calc_yaw_error(ego, self.path_xyws[:,0], self.path_xyws[:,1], target_ind)
            else:
                error_yaw = angle_wrap(self.path_xyws[target_ind, 2] - ego[2], 'RAD')
            speed = self.path_xyws[current_ind, 3]
            # publish a pose to visualise the target:
            self.publish_pose(target_ind, self.goal_pub)
            # generate a new command to drive the robot based on the errors:
            new_command = self.make_new_command(speed, error_yaw)
            new_lin     = new_command.linear.x
            new_ang     = new_command.angular.z

            # If a safety is 'enabled':
            if self.safety_mode in [Safety_Mode.SLOW, Safety_Mode.FAST]:
                # send command:
                self.cmd_pub.publish(new_command)

        # If the current command mode is set to return-to-nearest-zone-boundary:
        elif self.command_mode == Command_Mode.ZONE_RETURN:
            self.zone_return(ego, current_ind)

        # Print diagnostics:
        if self.PRINT_DISPLAY.get():
            self.print_display(new_linear=new_lin, new_angular=new_ang, 
                               current_ind=current_ind, zone=t_zone, 
                               lin_path_err=lin_err, ang_path_err=ang_err, speed=speed, error_yaw=error_yaw)
        self.publish_controller_info(current_ind, target_ind, heading_fixed, t_zone, adj_lookahead)

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