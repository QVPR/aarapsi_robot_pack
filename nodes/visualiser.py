#!/usr/bin/env python3

import rospy
import rospkg
import argparse as ap
import numpy as np
import sys
import os
import cv2

from std_msgs.msg           import String, ColorRGBA, Header
from geometry_msgs.msg      import Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from aarapsi_robot_pack.msg import Label
from sensor_msgs.msg        import CompressedImage

from pyaarapsi.core.argparse_tools          import check_positive_int, check_bool
from pyaarapsi.core.ros_tools               import NodeState, roslogger, LogType, set_rospy_log_lvl, q_from_yaw, compressed2np, np2compressed
from pyaarapsi.core.helper_tools            import formatException, angle_wrap
from pyaarapsi.core.enum_tools              import enum_name
from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.vpr_image_methods import convert_img_to_uint8, label_image, apply_icon
from pyaarapsi.vpr_classes.base             import Base_ROS_Class, base_optional_args

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = "monospace"
matplotlib.rc('font',family='monospace')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
matplotlib.rcParams['axes.linewidth'] = 0.0 #set the value globally

'''
Visualiser

Create visualisations that may be time expensive separately
to reduce interruptions to normal pipeline execution.

'''

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()
        
        self.node_ready(kwargs['order_id'])

    def init_params(self, rate_num, log_level, reset):
        super().init_params(rate_num, log_level, reset)

        self.SVM_MODE        = self.params.add(self.nodespace + "/svm_mode",            False,                  check_bool,                           force=reset)
        self.NUM_MARKERS     = self.params.add(self.nodespace + "/num_markers",         100,                    check_positive_int,                   force=reset)
        

    def init_vars(self):
        super().init_vars()

        self.icon_size          = 50
        self.icon_path          = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/media"

        self.good_icon          = cv2.resize(cv2.imread(self.icon_path + "/tick.png", cv2.IMREAD_UNCHANGED),  (self.icon_size,)*2, interpolation = cv2.INTER_AREA)
        self.poor_icon          = cv2.resize(cv2.imread(self.icon_path + "/cross.png", cv2.IMREAD_UNCHANGED), (self.icon_size,)*2, interpolation = cv2.INTER_AREA)

        self.state_msg          = Label()
        self.new_state_msg      = False

        self.last_ego           = [0.0, 0.0, 0.0]

        self.markers            = MarkerArray()
        self.markers.markers    = []
        self.marker_id          = 0

        self.colour_good        = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.80)
        self.colour_svm_good    = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.20)
        self.colour_bad         = ColorRGBA(r=0.9, g=0.2, b=0.1, a=0.20)
        self.colour_lost        = ColorRGBA(r=0.1, g=0.1, b=0.1, a=0.08)

        self.tol_hist           = np.zeros((100, 8))
        
        try:
            # Process reference data
            dataset_dict            = self.make_dataset_dict()
            self.ip    = VPRDatasetProcessor(dataset_dict, try_gen=False, ros=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

    def init_rospy(self):
        super().init_rospy()
        
        self.state_sub      = rospy.Subscriber(self.namespace + "/state", Label, self.state_callback, queue_size=1)
        self.confidence_pub  = self.add_pub(self.namespace + '/confidence',             MarkerArray,                                queue_size=1)
        self.display_pub     = self.add_pub(self.namespace + '/display/compressed',     CompressedImage,                            queue_size=1)

    def state_callback(self, msg: Label):
        self.state_msg     = msg
        self.new_state_msg = True

        # Generate / record statistics:
        self.gt_ego             = [self.state_msg.gt_ego.x,  self.state_msg.gt_ego.y,  self.state_msg.gt_ego.w]
        self.vpr_ego            = [self.state_msg.vpr_ego.x, self.state_msg.vpr_ego.y, self.state_msg.vpr_ego.w]
        self.err_ego            = [self.gt_ego[0] - self.vpr_ego[0], self.gt_ego[1] - self.vpr_ego[1], angle_wrap(self.gt_ego[2] - self.vpr_ego[2], 'RAD')]

        self.gt_ind             = self.state_msg.truth_index
        self.vpr_ind            = self.state_msg.match_index
        self.err_ind            = abs(self.gt_ind - self.vpr_ind)

        self.in_gt_tolerance    = self.state_msg.gt_class
        self.gt_error           = self.state_msg.gt_error
        self.in_svm_tolerance   = self.state_msg.svm_class

        self.svm_prob           = self.state_msg.svm_prob
        self.svm_zvalue         = self.state_msg.svm_z

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
                        #param.revert()
                        pass
                except IndexError:
                    pass
                except:
                    self.print(formatException(), LogType.ERROR)
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)
        self.parameters_ready = True

    def main(self):
        # Main loop process
        self.set_state(NodeState.MAIN)

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

    def make_control_visualisation(self):
        if not np.sqrt(np.sum(np.square(np.array(self.gt_ego) - np.array(self.last_ego)))) > 0.02:
            return
        
        self.last_ego           = self.gt_ego
        
        if self.SVM_MODE.get():
            if self.in_svm_tolerance:
                colour = self.colour_svm_good
            else:
                colour = self.colour_bad
            scale = abs(self.svm_zvalue)
        else:
            if self.in_gt_tolerance:
                colour = self.colour_good
                scale  = 0.2
            elif self.gt_error < 2:
                colour = self.colour_bad
                scale = self.gt_error + 0.01
            else:
                colour = self.colour_lost
                scale = 2

        new_marker                      = Marker()
        new_marker.header               = Header(stamp=rospy.Time.now(), frame_id='map')
        new_marker.type                 = new_marker.SPHERE
        new_marker.action               = new_marker.ADD
        new_marker.id                   = self.marker_id
        new_marker.color                = colour
        new_marker.scale                = Vector3(x=scale, y=scale, z=0.01)

        new_marker.pose.position        = Point(x=self.gt_ego[0], y=self.gt_ego[1])
        new_marker.pose.orientation     = q_from_yaw(self.gt_ego[2])

        assert not self.markers.markers is None
        self.markers.markers.append(new_marker)
        self.marker_id = (self.marker_id + 1) % self.NUM_MARKERS.get()
        while len(self.markers.markers) > self.NUM_MARKERS.get():
            self.markers.markers.pop(0)

        self.confidence_pub.publish(self.markers)

    def make_display_feed(self):
        if self.in_gt_tolerance:
            tol_icon = self.good_icon
        else:
            tol_icon = self.poor_icon

        if self.in_svm_tolerance:
            svm_icon = self.good_icon
        else:
            svm_icon = self.poor_icon

        self.tol_hist           = np.roll(self.tol_hist, -1, 0)
        self.tol_hist[-1, :]    = 0
        # 0: TN - No Gt No SVM
        # 1: FN - Yes Gt No SVM
        # 2: FP - No Gt Yes SVM
        # 3: TP - Yes Gt Yes SVM
        self.tol_hist[-1, self.in_gt_tolerance + self.in_svm_tolerance * 2] = 1
        col_sum                 = np.sum(self.tol_hist[:,0:4], 0)
        all_sum                 = np.sum(col_sum)
        self.tol_hist[-1, 4:]   = col_sum / all_sum

        feed_piece      = np.zeros((250, 250, 3), dtype=np.uint8)
        icon_piece      = np.zeros((250, 250, 3), dtype=np.uint8)
        apply_icon(icon_piece, position=(180,120), icon=svm_icon, make_copy=False)
        apply_icon(icon_piece, position=(180,180), icon=tol_icon, make_copy=False)

        label_image(icon_piece, 'SVM:', (100,120+35), colour=(255,255,255), border=None, make_copy=False)
        label_image(icon_piece, 'TOL:', (100,180+35), colour=(255,255,255), border=None, make_copy=False)

        label_image(icon_piece, 'REF',  (10,30),      colour=(100,255,100), border=None, make_copy=False, scale=0.8)
        label_image(icon_piece, 'QRY',  (10,70),      colour=(100,255,100), border=None, make_copy=False, scale=0.8)
        label_image(icon_piece, 'TRU',  (70,30),      colour=(100,255,100), border=None, make_copy=False, scale=0.8)
        cv2.line(icon_piece, ( 65,  10), ( 65,  80), (200,100,100), 2)
        cv2.line(icon_piece, ( 10,  42), (120,  42), (200,100,100), 2)

        ref_match_raw   = self.ip.dataset['dataset'][enum_name(self.FEAT_TYPE.get())][self.vpr_ind]
        ref_true_raw    = self.ip.dataset['dataset'][enum_name(self.FEAT_TYPE.get())][self.gt_ind]
        qry_raw         = compressed2np(self.state_msg.query_image)
        ref_match       = convert_img_to_uint8(ref_match_raw,   resize=(250,250), dstack=(not len(ref_match_raw.shape) == 3))
        ref_true        = convert_img_to_uint8(ref_true_raw,    resize=(250,250), dstack=(not len(ref_true_raw.shape) == 3))
        qry             = convert_img_to_uint8(qry_raw,         resize=(250,250), dstack=(not len(qry_raw.shape) == 3))
        
        fig, ax = plt.subplots(figsize=(5,2.5))
        ax.plot(self.tol_hist[:, 4], 'r', linewidth=3, label='TN: % 4d%%' % int(100*self.tol_hist[-1, 4]))
        ax.plot(self.tol_hist[:, 5], 'y', linewidth=3, label='FN: % 4d%%' % int(100*self.tol_hist[-1, 5]))
        ax.plot(self.tol_hist[:, 6], 'b', linewidth=3, label='FP: % 4d%%' % int(100*self.tol_hist[-1, 6]))
        ax.plot(self.tol_hist[:, 7], 'g', linewidth=3, label='TP: % 4d%%' % int(100*self.tol_hist[-1, 7]))
        ax.get_xaxis().set_visible(False)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        L = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        plt.setp(L.texts, family='monospace')
        fig.canvas.draw()

        plot_raw_flat   = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_raw_pad    = plot_raw_flat.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_raw        = plot_raw_pad[20:-20, 20:]
        plot            = convert_img_to_uint8(plot_raw, resize=(500, 250), dstack=(not len(qry_raw.shape) == 3))
        plt.close('all') # close matplotlib

        feed            = np.concatenate([  np.concatenate([feed_piece, feed_piece, ref_match,  ref_true    ], axis=1),
                                            np.concatenate([plot,                   qry,        icon_piece  ], axis=1)   ])
        
        cv2.line(feed, (500,  0), (500,500), (200,100,100), 2)
        cv2.line(feed, (750,  0), (750,500), (200,100,100), 2)
        cv2.line(feed, (500,250), (999,250), (200,100,100), 2)

        label_image(feed, 'Error Index:', ( 10, 35), colour=(255,255,255), border=None, make_copy=False)
        label_image(feed, ' %4d'         % self.err_ind,      (250,  35), colour=(255,255,255), border=None, make_copy=False)

        label_image(feed, 'Error X:',     ( 10, 70), colour=(255,255,255), border=None, make_copy=False)
        label_image(feed, '% 6.2f [m]'   % (self.err_ego[0]), (250,  70), colour=(255,255,255), border=None, make_copy=False)

        label_image(feed, 'Error Y:',     ( 10, 105), colour=(255,255,255), border=None, make_copy=False)
        label_image(feed, '% 6.2f [m]'   % (self.err_ego[1]), (250, 105), colour=(255,255,255), border=None, make_copy=False)

        label_image(feed, 'Error W:',     ( 10, 140), colour=(255,255,255), border=None, make_copy=False)
        label_image(feed, '% 6.2f [rad]' % (self.err_ego[2]), (250, 140), colour=(255,255,255), border=None, make_copy=False)

        label_image(feed, 'SVM Z:',       ( 10, 175), colour=(255,255,255), border=None, make_copy=False)
        label_image(feed, '% 6.2f'       % (self.svm_zvalue), (250, 175), colour=(255,255,255), border=None, make_copy=False)

        ros_msg                 = np2compressed(feed)
        ros_msg.header.stamp    = rospy.Time.now()
        ros_msg.header.frame_id = 'map'

        self.display_pub.publish(ros_msg)

    def loop_contents(self):
        if not (self.new_state_msg):
            self.print("Waiting.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return # denest
        self.rate_obj.sleep()
        self.new_state_msg = False

        self.make_control_visualisation()
        self.make_display_feed()

def do_args():
    parser = ap.ArgumentParser(prog="visualiser.py", 
                            description="ROS Visualiser Node",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='visualiser')

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