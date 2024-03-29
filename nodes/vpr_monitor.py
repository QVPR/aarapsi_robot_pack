#!/usr/bin/env python3

import rospy

import numpy as np
import argparse as ap
import sys

from rospy_message_converter import message_converter
from std_msgs.msg import String
from aarapsi_robot_pack.msg import RequestSVM, ResponseSVM, Label  # Our custom structures

from pyaarapsi.core.helper_tools            import formatException
from pyaarapsi.core.ros_tools               import roslogger, LogType, NodeState
from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor
from pyaarapsi.vpr_simple.svm_model_tool    import SVMModelProcessor
from pyaarapsi.vpr_classes.base             import Base_ROS_Class, base_optional_args

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()
        
        self.node_ready(kwargs['order_id'])
        
    def init_vars(self):
        super().init_vars()

        self.contour_msg            = None

        self.state_hist             = np.zeros((10,3)) # x, y, w
        self.state_size             = 0

        # Set up SVM
        self.svm                    = SVMModelProcessor(ros=True)
        _model = self.svm.load_model(self.make_svm_model_params())
        if not _model:
            raise Exception("Failed to find file with parameters matching: \n%s" % str(self.make_svm_model_params()))
        else:
            self.print("Model Ready (loaded: %s)." % str(_model))
        self.svm_requests           = []

        # flags to denest main loop:
        self.new_label              = False # new label received
        self.main_ready             = False # ensure pubs and subs don't go off early
        self.svm_swap_pending       = False
        ref_dataset_dict            = self.make_dataset_dict()
        try:
            self.ip                 = VPRDatasetProcessor(ref_dataset_dict, try_gen=False, ros=True)
        except:
            self.print(formatException(), LogType.ERROR)
            self.exit()

    def init_rospy(self):
        super().init_rospy()
        
        self.last_time              = rospy.Time.now()

        self.vpr_label_sub          = rospy.Subscriber( self.namespace + "/label",                  Label,          self.label_callback,        queue_size=1)
        self.svm_request_sub        = rospy.Subscriber( self.namespace + "/requests/svm/ready",     ResponseSVM,    self.svm_request_callback,  queue_size=1)
        self.svm_request_pub        = self.add_pub(     self.namespace + "/requests/svm/request",   RequestSVM,                                 queue_size=1)
        self.svm_state_pub          = self.add_pub(     self.namespace + "/state",                  Label,                                      queue_size=1)

    def update_SVM(self):
        svm_model_params       = self.make_svm_model_params()
        if not self.svm.swap(svm_model_params, generate=False, allow_false=True):
            self.print("Model swap failed. Previous model will be retained (Changed ROS parameter will revert)", LogType.WARN, throttle=5)
            self.svm_swap_pending   = True
            self.svm_requests.append(svm_model_params)
            svm_msg = message_converter.convert_dictionary_to_ros_message('aarapsi_robot_pack/RequestSVM', svm_model_params)
            self.svm_request_pub.publish(svm_msg)
            return False
        else:
            self.print("SVM model swapped.")
            self.svm_swap_pending = False
            return True

    def update_VPR(self):
        dataset_dict = self.make_dataset_dict()
        if not self.ip.swap(dataset_dict, generate=False, allow_false=True):
            self.print("VPR reference data swap failed. Previous set will be retained (changed ROS parameter will revert)", LogType.WARN)
            return False
        else:
            self.print("VPR reference data swapped.", LogType.INFO)
            return True

    def svm_request_callback(self, msg: ResponseSVM):
        pass

    def label_callback(self, msg: Label):
    # Store new label message and act as drop-in replacement for odom_callback + img_callback

        self.label            = msg

        self.state_hist       = np.roll(self.state_hist, 1, 0)
        self.state_hist[0,:]  = [msg.vpr_ego.x, msg.vpr_ego.y, msg.vpr_ego.w]
        self.state_size       = np.min([self.state_size + 1, self.state_hist.shape[0]])

        self.new_label        = True


    def param_helper(self, msg: String):
        svm_data_comp   = [i == msg.data for i in self.SVM_DATA_NAMES]
        try:
            param = np.array(self.SVM_DATA_PARAMS)[svm_data_comp][0]
            self.print("Change to SVM parameters detected.", LogType.WARN)
            if not self.update_SVM():
                param.revert()
        except IndexError:
            pass
        except:
            self.print(formatException(), LogType.DEBUG)

        ref_data_comp   = [i == msg.data for i in self.REF_DATA_NAMES]
        try:
            param = np.array(self.REF_DATA_PARAMS)[ref_data_comp][0]
            self.print("Change to VPR reference data parameters detected.", LogType.WARN)
            if not self.update_VPR():
                param.revert()
        except IndexError:
            pass
        except:
            self.print(formatException(), LogType.DEBUG)

    def publish_ros_info(self, svm_z, svm_prob, svm_class, svm_factors):
        # Populate and publish SVM State details

        time                        = rospy.Time.now()
        self.label.header.stamp     = time
        self.label.stamps.append(time) #type: ignore
        self.label.step             = self.label.MONITOR

        self.label.svm_prob	        = svm_prob # Continuous monitor state estimate 
        self.label.svm_z	        = svm_z # Monitor probability estimate
        self.label.svm_class        = svm_class# Binary monitor state estimate
        self.label.svm_factors      = list(svm_factors)

        self.svm_state_pub.publish(self.label)

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

    def loop_contents(self):

        if self.svm_swap_pending:
            self.update_SVM()

        if not (self.new_label and self.main_ready): # denest
            self.print("Waiting for a new label.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return
        nmrc.rate_obj.sleep()
        self.new_label = False

        # Predict model information, but have a try-except to catch if model is transitioning state
        predict_success = False
        pred, zvalues, factors, prob = None, None, None, None # Initialise variables,
        while not predict_success:
            if rospy.is_shutdown():
                self.exit()
            try:
                rXY = np.stack([self.ip.dataset['dataset']['px'], self.ip.dataset['dataset']['py']], 1)
                (pred, zvalues, [factors], prob) = self.svm.predict(np.array(self.label.distance_vector), self.label.match_index, rXY, init_pos=self.state_hist[1, 0:2])
                predict_success = True
            except:
                self.print("Predict failed. Trying again ...", LogType.WARN, throttle=1)
                self.print(formatException(), LogType.DEBUG, throttle=1)
                rospy.sleep(0.005)

        # Make ROS messages
        self.publish_ros_info(zvalues, prob, pred, factors)

def do_args():
    parser = ap.ArgumentParser(prog="vpr_monitor.py", 
                                description="ROS implementation of Helen Carson's Integrity Monitor, for integration with QVPR's VPR Primer",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='vpr_monitor')
    
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