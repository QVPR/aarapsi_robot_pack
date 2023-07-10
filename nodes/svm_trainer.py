#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from rospy_message_converter import message_converter

from pyaarapsi.core.ros_tools                   import NodeState, roslogger, LogType
from pyaarapsi.core.helper_tools                import formatException, vis_dict
from pyaarapsi.vpr_simple.svm_model_tool        import SVMModelProcessor
from pyaarapsi.vpr_classes.base                 import Base_ROS_Class, base_optional_args

from aarapsi_robot_pack.msg import ResponseSVM, RequestSVM, ResponseDataset, RequestDataset

'''

ROS SVM Trainer Node

This node subscribes to the same parameter updates as the vpr_monitor node.
It performs the same logic, however it triggers training when no model exists
that matches the required parameters. This enables training to be performed
in a separate thread to the monitor, allowing for fewer interruptions to the
monitor's performance and activities.

'''
class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.generate_new_svm(self.make_svm_model_params())
        self.node_ready(kwargs['order_id'])
        
    def init_vars(self):
        super().init_vars()
        # Set up SVM
        self.svm                    = SVMModelProcessor(ros=True)
        self.svm_queue              = []
        self.dataset_queue          = []

    def init_rospy(self):
        super().init_rospy()
        
        self.svm_request_sub        = rospy.Subscriber(self.namespace + '/requests/svm/request', RequestSVM, self.svm_request_callback, queue_size=1)
        self.dataset_request_sub    = rospy.Subscriber(self.namespace + '/requests/dataset/ready', ResponseDataset, self.dataset_request_callback, queue_size=1)
        self.svm_request_pub        = self.add_pub(self.namespace + '/requests/svm/ready', ResponseSVM, queue_size=1)
        self.dataset_request_pub    = self.add_pub(self.namespace + "/requests/dataset/request", RequestDataset, queue_size=1)
        
    def dataset_request_callback(self, msg):
        if msg.success == False:
            self.print('Dataset request processed, error. Parameters: %s' % str(msg.params), LogType.ERROR)
        try:
            index = self.dataset_queue.index(msg.params)
            self.print('Dataset request processed, success. Removing from dataset queue.')
            self.dataset_queue.pop(index)

        except ValueError:
            pass

    def svm_request_callback(self, msg):
        self.svm_queue.append(msg)

    def update_SVM(self):
        if not len(self.svm_queue):
            return
        params_for_swap = self.svm_queue.pop(-1)
        dict_params_for_swap = message_converter.convert_ros_message_to_dictionary(params_for_swap)
        self.generate_new_svm(dict_params_for_swap)
        if not self.svm.swap(dict_params_for_swap, generate=True, allow_false=True):
            self.print("Model generation failed.", LogType.WARN, throttle=5)
            self.svm_request_pub.publish(ResponseSVM(params=params_for_swap, success=False))
        else:
            self.print("SVM model generated.")
            self.svm_request_pub.publish(ResponseSVM(params=params_for_swap, success=True))

    def generate_new_svm(self, svm_model_params):
        if not self.svm.load_model(svm_model_params): # No model currently exists;
            output_statuses = self.svm.generate_model(**svm_model_params, save=True)
            if not all(output_statuses): # if the model failed to generate, datasets not ready, therefore...
                # ...check what failed, and queue these datasets to be built:
                if not output_statuses[0]: # qry set failed to generate:
                    dataset_msg = message_converter.convert_dictionary_to_ros_message('aarapsi_robot_pack/RequestDataset', svm_model_params['qry'])
                    self.dataset_queue.append(dataset_msg)
                if not output_statuses[1]: # ref set failed to generate:
                    dataset_msg = message_converter.convert_dictionary_to_ros_message('aarapsi_robot_pack/RequestDataset', svm_model_params['ref'])
                    self.dataset_queue.append(dataset_msg)

                self.dataset_request_pub.publish(self.dataset_queue[0])

                wait_intervals = 0
                while len(self.dataset_queue):
                    if rospy.is_shutdown():
                        sys.exit()
                    self.print('Waiting for SVM dataset construction...', throttle=5)
                    self.rate_obj.sleep()
                    wait_intervals += 1
                    if wait_intervals > 10 / (1/self.RATE_NUM.get()):
                        # Resend the oldest queue'd element every 10 seconds
                        try:
                            self.dataset_request_pub.publish(self.dataset_queue[0])
                        except:
                            pass
                        wait_intervals = 0

                if not all(self.svm.generate_model(**svm_model_params, save=True)):
                    raise Exception('Model generation failed even after new datasets were constructed!')

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
        self.rate_obj.sleep()
        self.update_SVM()

def do_args():
    parser = ap.ArgumentParser(prog="svm_trainer.py", 
                            description="ROS SVM Trainer Node",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")

    # Optional Arguments:
    parser = base_optional_args(parser, node_name='svm_trainer')

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