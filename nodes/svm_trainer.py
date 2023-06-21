#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from rospy_message_converter import message_converter
from std_msgs.msg import String

from pyaarapsi.core.argparse_tools              import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools                   import Base_ROS_Class, NodeState, roslogger, LogType, set_rospy_log_lvl
from pyaarapsi.core.helper_tools                import formatException
from pyaarapsi.vpr_simple.svm_model_tool        import SVMModelProcessor

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
    def __init__(self, node_name, rate_num, namespace, anon, log_level, reset, order_id=0):
        super().__init__(node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30)

        self.init_params(rate_num, log_level, reset)
        self.init_vars()
        self.init_rospy()

        self.generate_new_svm(self.make_svm_model_params())
        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_rospy(self):
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())
        self.param_sub              = rospy.Subscriber(self.namespace + "/params_update", String, self.param_callback, queue_size=100)
        self.svm_request_sub        = rospy.Subscriber(self.namespace + '/requests/svm/request', RequestSVM, self.svm_request_callback, queue_size=1)
        self.dataset_request_sub    = rospy.Subscriber(self.namespace + '/requests/dataset/ready', ResponseDataset, self.dataset_request_callback, queue_size=1)
        self.svm_request_pub        = self.add_pub(self.namespace + '/requests/svm/ready', ResponseSVM, queue_size=1)
        self.dataset_request_pub    = self.add_pub(self.namespace + "/requests/dataset/request", RequestDataset, queue_size=1)
        
    def init_vars(self):
        # Set up SVM
        self.svm                    = SVMModelProcessor(ros=True)
        self.svm_queue              = []
        self.dataset_queue          = []
        
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

    def param_callback(self, msg):
        if self.params.exists(msg.data):
            self.print("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG)
            self.params.update(msg.data)

            if msg.data == self.LOG_LEVEL.name:
                set_rospy_log_lvl(self.LOG_LEVEL.get())
            elif msg.data == self.RATE_NUM.name:
                self.rate_obj = rospy.Rate(self.RATE_NUM.get())
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)

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
        self.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            self.loop_contents()
    
    def loop_contents(self):
        self.update_SVM()

def do_args():
    parser = ap.ArgumentParser(prog="svm_trainer.py", 
                            description="ROS SVM Trainer Node",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")

    # Optional Arguments:
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="svm_trainer",    help="Set node name (default: %(default)s).")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=2.0,              help='Set node rate (default: %(default)s).')
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=False,            help="Set whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Set ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Set ROS log level (default: %(default)s).")
    parser.add_argument('--reset',            '-R',  type=check_bool,                   default=False,            help='Force reset of parameters to specified ones (default: %(default)s)')
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')

    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])
    return args

if __name__ == '__main__':
    args        = do_args()
    try:
        nmrc = Main_ROS_Class(args['node_name'], args['rate'], args['namespace'], args['anon'], args['log_level'], args['reset'], order_id=args['order_id'])
        nmrc.print("Initialisation complete. Listening for queries...", LogType.INFO)
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO, ros=False) # False as rosnode likely terminated
        sys.exit()
    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)