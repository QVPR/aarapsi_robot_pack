#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
import copy

from rospy_message_converter import message_converter
from pyaarapsi.core.argparse_tools          import check_bool
from pyaarapsi.core.ros_tools               import NodeState, roslogger, LogType, compressed2np
from pyaarapsi.core.helper_tools            import formatException, np_ndarray_to_uint8_list, Timer
from pyaarapsi.core.enum_tools              import enum_get
from pyaarapsi.vpr_classes.base             import Base_ROS_Class, base_optional_args

from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool  import VPRDatasetProcessor

from aarapsi_robot_pack.msg import RequestDataset, ResponseDataset
from aarapsi_robot_pack.srv import DoExtraction, DoExtractionResponse, DoExtractionRequest

'''

ROS Dataset Trainer Node

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
        self.init_vars(kwargs['use_gpu'])
        self.init_rospy()

        self.node_ready(kwargs['order_id'])
        
    def init_vars(self, use_gpu):
        super().init_vars()
        # Process reference data
        self.use_gpu            = use_gpu
        dataset_dict            = self.make_dataset_dict()
        try:
            self.vpr            = VPRDatasetProcessor(dataset_params=None, try_gen=True, cuda=use_gpu, \
                                        autosave=True, use_tqdm=True, ros=True)
            if use_gpu:
                self.vpr.init_nns()

            self.vpr.load_dataset(dataset_params=dataset_dict, try_gen=True)
                
        except Exception as e:
            self.print(formatException(), LogType.ERROR)
            self.print(formatException(dump=True), LogType.DEBUG)
            self.exit()
        self.dataset_queue      = []

    def init_rospy(self):
        super().init_rospy()
        
        self.srv_extraction         = rospy.Service(self.namespace + '/do_extraction', DoExtraction, self.handle_do_extraction)
        self.dataset_request_sub    = rospy.Subscriber(self.namespace + '/requests/dataset/request', RequestDataset, self.dataset_request_callback, queue_size=10)
        self.dataset_request_pub    = self.add_pub(self.namespace + '/requests/dataset/ready', ResponseDataset, queue_size=1)
        
    def dataset_request_callback(self, msg):
        self.print("New parameters received.")
        self.dataset_queue.append(msg)

    def update_VPR(self):
        if not len(self.dataset_queue):
            return
        params_for_swap = self.dataset_queue.pop(-1)
        if not self.vpr.swap(message_converter.convert_ros_message_to_dictionary(params_for_swap), generate=True, allow_false=True):
            self.print("Dataset generation failed.", LogType.WARN, throttle=5)
            self.dataset_request_pub.publish(ResponseDataset(params=params_for_swap, success=False))
        else:
            self.print("Dataset generated.")
            self.dataset_request_pub.publish(ResponseDataset(params=params_for_swap, success=True))

    def handle_do_extraction(self, req: DoExtractionRequest):
        ans                 = DoExtractionResponse()
        success             = True
        try:
            query           = compressed2np(req.input)
            feat_type       = enum_get(req.feat_type, FeatureType)
            assert not feat_type is None
            img_dims        = list(req.img_dims)
            ft_qry          = self.vpr.getFeat(query, feat_type, use_tqdm=False, dims=img_dims)
            ans.output      = np_ndarray_to_uint8_list(ft_qry)
            ans.vector_dims = list(ft_qry.shape)
        except:
            self.print(formatException(), LogType.ERROR)
            success         = False

        ans.success         = success
        self.print("Service requested, Success=%s" % (str(success)), LogType.DEBUG)
        return ans

    
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
        self.update_VPR()

def do_args():
    parser = ap.ArgumentParser(prog="dataset_trainer.py", 
                            description="ROS Dataset Trainer Node",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")

    # Optional Arguments:
    parser = base_optional_args(parser, node_name='dataset_trainer', rate=10.0)
    parser.add_argument('--use-gpu', '-G', type=check_bool, default=True, help='Specify whether to use GPU (default: %(default)s).')

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