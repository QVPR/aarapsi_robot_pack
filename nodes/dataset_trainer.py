#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from rospy_message_converter import message_converter
from std_msgs.msg import String
from pyaarapsi.core.argparse_tools import check_positive_float, check_positive_int, check_bool, check_string, check_positive_two_int_list, check_enum
from pyaarapsi.core.ros_tools import NodeState, roslogger, LogType, set_rospy_log_lvl, init_node
from pyaarapsi.core.helper_tools import formatException, np_ndarray_to_uint8_list, uint8_list_to_np_ndarray
from pyaarapsi.core.enum_tools import enum_name, enum_get

from pyaarapsi.vpr_simple.vpr_helpers            import FeatureType
from pyaarapsi.vpr_simple.vpr_dataset_tool       import VPRDatasetProcessor

from aarapsi_robot_pack.msg import RequestDataset, ResponseDataset
from aarapsi_robot_pack.srv import DoExtraction, DoExtractionResponse

import torch, gc

'''

ROS Dataset Trainer Node

This node subscribes to the same parameter updates as the vpr_monitor node.
It performs the same logic, however it triggers training when no model exists
that matches the required parameters. This enables training to be performed
in a separate thread to the monitor, allowing for fewer interruptions to the
monitor's performance and activities.

'''

class mrc():
    def __init__(self, node_name, rate_num, namespace, anon, log_level, reset, use_gpu=True, order_id=0):

        if not init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30):
            sys.exit()

        self.init_params(log_level, rate_num, reset)
        self.init_vars(use_gpu)
        self.init_rospy()

        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_rospy(self):
        self.rate_obj               = rospy.Rate(self.RATE_NUM.get())
        self.param_sub              = rospy.Subscriber(self.namespace + "/params_update", String, self.param_callback, queue_size=100)
        self.srv_extraction         = rospy.Service(self.namespace + '/do_extraction', DoExtraction, self.handle_do_extraction)
        self.dataset_request_sub    = rospy.Subscriber(self.namespace + '/requests/dataset/request', RequestDataset, self.dataset_request_callback, queue_size=10)
        self.dataset_request_pub    = self.ROS_HOME.add_pub(self.namespace + '/requests/dataset/ready', ResponseDataset, queue_size=1)

    def init_params(self, rate_num, log_level, reset):
        self.FEAT_TYPE              = self.ROS_HOME.params.add(self.namespace + "/feature_type",        None,                   lambda x: check_enum(x, FeatureType),           force=False)
        self.IMG_DIMS               = self.ROS_HOME.params.add(self.namespace + "/img_dims",            None,                   check_positive_two_int_list,                    force=False)
        self.NPZ_DBP                = self.ROS_HOME.params.add(self.namespace + "/npz_dbp",             None,                   check_string,                                   force=False)
        self.BAG_DBP                = self.ROS_HOME.params.add(self.namespace + "/bag_dbp",             None,                   check_string,                                   force=False)
        self.IMG_TOPIC              = self.ROS_HOME.params.add(self.namespace + "/img_topic",           None,                   check_string,                                   force=False)
        self.ODOM_TOPIC             = self.ROS_HOME.params.add(self.namespace + "/odom_topic",          None,                   check_string,                                   force=False)
        
        self.REF_BAG_NAME           = self.ROS_HOME.params.add(self.namespace + "/ref/bag_name",        None,                   check_string,                                   force=False)
        self.REF_FILTERS            = self.ROS_HOME.params.add(self.namespace + "/ref/filters",         None,                   check_string,                                   force=False)
        self.REF_SAMPLE_RATE        = self.ROS_HOME.params.add(self.namespace + "/ref/sample_rate",     None,                   check_positive_float,                           force=False) # Hz
        
        self.RATE_NUM               = self.ROS_HOME.params.add(self.nodespace + "/rate",                rate_num,               check_positive_float,                           force=reset)
        self.LOG_LEVEL              = self.ROS_HOME.params.add(self.nodespace + "/log_level",           log_level,              check_positive_int,                             force=reset)
        
    def init_vars(self, use_gpu):
        # Process reference data
        self.use_gpu            = use_gpu
        self.vpr                = VPRDatasetProcessor(self.make_dataset_dict(), try_gen=True, init_hybridnet=use_gpu, init_netvlad=use_gpu, cuda=use_gpu, \
                                                      autosave=True, use_tqdm=True, ros=True)
        self.dataset_queue      = []
        
    def dataset_request_callback(self, msg):
        self.print("New parameters received.")
        self.dataset_queue.append(msg)

    def make_dataset_dict(self):
        return dict(bag_name=self.REF_BAG_NAME.get(), npz_dbp=self.NPZ_DBP.get(), bag_dbp=self.BAG_DBP.get(), \
                    odom_topic=self.ODOM_TOPIC.get(), img_topics=[self.IMG_TOPIC.get()], sample_rate=self.REF_SAMPLE_RATE.get(), \
                    ft_types=enum_name(self.FEAT_TYPE.get(),wrap=True), img_dims=self.IMG_DIMS.get(), filters='{}')

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

    def handle_do_extraction(self, req):
        ans                 = DoExtractionResponse()
        success             = True
        try:
            query           = uint8_list_to_np_ndarray(req.input)
            feat_type       = enum_get(req.feat_type, FeatureType)
            img_dims        = req.img_dims
            ft_qry          = self.vpr.getFeat(query, feat_type, use_tqdm=False, dims=img_dims)
            ans.output      = np_ndarray_to_uint8_list(ft_qry)
            ans.vector_dims = list(ft_qry.shape)
        except:
            success         = False

        ans.success         = success
        self.print("Service requested, Success=%s" % (str(success)), LogType.DEBUG)
        return ans

    def debug_cb(self, msg):
        if msg.node_name == self.node_name:
            if msg.instruction == 0:
                self.print(self.make_dataset_dict(), LogType.DEBUG)
            elif msg.instruction == 1:
                count = 0
                for obj in gc.get_objects():
                    try:
                        if isinstance(obj, torch.nn.modules.module.Module):
                            print(type(obj), obj.size())
                        else:
                            count +=1
                    except:
                        count += 1
                print("skipped %s objects." % str(count))
            else:
                self.print(msg.instruction, LogType.DEBUG)

    def param_callback(self, msg):
        if self.ROS_HOME.params.exists(msg.data):
            self.print("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG)
            self.ROS_HOME.params.update(msg.data)

            if msg.data == self.LOG_LEVEL.name:
                set_rospy_log_lvl(self.LOG_LEVEL.get())
            elif msg.data == self.RATE_NUM.name:
                self.rate_obj = rospy.Rate(self.RATE_NUM.get())
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)

    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            self.loop_contents()
    
    def loop_contents(self):
        self.update_VPR()
        
    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

def do_args():
    parser = ap.ArgumentParser(prog="dataset_trainer.py", 
                            description="ROS Dataset Trainer Node",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")

    # Optional Arguments:
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="dataset_trainer",help="Set node name (default: %(default)s).")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=2.0,              help='Set node rate (default: %(default)s).')
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=False,            help="Set whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Set ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Set ROS log level (default: %(default)s).")
    parser.add_argument('--reset',            '-R',  type=check_bool,                   default=False,            help='Force reset of parameters to specified ones (default: %(default)s).')
    parser.add_argument('--use-gpu',          '-G',  type=check_bool,                   default=True,             help='Specify whether to use GPU (default: %(default)s).')
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')

    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])
    return args

if __name__ == '__main__':
    args        = do_args()
    try:
        nmrc = mrc(args['node_name'], args['rate'], args['namespace'], args['anon'], args['log_level'], args['reset'], use_gpu=args['use_gpu'], order_id=args['order_id'])
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