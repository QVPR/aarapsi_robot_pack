#!/usr/bin/env python3

import rospy
import cv2
import sys
import argparse as ap
from cv_bridge import CvBridge
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, CompressedImage
from pyaarapsi.core.argparse_tools import check_positive_float, check_positive_two_int_tuple, check_bool, check_string, check_positive_int
from pyaarapsi.core.ros_tools import imgmsgtrans, NodeState, roslogger, LogType, set_rospy_log_lvl, init_node
from pyaarapsi.core.helper_tools import formatException

'''
ROS Throttle Tool

Performs a rate (and optionally, data) throttling on topics
Primarily defined to operate on sensor_msgs/(Compressed)Image topics
'''

class Throttle_Topic:
    '''
    Throttle Topic Class

    Purpose:
    - Wrap subscribers, publishers, and timer into single class 
    - Ease of implementation
    '''
    def __init__(self, topic_in, topic_out, namespace, exts, types, rate, ROS_HOME, hist_len=3, transform=None, printer=rospy.loginfo):

        self.topic_in   = topic_in
        self.topic_out  = topic_out
        self.namespace  = namespace
        self.exts       = exts
        self.types      = types
        self.hist_len   = hist_len
        self.transform  = transform
        self.ROS_HOME   = ROS_HOME
        if transform is None:
            self.transform = self.empty
        self.subs       = [rospy.Subscriber(self.topic_in + self.exts[i], self.types[i], self.cb, queue_size=1) \
                           for i in range(len(self.exts))]
        self.pubs       = [self.ROS_HOME.add_pub(self.namespace + self.topic_out + self.exts[i], self.types[i], queue_size=1) \
                           for i in range(len(self.exts))]
        self.timer_obj  = rospy.Timer(rospy.Duration(1/rate), self.timer_cb)
        self.msgs       = [[] for i in types]

        exts_string     = ''.join(self.exts)
        printer("Throttling %s[%s] to %s[%s] at %0.2f Hz" % (topic_in, exts_string, self.namespace + topic_out, exts_string, rate))

    def cb(self, msg):
        index = self.types.index(type(msg))
        self.msgs[index].append(msg)
        while len(self.msgs[index]) > self.hist_len:
            self.msgs[index].pop(0)

    def empty(self, msg):
    # dummy transform function
        return msg

    def timer_cb(self, event):
        for i in range(len(self.pubs)):
            if len(self.msgs[i]) > 0:
                msg_to_pub = self.msgs[i].pop(0)
                self.pubs[i].publish(self.transform(msg_to_pub))

class mrc:
    def __init__(self, node_name, rate_num, namespace, anon, mode, resize_dims, log_level, reset=True, order_id=0):
        
        if not init_node(self, node_name, namespace, rate_num, anon, log_level, order_id=order_id, throttle=30):
            sys.exit()

        self.init_params(rate_num, log_level, reset)
        self.init_vars(mode, resize_dims)
        self.init_rospy()

        rospy.set_param(self.namespace + '/launch_step', order_id + 1)

    def init_params(self, rate_num, log_level, reset):
        self.RATE_NUM       = self.ROS_HOME.params.add(self.nodespace + "/rate",      rate_num,  check_positive_float,   force=reset)
        self.LOG_LEVEL      = self.ROS_HOME.params.add(self.nodespace + "/log_level", log_level, check_positive_int,     force=reset)

    def init_vars(self, mode, resize_dims):
        self.mode           = mode
        self.resize_dims    = resize_dims

        # set up topics
        self.cin, self.cout = self.get_cam_topics()
        if mode == 0:
            self.exts            = ['']
            self.types           = [Image]
        elif mode == 1:
            self.exts            = ['/compressed']
            self.types           = [CompressedImage]
        elif mode == 2:
            self.exts            = ['', '/compressed']
            self.types           = [Image, CompressedImage]
        else:
            raise Exception('Unknown mode')
        
        self.bridge         = CvBridge()

    def init_rospy(self):
        self.rate_obj       = rospy.Rate(self.RATE_NUM.get())
        self.param_sub      = rospy.Subscriber(self.namespace + "/params_update", String, self.param_cb, queue_size=100)

        # Set up throttles:
        self.throttles      = [Throttle_Topic(self.cin[i], self.cout[i], self.namespace, self.exts, self.types, self.RATE_NUM.get(), self.ROS_HOME, \
                                    transform=lambda x: self.img_resize(x, mode="rectangle"), printer=self.print) \
                                if 'stitched' in self.cin[i] else \
                               Throttle_Topic(self.cin[i], self.cout[i], self.namespace, self.exts, self.types, self.RATE_NUM.get(), self.ROS_HOME, \
                                    transform=lambda x: self.img_resize(x, mode="square"), printer=self.print) \
                                for i in range(len(self.cin))]

    def param_cb(self, msg):
        if self.ROS_HOME.params.exists(msg.data):
            self.print("Change to parameter [%s]; logged." % msg.data, LogType.DEBUG)
            self.ROS_HOME.params.update(msg.data)

            if msg.data == self.LOG_LEVEL.name:
                set_rospy_log_lvl(self.LOG_LEVEL.get())
            elif msg.data == self.RATE_NUM.name:
                self.rate_obj = rospy.Rate(self.RATE_NUM.get())
        else:
            self.print("Change to untracked parameter [%s]; ignored." % msg.data, LogType.DEBUG)
        
    def get_cam_topics(self):
        # Helper function to abstract topic generation
        ns_in               = '/ros_indigosdk_occam'
        ns_out              = '/occam'
        topics              = ['/image0', '/image1', '/image2', '/image3', '/image4', '/stitched_image0']
        throttles_in        = []
        throttles_out       = []
        for topic in topics:
            throttles_in.append(ns_in + topic)
            throttles_out.append(ns_out + topic)
        return throttles_in, throttles_out
    
    def img_resize(self, msg, frame_id="occam", mode="square"):
        if mode == "square":
            def transform(img, resize_dims):
                return cv2.resize(img, resize_dims, interpolation=cv2.INTER_AREA)
        elif mode == "rectangle":
            def transform(img, resize_dims):
                img_size    = img.shape
                resize_dims = (round((img_size[1]/img_size[0])*resize_dims[0]), resize_dims[0])
                return cv2.resize(img, resize_dims, interpolation=cv2.INTER_AREA)
        else:
            raise Exception("Undefined mode %s, should be either 'square' or 'rectangle'." % mode)
        rmsg                = imgmsgtrans(msg, lambda img: transform(img, self.resize_dims), bridge=self.bridge)
        rmsg.header         = Header(stamp=rospy.Time.now(), frame_id=frame_id, seq=msg.header.seq)
        return rmsg
    
    def main(self):
        self.ROS_HOME.set_state(NodeState.MAIN)

        # loop forever until signal shutdown
        while not rospy.is_shutdown():
            self.rate_obj.sleep()

    def print(self, text, logtype=LogType.INFO, throttle=0, ros=None, name=None, no_stamp=None):
        if ros is None:
            ros = self.ROS_HOME.logros
        if name is None:
            name = self.ROS_HOME.node_name
        if no_stamp is None:
            no_stamp = self.ROS_HOME.logstamp
        roslogger(text, logtype, throttle=throttle, ros=ros, name=name, no_stamp=no_stamp)

def do_args():
    parser = ap.ArgumentParser(prog="throttle.py", 
                                description="ROS Topic Throttle Tool",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--rate',             '-r',  type=check_positive_float,         default=10.0,             help='Set node rate (default: %(default)s).')
    parser.add_argument('--img-dims',         '-i',  type=check_positive_two_int_tuple, default=(64,64),          help='Set image dimensions (default: %(default)s).')
    parser.add_argument('--node-name',        '-N',  type=check_string,                 default="throttle",       help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon',             '-a',  type=check_bool,                   default=True,             help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace',        '-n',  type=check_string,                 default="/vpr_nodes",     help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level',        '-V',  type=int, choices=[1,2,4,8,16],    default=2,                help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--mode',             '-m',  type=int, choices=[0,1,2],         default=2,                help="Specify whether to throttle raw (0), compressed (1), or both (2) topics (default: %(default)s).")
    parser.add_argument('--order-id',         '-ID', type=int,                          default=0,                help='Specify boot order of pipeline nodes (default: %(default)s).')
    
    raw_args = parser.parse_known_args()
    return vars(raw_args[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = mrc(args['node_name'], args['rate'], args['namespace'], args['anon'], args['mode'], \
                   args['img_dims'], args['log_level'], order_id=args['order_id'])
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

    
