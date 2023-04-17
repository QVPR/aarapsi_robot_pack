#!/usr/bin/env python3

import rospy
import cv2
import sys
import argparse as ap
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from pyaarapsi.core.argparse_tools import check_positive_float, check_positive_two_int_tuple, check_bool, check_string
from pyaarapsi.core.ros_tools import imgmsgtrans, Heartbeat, NodeState, roslogger, LogType

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
    def __init__(self, topic_in, topic_out, namespace, exts, types, rate, hist_len=3, transform=None):

        self.topic_in   = topic_in
        self.topic_out  = topic_out
        self.namespace  = namespace
        self.exts       = exts
        self.types      = types
        self.hist_len   = hist_len
        self.transform  = transform
        if transform is None:
            self.transform = self.empty
        self.subs       = [rospy.Subscriber(self.topic_in + self.exts[i], self.types[i], self.cb, queue_size=1) \
                           for i in range(len(self.exts))]
        self.pubs       = [rospy.Publisher(self.namespace + self.topic_out + self.exts[i], self.types[i], queue_size=1) \
                           for i in range(len(self.exts))]
        self.timer      = rospy.Timer(rospy.Duration(1/rate), self.timer)
        self.msgs       = [[] for i in types]

        exts_string     = ''.join(self.exts)
        rospy.loginfo("Throttling %s[%s] to %s[%s] at %0.2f Hz" % (topic_in, exts_string, topic_out, exts_string, rate))

    def cb(self, msg):
        index = self.types.index(type(msg))
        self.msgs[index].append(msg)
        while len(self.msgs[index]) > self.hist_len:
            self.msgs[index].pop(0)

    def empty(self, msg):
    # dummy transform function
        return msg

    def timer(self, event):
        for i in range(len(self.pubs)):
            if len(self.msgs[i]) > 0:
                msg_to_pub = self.msgs[i].pop(0)
                self.pubs[i].publish(self.transform(msg_to_pub))

class mrc:
    def __init__(self, node_name, rate, namespace, anon, mode, resize_dims, log_level):
        
        self.node_name      = node_name
        self.namespace      = namespace
        self.anon           = anon
        self.mode           = mode
        self.resize_dims    = resize_dims
        self.log_level      = log_level
        self.rate_num       = rate

        rospy.init_node(self.node_name, anonymous=self.anon, log_level=self.log_level)
        roslogger('Starting %s node.' % (self.node_name), LogType.INFO, ros=True)
        self.rate_obj       = rospy.Rate(self.rate_num)
        self.heartbeat      = Heartbeat(self.node_name, self.namespace, NodeState.INIT, self.rate_num)

        # set up topics
        c_in, c_out = self.get_cam_topics()
        if mode == 0:
            exts            = ['']
            types           = [Image]
        elif mode == 1:
            exts            = ['/compressed']
            types           = [CompressedImage]
        elif mode == 2:
            exts            = ['', '/compressed']
            types           = [Image, CompressedImage]
        else:
            raise Exception('Unknown mode')
        
        self.bridge         = CvBridge()

        # Set up throttles:
        self.throttles      = [Throttle_Topic(c_in[i], c_out[i], namespace, exts, types, rate, \
                                    transform=lambda x: self.img_resize(x, mode="rectangle")) \
                                if 'stitched' in c_in[i] else \
                               Throttle_Topic(c_in[i], c_out[i], namespace, exts, types, rate, \
                                    transform=lambda x: self.img_resize(x, mode="square")) \
                                for i in range(len(c_in))]
        
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
        self.heartbeat.set_state(NodeState.MAIN)

        # loop forever until signal shutdown
        while not rospy.is_shutdown():
            self.rate_obj.sleep()

if __name__ == '__main__':
    
    parser = ap.ArgumentParser(prog="throttle.py", 
                                description="ROS Topic Throttle Tool",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--rate',      '-r', type=check_positive_float,         default=10.0,           help='Set node rate (default: %(default)s).')
    parser.add_argument('--img-dims',  '-i', type=check_positive_two_int_tuple, default=(64,64),        help='Set image dimensions (default: %(default)s).')
    parser.add_argument('--node-name', '-N', type=check_string,                 default="throttle",     help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon',      '-a', type=check_bool,                   default=True,           help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--namespace', '-n', type=check_string,                 default="/vpr_nodes",   help="Specify ROS namespace (default: %(default)s).")
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16],    default=2,              help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--mode',      '-m', type=int, choices=[0,1,2],         default=2,              help="Specify whether to throttle raw (0), compressed (1), or both (2) topics (default: %(default)s).")

    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])

    node_name   = args['node_name']
    rate        = args['rate']
    namespace   = args['namespace']
    anon        = args['anon']
    mode        = args['mode']
    resize_dims = args['img_dims']
    log_level   = args['log_level']

    try:
        nmrc = mrc(node_name, rate, namespace, anon, mode, resize_dims, log_level)
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO, ros=True)
        sys.exit()
    except:
        roslogger("Error state reached, system exit triggered.", LogType.INFO, ros=True)

    
