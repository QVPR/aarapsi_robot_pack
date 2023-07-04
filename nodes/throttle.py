#!/usr/bin/env python3

import rospy
import sys
import argparse as ap
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, CompressedImage

from pyaarapsi.core.argparse_tools  import check_positive_float, check_positive_two_int_tuple, check_bool, check_string, check_positive_int
from pyaarapsi.core.ros_tools       import imgmsgtrans, NodeState, roslogger, LogType, set_rospy_log_lvl
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.enum_tools      import enum_value_options
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

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
    def __init__(self, topic_in, topic_out, namespace, exts, types, rate, mrc, hist_len=3, transform=None, printer=rospy.loginfo):

        self.topic_in   = topic_in
        self.topic_out  = topic_out
        self.namespace  = namespace
        self.exts       = exts
        self.types      = types
        self.hist_len   = hist_len
        self.transform  = transform
        self.mrc        = mrc
        if transform is None:
            self.transform = self.empty
        self.subs       = [rospy.Subscriber(self.topic_in + self.exts[i], self.types[i], self.cb, queue_size=1) \
                           for i in range(len(self.exts))]
        self.pubs       = [self.mrc.add_pub(self.namespace + self.topic_out + self.exts[i], self.types[i], queue_size=1) \
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

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars(kwargs['mode'], kwargs['img_dims'])
        self.init_rospy()
        
        self.node_ready(kwargs['order_id'])

    def init_vars(self, mode, img_dims):
        super().init_vars()

        self.mode           = mode
        self.img_dims       = img_dims

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
        super().init_rospy()
        
        # Set up throttles:
        self.throttles      = [Throttle_Topic(self.cin[i], self.cout[i], self.namespace, self.exts, self.types, self.RATE_NUM.get(), self, \
                                    transform=lambda x: self.img_resize(x, mode="rectangle"), printer=self.print) \
                                if 'stitched' in self.cin[i] else \
                               Throttle_Topic(self.cin[i], self.cout[i], self.namespace, self.exts, self.types, self.RATE_NUM.get(), self, \
                                    transform=lambda x: self.img_resize(x, mode="square"), printer=self.print) \
                                for i in range(len(self.cin))]
        
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
            def transform(img, img_dims):
                return cv2.resize(img, img_dims, interpolation=cv2.INTER_AREA)
        elif mode == "rectangle":
            def transform(img, img_dims):
                img_size    = img.shape
                img_dims = (round((img_size[1]/img_size[0])*img_dims[0]), img_dims[0])
                return cv2.resize(img, img_dims, interpolation=cv2.INTER_AREA)
        else:
            raise Exception("Undefined mode %s, should be either 'square' or 'rectangle'." % mode)
        rmsg                = imgmsgtrans(msg, lambda img: transform(img, self.img_dims), bridge=self.bridge)
        rmsg.header         = Header(stamp=rospy.Time.now(), frame_id=frame_id, seq=msg.header.seq)
        return rmsg
    
    def main(self):
        self.set_state(NodeState.MAIN)

        # loop forever until signal shutdown
        while not rospy.is_shutdown():
            self.rate_obj.sleep()

def do_args():
    parser = ap.ArgumentParser(prog="throttle.py", 
                                description="ROS Topic Throttle Tool",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='throttle')
    parser.add_argument('--img-dims', '-i', type=check_positive_two_int_tuple, default=(64,64), help='Set image dimensions (default: %(default)s).')
    parser.add_argument('--mode',     '-m', type=int, choices=[0,1,2],         default=2,       help="Specify whether to throttle raw (0), compressed (1), or both (2) topics (default: %(default)s).")

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

    
