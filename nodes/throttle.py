#!/usr/bin/env python3

import rospy
import argparse as ap
from sensor_msgs.msg import Image, CompressedImage
from aarapsi_robot_pack.core.argparse_tools import check_positive_float, check_positive_two_int_tuple, check_bool

class Throttle_Topic:
    def __init__(self, topic_in, topic_out, exts, types, rate, hist_len=3, transform=None):
        self.topic_in   = topic_in
        self.topic_out  = topic_out
        self.exts       = exts
        self.types      = types
        self.hist_len   = hist_len
        self.transform  = transform
        if transform is None:
            self.transform = self.empty
        self.subs       = [rospy.Subscriber(self.topic_in + self.exts[i], self.types[i], lambda x: self.cb(x, i), queue_size=1) \
                           for i in range(len(self.exts))]
        self.pubs       = [rospy.Publisher(self.topic_out + self.exts[i], self.types[i], queue_size=1) \
                           for i in range(len(self.exts))]
        self.timer      = rospy.Timer(rospy.Duration(1/rate), self.timer)
        self.msgs       = [[] for i in types]

        exts_string     = ''.join(self.exts)
        rospy.loginfo("Throttling %s[%s] to %s[%s] at %0.2f Hz" % (topic_in, exts_string, topic_out, exts_string, rate))

    def cb(self, msg, index):
        self.msgs[index].append(msg)
        while len(self.msgs[index]) > self.hist_len:
            self.msgs[index].pop(0)

    def empty(self, msg):
    # dummy transform function
        return msg

    def timer(self, event):
        for i in range(len(self.pubs)):
            if len(self.msgs[i]) > 0:
                self.pubs[i].publish(self.transform(self.msgs[i].pop(0)))

def get_cam_topics():
# Helper function to abstract topic generation
    ns_in   = '/ros_indigosdk_occam'
    ns_out  = '/occam'
    topics  = ['/image0', '/image1', '/image2', '/image3', '/image4', \
              '/stitched_image0']
    topics_to_throttle_in = []
    topics_to_throttle_out = []
    for topic in topics:
        topics_to_throttle_in.append(ns_in + topic)
        topics_to_throttle_out.append(ns_out + topic)
    return topics_to_throttle_in, topics_to_throttle_out

from cv_bridge import CvBridge
import cv2
bridge = CvBridge()

def img_resize(msg, resize_dims=None):
    if resize_dims is None:
        return msg
    global bridge
    if isinstance(msg, CompressedImage):
        img         = bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
        resized_img = cv2.resize(img, resize_dims, interpolation=cv2.INTER_AREA)
        rospy.loginfo_once("[CompressedImage] Resizing from %s" % (str(img.shape)))
        resized_msg = bridge.cv2_to_compressed_imgmsg(resized_img, "jpeg")
        return resized_msg
    elif isinstance(msg, Image):
        img         = bridge.imgmsg_to_cv2(msg, "passthrough")
        resized_img = cv2.resize(img, resize_dims, interpolation=cv2.INTER_AREA)
        rospy.loginfo_once("[Image] Resizing from %s" % (str(img.shape)))
        resized_msg = bridge.cv2_to_imgmsg(resized_img, "bgr8")
        return resized_msg
    else:
        print(type(msg))
        raise Exception("msg is not of type Image or CompressedImage")

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="topic throttle", 
                                description="ROS Topic Throttle Tool",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    parser.add_argument('--rate', '-r', type=check_positive_float, default=10.0, help='Set node rate (default: %(default)s).')
    parser.add_argument('--img-dims', '-i', type=check_positive_two_int_tuple, default=(64,64), help='Set image dimensions (default: %(default)s).')
    parser.add_argument('--node-name', '-N', default="throttle", help="Specify node name (default: %(default)s).")
    parser.add_argument('--anon', '-a', type=check_bool, default=True, help="Specify whether node should be anonymous (default: %(default)s).")
    parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2, help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--mode', '-m', type=int, choices=[0,1,2], default=2, help="Specify whether to throttle raw (0), compressed (1), or both (2) topics (default: %(default)s).")

    raw_args = parser.parse_known_args()
    args = vars(raw_args[0])

    rate        = args['rate']
    resize_dims = args['img_dims']
    log_level   = args['log_level']
    node_name   = args['node_name']
    anon        = args['anon']
    mode        = args['mode']

    rospy.init_node(node_name, anonymous=anon, log_level=log_level)
    rospy.logdebug("PARAMETERS:\n\t\t\t\tNode Name: %s\n\t\t\t\tMode: %s\n\t\t\t\tAnonymous: %s\n\t\t\t\tLogging Level: %s\n\t\t\t\tRate: %s\n\t\t\t\tResize Dimensions: %s" \
                   % (str(node_name), str(mode), str(anon), str(log_level), str(rate), str(resize_dims)))
    rate_obj    = rospy.Rate(rate)

    # set up topics
    c_in, c_out = get_cam_topics()
    if mode == 0:
       exts = ['']
       types = [Image]
    elif mode == 1:
       exts = ['/compressed']
       types = [CompressedImage]
    elif mode == 2:
       exts = ['', '/compressed']
       types = [Image, CompressedImage]
    else:
       raise Exception('Unknown mode')

    # set up throttles
    throttled_topics = [Throttle_Topic(c_in[i], c_out[i], \
                                       exts, types, rate, transform=lambda x: img_resize(x, resize_dims)) \
                            for i in range(len(c_in))]
    
    # loop forever until signal shutdown
    while not rospy.is_shutdown():
        rate_obj.sleep()
