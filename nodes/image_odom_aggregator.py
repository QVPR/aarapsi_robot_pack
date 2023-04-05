#!/usr/bin/env python3

import rospy
import argparse as ap

from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from aarapsi_robot_pack.msg import ImageOdom, CompressedImageOdom

from aarapsi_robot_pack.core.argparse_tools import check_positive_float, check_bool, check_string

class mrc:
    def __init__(self, node_name="image_odom_aggregator", rate=10.0, anon=True, \
                 img_topic='/occam/image0', odom_topic='/odom/filtered', pub_topic='/data/img_odom', \
                 compressed=True, log_level=rospy.INFO):
        
        rospy.init_node(node_name, anonymous=anon, log_level=log_level)
        rospy.loginfo('Starting %s node.' % (node_name))

        self.rate_num   = rate
        self.rate_obj   = rospy.Rate(self.rate_num)

        self.img        = None
        self.odom       = None
        self.new_img    = False
        self.new_odom   = False

        self.odom_topic = odom_topic

        self.compressed = compressed
        if self.compressed:
            self.img_topic = img_topic + '/compressed'
            self.pub_topic = pub_topic + '/compressed'
            self.img_type  = CompressedImage
            self.pub_type  = CompressedImageOdom
        else:
            self.img_topic = img_topic
            self.pub_topic = pub_topic
            self.img_type  = Image
            self.pub_type  = ImageOdom

        self.img_sub    = rospy.Subscriber(self.img_topic, self.img_type, self.img_cb, queue_size=1)
        self.odom_sub   = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.pub        = rospy.Publisher(self.pub_topic, self.pub_type, queue_size=1)

    def odom_cb(self, msg):
        self.odom       = msg
        self.new_odom   = True

    def img_cb(self, msg):
        self.img        = msg
        self.new_img    = True

    def main(self):
        while not rospy.is_shutdown():
            self.rate_obj.sleep()
            if not (self.new_img and self.new_odom):
                continue # denest
            self.new_img                = False
            self.new_odom               = False

            msg_to_pub                  = self.pub_type()
            msg_to_pub.header.stamp     = rospy.Time.now()
            msg_to_pub.header.frame_id  = self.odom.header.frame_id
            msg_to_pub.odom             = self.odom
            msg_to_pub.image            = self.img

            self.pub.publish(msg_to_pub)
            del msg_to_pub

if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(prog="image+odom aggregator", 
                                description="ROS Aggregator Tool",
                                epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
        parser.add_argument('--node-name', '-N', type=check_string,              default="img_odom_agg",   help="Specify node name (default: %(default)s).")
        parser.add_argument('--rate', '-r',      type=check_positive_float,      default=10.0,             help='Set node rate (default: %(default)s).')
        parser.add_argument('--anon', '-a',      type=check_bool,                default=False,            help="Specify whether node should be anonymous (default: %(default)s).")
        parser.add_argument('--img-topic',       type=check_string,              default='/occam/image0',  help="Specify input image topic (exclude /compressed) (default: %(default)s).")
        parser.add_argument('--odom-topic',      type=check_string,              default='/odom/filtered', help="Specify input odometry topic (exclude /compressed) (default: %(default)s).")
        parser.add_argument('--pub-topic',       type=check_string,              default='/data/img_odom', help="Specify output topic (exclude /compressed) (default: %(default)s).")
        parser.add_argument('--compress', '-C',  type=check_bool,                default=True,             help='Enable image compression (default: %(default)s)')
        parser.add_argument('--log-level', '-V', type=int, choices=[1,2,4,8,16], default=2,                help="Specify ROS log level (default: %(default)s).")
        
        raw_args = parser.parse_known_args()
        args = vars(raw_args[0])

        node_name   = args['node_name']
        rate        = args['rate']
        anon        = args['anon']
        img_topic   = args['img_topic']
        odom_topic  = args['odom_topic']
        pub_topic   = args['pub_topic']
        compressed  = args['compress']
        log_level   = args['log_level']

        nmrc = mrc(node_name, rate, anon, img_topic, odom_topic, pub_topic, compressed, log_level)
        nmrc.main()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass