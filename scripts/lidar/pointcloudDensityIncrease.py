#!/usr/bin/env python3

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

import numpy as np
import math
import cv2
from cv_bridge import CvBridge
import sys
import matplotlib
matplotlib.use("Qt5agg")
from matplotlib import pyplot as plt
from sensor_msgs.msg import CompressedImage
from pyaarapsi.core.helper_tools import Timer
import time

class mrc:
    def __init__(self, node_name='lidar2panorama', anon=True, rate_num=20.0):
        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        self.lidar_topic    = '/velodyne_points'
        self.image_topic    = '/ros_indigosdk_occam/image%d/compressed'

        self.lidar_sub      = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1)

        self.img_subs       = []
        self.img_msgs       = [None]*5
        self.new_imgs       = [False]*5

        self.bridge = CvBridge()
    
        for i in range(5):
            self.img_subs.append(rospy.Subscriber(self.image_topic % (i), CompressedImage, self.image_callback, (i)))

        ## Parse all the inputs:
        self.rate_num       = rate_num # Hz
        self.rate_obj       = rospy.Rate(self.rate_num)

        self.lidar_msg      = PointCloud2()
        self.new_lidar_msg  = False

        self.plotting_init()

    def image_callback(self, msg, id):
        self.img_msgs[id]   = msg
        self.new_imgs[id]   = True

    def lidar_callback(self, msg):
        self.lidar_msg      = msg
        self.new_lidar_msg  = True

    def plotting_init(self):
        self.fig, self.axes = plt.subplots(1,1)

    def main_loop(self):

        timer = Timer()
        timer.add()
        if not self.new_lidar_msg and not all(self.new_imgs):
            return
        
        self.new_lidar_msg = False

        pointcloud_xyzi = np.array(list(point_cloud2.read_points(self.lidar_msg, skip_nans=True, field_names=("x", "y", "z", "intensity", "ring"))))# "time"
        pc_xyzi = pointcloud_xyzi[pointcloud_xyzi[:,2] > -0.5] # ignore erroneous rows when they appear
        x = np.array(pc_xyzi[:,0])
        y = np.array(pc_xyzi[:,1])
        z = np.array(pc_xyzi[:,2])
        r = np.array(pc_xyzi[:,3]) # intensity, reflectance
        l = np.array(pc_xyzi[:,4]) # layer, ring

        self.axes.scatter(x,y,z)

        plt.show()
        sys.exit()


if __name__ == '__main__':
    try:
        nmrc = mrc()

        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            nmrc.main_loop()

        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass
