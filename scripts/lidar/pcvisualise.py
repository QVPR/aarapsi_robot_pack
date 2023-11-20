#!/usr/bin/env python3

import rospy
import rosbag
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, CompressedImage, PointField

import numpy as np
import cv2
from cv_bridge import CvBridge
import math
import sys
import os
import time
import struct

import matplotlib
matplotlib.use("Qt5agg")
from matplotlib import pyplot as plt

from pyaarapsi.core.ros_tools import open_rosbag

class mrc:
    def __init__(self, node_name='pcvisualise', anon=True, rate_num=20.0):
        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        self.cloud_topic    = '/cloud/raw'
        self.cloud_pub      = rospy.Publisher(self.cloud_topic, PointCloud2, queue_size=1)

        self.filtered_cloud_topic    = '/cloud/filtered'
        self.filtered_cloud_pub      = rospy.Publisher(self.filtered_cloud_topic, PointCloud2, queue_size=1)

        ## Parse all the inputs:
        self.rate_num       = rate_num # Hz
        self.rate_obj       = rospy.Rate(self.rate_num)

    def main(self):

        bag_path = '/home/claxton/Documents/bags/'
        bag_name = 'trip1.bag'
        
        for topic, msg, timestamp in open_rosbag(bag_path + bag_name, topics=['/velodyne_points']):
            if rospy.is_shutdown():
                sys.exit()
            nmrc.rate_obj.sleep()
            if topic == '/velodyne_points':
                self.cloud_pub.publish(msg)
                pc_np = np.array(list(point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity", "ring"))))# "time"

                # Extract Data from PointCloud:
                x = np.array(pc_np[:,0], dtype=np.float32)
                y = np.array(pc_np[:,1], dtype=np.float32)
                z = np.array(pc_np[:,2], dtype=np.float32)
                i = np.array(pc_np[:,3], dtype=np.float32) # intensity, reflectance
                r = np.array(pc_np[:,4], dtype=np.float32) # ring, layer

                ####
                # Operate on PointCloud Data:

                z = z - (z % 0.05) # discretise z-data

                #
                ####

                # Reform Data into PointCloud:
                header = Header(frame_id = 'velodyne', stamp = rospy.Time.now())
                fields = [PointField('x',           0,  PointField.FLOAT32, 1), \
                          PointField('y',           4,  PointField.FLOAT32, 1), \
                          PointField('z',           8,  PointField.FLOAT32, 1), \
                          PointField('intensity',   16, PointField.FLOAT32, 1), \
                          PointField('ring',        20, PointField.FLOAT32, 1)]
                points = np.column_stack((x, y, z, i, r))
                pointcloud_reconstructed = point_cloud2.create_cloud(header, fields, points)
                self.filtered_cloud_pub.publish(pointcloud_reconstructed)
                input("Press Enter to continue...")

            # ros_bag.close()


if __name__ == '__main__':
    try:
        nmrc = mrc()
        nmrc.main()

        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass
