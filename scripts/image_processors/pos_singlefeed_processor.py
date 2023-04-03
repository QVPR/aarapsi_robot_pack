#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage#, Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import rospkg
from pathlib import Path
import time
from tqdm import tqdm

# Odometry Image Processor Node:

# - Subscribes to an image feed and an odometry feed
# - Stores images and position data (position data labelled as odometry)

class mrc: # main ROS class
    def __init__(self):
        self.robot_x    = 0
        self.robot_y    = 0
        self.robot_z    = 0
        self.frame_n    = 0 # n'th image recorded

        rospy.init_node('pos_singlefeed_processor', anonymous=True)
        rospy.loginfo('Starting pos_singlefeed_processor node.')
        self.rate_num   = 0.5 # Hz
        self.rate_obj   = rospy.Rate(self.rate_num)

        self.bridge     = CvBridge() # to convert sensor_msgs/Image (or CompressedImage) to cv2

        self.odom_sub   = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)
        self.img0_sub   = rospy.Subscriber("/ros_indigosdk_occam/image0/compressed", CompressedImage, self.img0_callback)
        # alternative: self.img0_sub   = rospy.Subscriber("/ros_indigosdk_occam/image0", Image, self.img0_callback)

        #self.store_img0 = self.bridge.compressed_imgmsg_to_cv2(CompressedImage, "bgr8") # populate, empty
        ## alternative: self.store_img0 = self.bridge.imgmsg_to_cv2(Image, "bgr8")

        # flags to denest main loop:
        self.new_img0   = False
        self.new_odom   = False

    def odom_callback(self, msg):
    # /odometry/filtered (nav_msgs/Odometry)
    # Store new robot position

        self.robot_x    = round(msg.pose.pose.position.x, 3)
        self.robot_y    = round(msg.pose.pose.position.y, 3)
        self.robot_z    = round(msg.pose.pose.position.z, 3)
        self.new_odom   = True

    def img0_callback(self, msg):
    # /ros_indigosdk_occam/image0/compressed (sensor_msgs/CompressedImage)
    # Store newest image received

        self.store_img0 = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        # alternative: self.store_img0 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.new_img0   = True

def odom_image_processor():
    #!# Variables to Update
    set_name            = "set_1" # Name of folder that will house generated information
    pack_name           = 'aarapsi_robot_pack' # in case it needs to be adjusted
    nmrc                = mrc() # make new class instance
    path_for_dataset    = rospkg.RosPack().get_path(pack_name) + "/data/" + set_name

    # Handle checks for folder construction:
    rospy.loginfo("Attempting to construct data storage system")
    try:
        Path(path_for_dataset).mkdir(parents=False, exist_ok=False) # throw both error states
    except FileNotFoundError:
        rospy.logerr("Error: parent directory does not exist. Exitting...")
        return
    except FileExistsError:
        rospy.logwarn("Directory already exists - this will overwrite existing data! Pausing in case of error (3s)...")
        for i, count in tqdm(enumerate(range(0, 30, 2))): # keep spooling and listen for Ctrl+C
            time.sleep(0.2)
            if rospy.is_shutdown():
                return
        rospy.logwarn("Continuing...")
    Path(path_for_dataset + '/images/').mkdir(parents=True, exist_ok=True)
    Path(path_for_dataset + '/odo/').mkdir(parents=True, exist_ok=True)

    rospy.loginfo("Ready, listening...")

    # Main loop:
    while not rospy.is_shutdown():

        nmrc.rate_obj.sleep()

        if not nmrc.new_img0: # denest
            continue
        
        # Store a new image and log new odom
        rospy.loginfo("Logging frame {}".format(str(nmrc.frame_n).zfill(6)))
        cv.imwrite(path_for_dataset + '/images/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6)), nmrc.store_img0)
        np.savetxt(path_for_dataset + '/odo/frame_id_{}.csv'.format(str(nmrc.frame_n).zfill(6)), np.array([nmrc.robot_x, nmrc.robot_y, nmrc.robot_z]),delimiter=',')
        nmrc.frame_n += 1

        # Clear denest flags:
        nmrc.new_img0 = False
        nmrc.new_odom = False

if __name__ == '__main__':
    try:
        odom_image_processor()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass
