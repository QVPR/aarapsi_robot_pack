#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospkg
from pathlib import Path
import time
from matplotlib import pyplot as plt
from tqdm import tqdm

# Odometry Multi-Image Processor Node:

# - Subscribes to multiple (3) image feeds and an odometry feed
# - Stores images and position data (position data labelled as odometry)

class mrc: # main ROS class
    def __init__(self):
        self.robot_x            = 0
        self.robot_y            = 0
        self.robot_z            = 0
        self.frame_n            = 0 # n'th image recorded

        rospy.init_node('pos_multifeed_processor', anonymous=True)
        rospy.loginfo('Starting pos_multifeed_processor node.')
        
        self.rate_num           = 4.0 # Hz
        self.rate_obj           = rospy.Rate(self.rate_num)

        self.bridge             = CvBridge() # to convert sensor_msgs/CompressedImage to cv2.

        self.odom_sub           = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)
        self.img_frwd_sub       = rospy.Subscriber("/ros_indigosdk_occam/image0/compressed", CompressedImage, self.img_frwd_callback, queue_size=1) # centre-view
        self.img_rght_sub       = rospy.Subscriber("/ros_indigosdk_occam/image1/compressed", CompressedImage, self.img_rght_callback, queue_size=1) # right-view
        self.img_left_sub       = rospy.Subscriber("/ros_indigosdk_occam/image4/compressed", CompressedImage, self.img_left_callback, queue_size=1) # left-view
        self.panorama_pub       = rospy.Publisher("/ros_indigosdk_occam/merged/pano/compressed", CompressedImage, queue_size=1)
        self.l_merged_pub       = rospy.Publisher("/ros_indigosdk_occam/merged/left/compressed", CompressedImage, queue_size=1)
        self.r_merged_pub       = rospy.Publisher("/ros_indigosdk_occam/merged/right/compressed", CompressedImage, queue_size=1)

        # flags to denest main loop:
        self.new_img_frwd       = False
        self.new_img_rght       = False
        self.new_img_left       = False
        self.new_odom           = False

    def odom_callback(self, msg):
    # /odometry/filtered (nav_msgs/Odometry)
    # Store new robot position

        self.robot_x            = round(msg.pose.pose.position.x, 3)
        self.robot_y            = round(msg.pose.pose.position.y, 3)
        self.robot_z            = round(msg.pose.pose.position.z, 3)
        self.new_odom           = True

    def img_frwd_callback(self, msg):
    # /ros_indigosdk_occam/image0/compressed (sensor_msgs/CompressedImage)
    # Store newest forward-facing image received
        self.store_img_frwd     = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.new_img_frwd       = True

    def img_rght_callback(self, msg):
    # /ros_indigosdk_occam/image1/compressed (sensor_msgs/CompressedImage)
    # Store newest right-facing image received

        self.store_img_rght     = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.new_img_rght       = True

    def img_left_callback(self, msg):
    # /ros_indigosdk_occam/image4/compressed (sensor_msgs/CompressedImage)
    # Store newest left-facing image received

        self.store_img_left     = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.new_img_left       = True

# https://github.com/lukasalexanderweber/stitching_tutorial/blob/master/docs/Stitching%20Tutorial.md
def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def plot_images_two_rows(imgs1, imgs2, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(2, len(imgs1), figsize=figsize_in_inches)
    for col, img in enumerate(imgs1):
        axs[0,col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for col, img in enumerate(imgs2):
        axs[1,col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def odom_multiimage_processor():
    #!# Variables to Update
    set_name                    = "set_1" # Name of folder that will house generated information
    pack_name                   = 'aarapsi_robot_pack' # in case it needs to be adjusted
    nmrc                        = mrc() # make new class instance
    path_for_dataset            = rospkg.RosPack().get_path(pack_name) + "/data/" + set_name
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

    # Make subfolders:
    Path(path_for_dataset + '/left/').mkdir(parents=True, exist_ok=True)
    Path(path_for_dataset + '/forward/').mkdir(parents=True, exist_ok=True)
    Path(path_for_dataset + '/right/').mkdir(parents=True, exist_ok=True)

    Path(path_for_dataset + '/left_corrected/').mkdir(parents=True, exist_ok=True)
    Path(path_for_dataset + '/forward_corrected/').mkdir(parents=True, exist_ok=True)
    Path(path_for_dataset + '/right_corrected/').mkdir(parents=True, exist_ok=True)

    Path(path_for_dataset + '/leftmerge/').mkdir(parents=True, exist_ok=True)
    Path(path_for_dataset + '/rightmerge/').mkdir(parents=True, exist_ok=True)
    Path(path_for_dataset + '/panorama/').mkdir(parents=True, exist_ok=True)

    Path(path_for_dataset + '/odo/').mkdir(parents=True, exist_ok=True)

    rospy.loginfo("Ready, listening...")

    # Main loop:
    while not rospy.is_shutdown():

        nmrc.rate_obj.sleep()


        if not (nmrc.new_img_frwd and nmrc.new_img_left and nmrc.new_img_rght and nmrc.new_odom): # denest
           continue

        rospy.loginfo("Logging frame {}".format(str(nmrc.frame_n).zfill(6)))

        # Clear denest flags:
        nmrc.new_img_frwd   = False
        nmrc.new_img_rght   = False
        nmrc.new_img_left   = False
        nmrc.new_odom       = False

        ### Correct barrel distortion:
        ## needs revision; calibration, per-camera coefficients. all of this was a guessing game aided by a dirty GUI
        # Idea: https://stackoverflow.com/questions/26602981/correct-barrel-distortion-in-opencv-manually-without-chessboard-image
        # GUI: https://github.com/kaustubh-sadekar/VirtualCam
        width = nmrc.store_img_frwd.shape[1]
        height = nmrc.store_img_frwd.shape[0]

        ## coefficients for left and forward camera:
        distCoeff1 = np.zeros((4,1),np.float64)
        distCoeff1[0,0] = -8.0e-5#k1
        distCoeff1[1,0] = -1.0e-8#k2
        distCoeff1[2,0] = 0#p1
        distCoeff1[3,0] = 0#p2

        cam1 = np.eye(3,dtype=np.float32)
        cam1[0,2] = width/2.0  # define center x
        cam1[1,2] = height/2.0 # define center y
        cam1[0,0] = 10.        # define focal length x
        cam1[1,1] = 10.        # define focal length y

        ## coefficients for right camera:
        distCoeff2 = np.zeros((4,1),np.float64)
        distCoeff2[0,0] = -1.0e-5#k1
        distCoeff2[1,0] = 0#k2
        distCoeff2[2,0] = 0#p1
        distCoeff2[3,0] = 0#p2

        cam2 = np.eye(3,dtype=np.float32)
        cam2[0,2] = width/2.0 - 200  # define center x
        cam2[1,2] = height/2.0 - 600 # define center y
        cam2[0,0] = 10.        # define focal length x
        cam2[1,1] = 10.        # define focal length y

        ## perform distortion removal:
        crrctd_left = cv2.undistort(nmrc.store_img_left, cam1, distCoeff1)[20:-5, :]
        crrctd_frwd = cv2.undistort(nmrc.store_img_frwd, cam1, distCoeff1)[20:-5, :]
        crrctd_rght = cv2.undistort(nmrc.store_img_rght, cam2, distCoeff2)[20:-5, :]

        ## set shorthands for all file paths:
        left_image = path_for_dataset + '/left/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6))
        frwd_image = path_for_dataset + '/forward/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6))   
        rght_image = path_for_dataset + '/right/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6))  

        left_crrct = path_for_dataset + '/left_corrected/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6))
        frwd_crrct = path_for_dataset + '/forward_corrected/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6))   
        rght_crrct = path_for_dataset + '/right_corrected/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6))  

        pnrm_image = path_for_dataset + '/panorama/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6))
        left_merge = path_for_dataset + '/leftmerge/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6))  
        rght_merge = path_for_dataset + '/rightmerge/frame_id_{}.png'.format(str(nmrc.frame_n).zfill(6))

        ## create panoramic image:
        crrctd_pnrm = np.concatenate((crrctd_left[:, 80:-19], crrctd_frwd[:, 20:-19], crrctd_rght[:, 100:(crrctd_rght.shape[1]-1)]), axis=1)

        ## create left/right merge images using extractions from panorama:
        dm_ol   = 2.0/3.0                       # decimal overlap, include two thirds of middle frame
        hw_p    = int(crrctd_pnrm.shape[1] / 2) # halfwidth of panorama
        fw_i    = int(crrctd_frwd.shape[1])     # fullwidth of forward camera

        left_img_left_bound = int(hw_p - (fw_i * (dm_ol)))
        left_img_rght_bound = int(hw_p + (fw_i * (1 - dm_ol)))
        rght_img_left_bound = int(hw_p - (fw_i * (1 - dm_ol)))
        rght_img_rght_bound = int(hw_p + (fw_i * (dm_ol)))

        left_img_merge = crrctd_pnrm[:, left_img_left_bound : left_img_rght_bound]
        rght_img_merge = crrctd_pnrm[:, rght_img_left_bound : rght_img_rght_bound]

        ## Save images to folders: 
        # Left/Forward/Right
        cv2.imwrite(left_image, nmrc.store_img_left) 
        cv2.imwrite(frwd_image, nmrc.store_img_frwd) 
        cv2.imwrite(rght_image, nmrc.store_img_rght) 
        # Left/Forward/Right Corrected
        cv2.imwrite(left_crrct, crrctd_left)
        cv2.imwrite(frwd_crrct, crrctd_frwd)
        cv2.imwrite(rght_crrct, crrctd_rght)
        # Panorama, Left+Forward/Forward+Right Merge
        cv2.imwrite(pnrm_image, crrctd_pnrm)
        cv2.imwrite(left_merge, left_img_merge)
        cv2.imwrite(rght_merge, rght_img_merge)

        np.savetxt(path_for_dataset + '/odo/frame_id_{}.csv'.format(str(nmrc.frame_n).zfill(6)), np.array([nmrc.robot_x, nmrc.robot_y, nmrc.robot_z]),delimiter=',')
        
        ## Publish to ROS for viewing pleasure (optional)
        # convert to ROS message first
        cimage_ros_pano = nmrc.bridge.cv2_to_compressed_imgmsg(crrctd_pnrm, "png")
        cimage_ros_left = nmrc.bridge.cv2_to_compressed_imgmsg(left_img_merge, "png")
        cimage_ros_rght = nmrc.bridge.cv2_to_compressed_imgmsg(rght_img_merge, "png")
        # publish
        nmrc.panorama_pub.publish(cimage_ros_pano)
        nmrc.l_merged_pub.publish(cimage_ros_left)
        nmrc.r_merged_pub.publish(cimage_ros_rght)

        nmrc.frame_n += 1

if __name__ == '__main__':
    try:
        odom_multiimage_processor()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass