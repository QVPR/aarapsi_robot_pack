#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

### Correct barrel distortion:
## needs revision; calibration, per-camera coefficients. all of this was a guessing game aided by a dirty GUI
# Idea: https://stackoverflow.com/questions/26602981/correct-barrel-distortion-in-opencv-manually-without-chessboard-image
# GUI: https://github.com/kaustubh-sadekar/VirtualCam

class mrc: # main ROS class
    def __init__(self):

        rospy.init_node('multicam_fusion', anonymous=False)
        rospy.loginfo('Starting multicam_fusion node.')
        
        self.rate_num           = 20.0 # Hz

        self.bridge             = CvBridge() # to convert sensor_msgs/CompressedImage to cv2.

        # flags to denest main loop:
        self.new_imgs           = [False] * 5
        self.imgs               = [None] * 5

        self.distCoeff          = np.zeros((4,1),np.float64)
        self.distCoeff[0,0]     = -25.0e-5#k1
        self.distCoeff[1,0]     = 10.0e-8#k2
        self.distCoeff[2,0]     = 0#p1
        self.distCoeff[3,0]     = 0#p2

        self.cam                = np.eye(3,dtype=np.float32)
        self.cam[0,2]           = 720/2.0  # define center x
        self.cam[1,2]           = 480/2.0 # define center y
        self.cam[0,0]           = 10.        # define focal length x
        self.cam[1,1]           = 10.        # define focal length y


        self.rate_obj           = rospy.Rate(self.rate_num)

        rospy.Subscriber("/camera%d/image/compressed" % 0, CompressedImage, lambda msg: self.img_callback(0, msg), queue_size=1)
        rospy.Subscriber("/camera%d/image/compressed" % 1, CompressedImage, lambda msg: self.img_callback(1, msg), queue_size=1)
        rospy.Subscriber("/camera%d/image/compressed" % 2, CompressedImage, lambda msg: self.img_callback(2, msg), queue_size=1)
        rospy.Subscriber("/camera%d/image/compressed" % 3, CompressedImage, lambda msg: self.img_callback(3, msg), queue_size=1)
        rospy.Subscriber("/camera%d/image/compressed" % 4, CompressedImage, lambda msg: self.img_callback(4, msg), queue_size=1)

        self.pub                = rospy.Publisher("/ros_indigosdk_occam/stitched_image0/compressed", CompressedImage, queue_size=1)

    def img_callback(self, index, msg):
    # /ros_indigosdk_occam/image0/compressed (sensor_msgs/CompressedImage)
    # Store newest forward-facing image received
        self.new_imgs[index]    = True
        self.imgs[index]        = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

    def main(self):

        rospy.loginfo('Entering main loop.')

        # Main loop:
        while not rospy.is_shutdown():

            if not all(self.new_imgs): # denest
                rospy.loginfo_throttle(15, 'Waiting... %s' % str(self.new_imgs))
                rospy.sleep(0.001)
                continue

            nmrc.rate_obj.sleep()
            self.new_imgs = [False] * 5

            ## perform distortion removal:
            corrected_imgs = [cv2.undistort(i, self.cam, self.distCoeff)[:, 30:-30] for i in self.imgs]

            ## create panoramic image:
            panorama = np.concatenate(corrected_imgs[3:] + corrected_imgs[:3], axis=1)[:-20,:]
            corrected_pano = cv2.resize(panorama, (panorama.shape[1], int(panorama.shape[0]*1.5)), interpolation=cv2.INTER_AREA)
            
            
            ## Publish to ROS for viewing pleasure (optional)
            # convert to ROS message first
            ros_pano = nmrc.bridge.cv2_to_compressed_imgmsg(corrected_pano, "png")\
            # publish
            nmrc.pub.publish(ros_pano)

if __name__ == '__main__':
    try:
        nmrc = mrc()
        nmrc.main()
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass