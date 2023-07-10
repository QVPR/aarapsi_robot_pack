#!/usr/bin/env python3

import rospy
import sys
import numpy as np
import argparse as ap
import cv2
from sensor_msgs.msg import CompressedImage

from pyaarapsi.core.ros_tools       import LogType, roslogger, np2compressed, compressed2np, NodeState
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

### Correct barrel distortion:
## needs revision; calibration, per-camera coefficients. all of this was a guessing game aided by a dirty GUI
# Idea: https://stackoverflow.com/questions/26602981/correct-barrel-distortion-in-opencv-manually-without-chessboard-image
# GUI: https://github.com/kaustubh-sadekar/VirtualCam

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_params(self, rate_num, log_level, reset):
        super().init_params(rate_num, log_level, reset)

    def init_vars(self):
        super().init_vars()

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

    def init_rospy(self):
        super().init_rospy()

        self.cam0_sub           = rospy.Subscriber("/camera%d/image/compressed" % 0, CompressedImage, lambda msg: self.img_callback(0, msg), queue_size=1)
        self.cam1_sub           = rospy.Subscriber("/camera%d/image/compressed" % 1, CompressedImage, lambda msg: self.img_callback(1, msg), queue_size=1)
        self.cam2_sub           = rospy.Subscriber("/camera%d/image/compressed" % 2, CompressedImage, lambda msg: self.img_callback(2, msg), queue_size=1)
        self.cam3_sub           = rospy.Subscriber("/camera%d/image/compressed" % 3, CompressedImage, lambda msg: self.img_callback(3, msg), queue_size=1)
        self.cam4_sub           = rospy.Subscriber("/camera%d/image/compressed" % 4, CompressedImage, lambda msg: self.img_callback(4, msg), queue_size=1)

        self.pano_pub           = self.add_pub("/ros_indigosdk_occam/stitched_image0/compressed",    CompressedImage, queue_size=1)
        self.cam0_pub           = self.add_pub("/ros_indigosdk_occam/image%d/compressed" % 0,        CompressedImage, queue_size=1)
        self.cam1_pub           = self.add_pub("/ros_indigosdk_occam/image%d/compressed" % 1,        CompressedImage, queue_size=1)
        self.cam2_pub           = self.add_pub("/ros_indigosdk_occam/image%d/compressed" % 2,        CompressedImage, queue_size=1)
        self.cam3_pub           = self.add_pub("/ros_indigosdk_occam/image%d/compressed" % 3,        CompressedImage, queue_size=1)
        self.cam4_pub           = self.add_pub("/ros_indigosdk_occam/image%d/compressed" % 4,        CompressedImage, queue_size=1)

    def img_callback(self, index: int, msg: CompressedImage):
    # /ros_indigosdk_occam/image0/compressed (sensor_msgs/CompressedImage)
    # Store newest forward-facing image received
        self.new_imgs[index]    = True
        self.imgs[index]        = compressed2np(msg, "bgr8")

    
    def main(self):
        # Main loop process
        self.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            try:
                self.loop_contents()
            except rospy.exceptions.ROSInterruptException as e:
                pass
            except Exception as e:
                if self.parameters_ready:
                    raise Exception('Critical failure. ' + formatException()) from e
                else:
                    self.print('Main loop exception, attempting to handle; waiting for parameters to update. Details:\n' + formatException(), LogType.DEBUG, throttle=5)
                    rospy.sleep(0.5)

    def loop_contents(self):

        if not all(self.new_imgs): # denest
            self.print('Waiting... %s' % str(self.new_imgs), LogType.DEBUG, throttle=5)
            rospy.sleep(0.001)
            return
        
        self.rate_obj.sleep()
        self.new_imgs = [False] * 5

        try:
            ## perform distortion removal:
            corrected_imgs = [cv2.undistort(i, self.cam, self.distCoeff)[:, 30:-30] for i in self.imgs]
        except:
            self.print(formatException(), LogType.WARN)
            return

        ## create panoramic image:
        panorama = np.concatenate(corrected_imgs[3:] + corrected_imgs[:3], axis=1)[:-20,:]
        corrected_pano = cv2.resize(panorama, (panorama.shape[1], int(panorama.shape[0]*1.5)), interpolation=cv2.INTER_AREA)
        
        # publish:
        self.pano_pub.publish(np2compressed(corrected_pano, "png"))
        self.cam0_pub.publish(np2compressed(self.imgs[0],   "png"))
        self.cam1_pub.publish(np2compressed(self.imgs[1],   "png"))
        self.cam2_pub.publish(np2compressed(self.imgs[2],   "png"))
        self.cam3_pub.publish(np2compressed(self.imgs[3],   "png"))
        self.cam4_pub.publish(np2compressed(self.imgs[4],   "png"))

def do_args():
    parser = ap.ArgumentParser(prog="multicam_fusion.py", 
                            description="Fuser for simulated occam camera",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='multicam_fusion')

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