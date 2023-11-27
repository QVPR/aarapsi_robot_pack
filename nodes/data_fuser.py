#!/usr/bin/env python3

import rospy
import sys
import argparse as ap
import numpy as np

from sensor_msgs.msg                import CompressedImage
from nav_msgs.msg                   import Odometry
from aarapsi_robot_pack.msg         import Label, xyw

from pyaarapsi.core.ros_tools       import NodeState, LogType, pose2xyw, roslogger
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

'''
Data Fuser

This node exists to fuse a compressed image stream with an odometry stream 
into a single synchronised message structure that keeps images and
odometry aligned in time across networks. Without the use of this 
node, if either the odometry or image stream fails, nodes further 
down the pipeline may still act, likely to a poor result, or get 
into locked or bad states. This node ensures down-pipeline nodes can
only progress if both odometry and images arrive across the network.

'''

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

        self.img            = CompressedImage()
        self.gt_odom        = Odometry()
        self.robot_odom     = Odometry()
        self.new_img        = False
        self.new_gt_odom    = False
        self.new_robot_odom = False
        self.time           = rospy.Time.now()

    def init_rospy(self):
        super().init_rospy()
        
        self.img_sub        = rospy.Subscriber(self.IMG_TOPIC.get(), CompressedImage, self.img_cb, queue_size=1)
        self.gt_odom_sub    = rospy.Subscriber(self.SLAM_ODOM_TOPIC.get(), Odometry, self.gt_odom_cb, queue_size=1)
        self.robot_odom_sub = rospy.Subscriber(self.ROBOT_ODOM_TOPIC.get(), Odometry, self.robot_odom_cb, queue_size=1)
        self.pub            = self.add_pub(self.namespace + '/img_odom', Label, queue_size=1)

    def gt_odom_cb(self, msg: Odometry):
        self.gt_odom        = msg
        self.new_gt_odom    = True

    def robot_odom_cb(self, msg: Odometry):
        self.robot_odom     = msg
        self.new_robot_odom = True

    def img_cb(self, msg: CompressedImage):
        self.img            = msg
        self.new_img        = True

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

        if rospy.Time.now().to_sec() - self.time.to_sec() > 1:
            self.print("Long message delay experienced.", LogType.WARN, throttle=5)

        if not (self.new_img and self.new_gt_odom and self.new_robot_odom):
            rospy.sleep(0.005)
            return # denest
        
        self.rate_obj.sleep()
        self.time                   = rospy.Time.now()

        self.new_img                = False
        self.new_gt_odom            = False
        self.new_robot_odom         = False

        msg_to_pub                  = Label()
        msg_to_pub.header.stamp     = self.time
        msg_to_pub.header.frame_id  = 'map'
        msg_to_pub.gt_ego           = xyw(*pose2xyw(self.gt_odom.pose.pose))
        msg_to_pub.robot_ego        = xyw(*pose2xyw(self.robot_odom.pose.pose))
        msg_to_pub.query_image      = self.img
        msg_to_pub.stamps           = [self.time]
        msg_to_pub.step             = msg_to_pub.DATA
        msg_to_pub.id               = int((1 + np.random.rand()) * 100000000000)
        self.pub.publish(msg_to_pub)

def do_args():
    parser = ap.ArgumentParser(prog="data_fuser.py", 
                            description="ROS Compressed Image+Odom Data Fuser Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='data_fuser')

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