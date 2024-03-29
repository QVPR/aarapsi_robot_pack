#!/usr/bin/env python3

import rospy
import rospkg
import argparse as ap
import numpy as np
import sys
import os
import csv

from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from aarapsi_robot_pack.srv import SaveObj, SaveObjResponse

from pyaarapsi.core.argparse_tools      import check_positive_float, check_bool, check_string
from pyaarapsi.core.ros_tools           import NodeState, roslogger, LogType, pose2xyw, pose_covariance_to_stamped
from pyaarapsi.core.helper_tools        import formatException, vis_dict
from pyaarapsi.core.enum_tools          import enum_value_options
from pyaarapsi.core.file_system_tools   import scan_directory
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

'''
Pose Recorder

Subscribe to an odometry topic, and use it to generate an array of points as well as publish a path object for visualisation purposes.

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

        self.SEPARATION     = self.params.add(self.nodespace + "/separation", 0.2, check_positive_float, force=False)

    def init_vars(self):
        super().init_vars()

        self.points         = []
        self.path           = Path(header=Header(stamp=rospy.Time.now(), frame_id='map'))
        self.root           = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/paths'
        self.last_position  = None

    def init_rospy(self):
        super().init_rospy()

        self.sub        = rospy.Subscriber( self.ODOM_TOPIC.get(),      Odometry, self.odom_cb, queue_size=1)
        self.path_pub   = self.add_pub(     self.namespace + '/path',   Path,                   queue_size=1)
        self.path_srv   = rospy.Service(    self.nodespace + '/path',   SaveObj,  self.handle_path)

    def handle_path(self, req):
        '''
        service type: aarapsi_robot_pack/SaveObj
        '''
        response = SaveObjResponse()

        try:
            # Warn user if bad extension provided:
            if not ''.join(os.path.splitext(req.destination)[1:]) in ['', '.csv']:
                self.print('[handle_path] File extensions are ignored; will be saved as .csv', LogType.WARN)

            # Figure out correct file directory:
            if not len(os.path.dirname(req.destination)):
                file_dir = self.root
            elif os.path.exists(self.root + os.path.dirname(req.destination)):
                file_dir = self.root + '/' + os.path.dirname(req.destination)
            elif os.path.exists(req.destination):
                file_dir = os.path.dirname(req.destination)
            else:
                self.print('Could not find a directory that exists matching %s.' % str(req.destination), LogType.WARN)
                file_dir = self.root
            self.print('Directory selected as: %s' % str(file_dir), LogType.INFO)

            # Determine whether a file name was provided:
            file_name = os.path.splitext(os.path.basename(req.destination))[0].lower()
            if os.path.isdir(file_dir + '/' + file_name):
                file_dir = file_dir + '/' + file_name
                file_name = 'path_points'

            # Ensure unique file name:
            file_list, _, _ = scan_directory(file_dir, short_files=True)
            count = 0
            original_name = file_name
            while file_name in file_list:
                file_name = original_name + "_%d" % count
                count += 1

            file_path = file_dir + '/' + file_name + '.csv'

            if not len(self.points):
                raise Exception('No data to save.')

            # Save points:
            self.print("Saving points to file %s" % str(file_name), LogType.INFO)
            with open(file_path, "w") as f:
                write = csv.writer(f)
                write.writerow(np.array(self.points)[:,0]) # x
                write.writerow(np.array(self.points)[:,1]) # y
                write.writerow(np.array(self.points)[:,2]) # w
                f.close()
            response.success = True

        except:
            self.print("Save operation failed.", LogType.ERROR)
            self.print(formatException(), LogType.DEBUG)
            response.success = False

        return response

    def odom_cb(self, msg):
        '''
        message type: nav_msgs/Odometry  
        '''
        if self.last_position is None:
            self.last_position = pose2xyw(msg.pose.pose)
            self.points.append(self.last_position)
            self.path.poses.append(pose_covariance_to_stamped(msg.pose))
            self.path.header.stamp = rospy.Time.now()
            self.path_pub.publish(self.path)
            return
        
        new_pos = pose2xyw(msg.pose.pose)

        if not (np.sqrt(((new_pos[0] - self.last_position[0]) ** 2) + ((new_pos[1] - self.last_position[1]) ** 2)) > self.SEPARATION.get()):
            return
        
        self.last_position = new_pos

        self.print('New point.')
        
        self.points.append(new_pos)
        self.path.poses.append(pose_covariance_to_stamped(msg.pose))
        self.path.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path)

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
        self.rate_obj.sleep()

def do_args():
    parser = ap.ArgumentParser(prog="pose_recorder.py", 
                            description="Pose Recorder",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='pose_recorder')

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