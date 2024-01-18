#!/usr/bin/env python3

import rospy 
import ros_numpy
import tf2_ros
import argparse as ap
import numpy as np
import sys

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped

from pyaarapsi.core.ros_tools       import NodeState, roslogger, LogType, q_from_yaw
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.core.argparse_tools  import check_enum, check_positive_float
from pyaarapsi.core.transforms      import Transform_Builder, apply_homogeneous_transform
from pyaarapsi.vpr_classes.base     import Super_ROS_Class

NODE_NAME = 'pcrotate'
NAMESPACE = ''
ANON      = True

'''

Point Cloud Rotate

Subscribe, rotate, and re-publish a pointcloud at a fixed rate.

'''

class Main_ROS_Class(Super_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(**kwargs)
        self.init_vars(**kwargs)
        self.init_rospy(**kwargs)

        self.node_ready()
    
    def init_vars(self, speed, *args, **kwargs):
        super().init_vars(*args, **kwargs)

        self.speed          = speed
        self.msg            = PointCloud2()
        self.msg_received   = False
        self.rotation       = 0.0
        self.pc_np          = None
        self.tf             = Transform_Builder()
    
    def init_rospy(self, speed, *args, **kwargs):
        super().init_rospy(*args, **kwargs)

        self.pc_pub         = self.add_pub('/cloud_pcd_rotate', PointCloud2, queue_size=1)
        self.pc_sub         = rospy.Subscriber('/cloud_pcd', PointCloud2, self.pc_cb, queue_size=1)

        #self.tf_bc          = tf2_ros.TransformBroadcaster()

    def pc_cb(self, msg: PointCloud2):
        self.msg                    = msg
        self.msg_received           = True
        self.msg.header.frame_id    = 'pcrotate'
        self.pc_np                  = None
        #self.pc_pub.publish(self.msg)

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
        if not (self.msg_received and self.main_ready): # denest
            self.print("Waiting.", LogType.DEBUG, throttle=60) # print every 60 seconds
            rospy.sleep(0.005)
            return
        
        if self.pc_np is None:
            _pc_np = np.array(list(point_cloud2.read_points(self.msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))))
            self.pc_np = np.array(_pc_np[:,0:4], dtype=np.float32)
            self.pc_np -= np.mean(self.pc_np,axis=0)
            self.pc_structure = np.zeros(len(self.pc_np), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('intensity', np.float32),
            ])

        try:
            self.pc_structure['x'], self.pc_structure['y'], self.pc_structure['z'] = \
                apply_homogeneous_transform(self.tf.rotate(0,0,self.speed/self.RATE_NUM.get()).get(), \
                                            self.pc_np[:,0], self.pc_np[:,1], self.pc_np[:,2], False)
            self.pc_structure['intensity'] = self.pc_np[:,3]

            pc_msg = ros_numpy.msgify(PointCloud2, self.pc_structure, stamp=rospy.Time.now(), frame_id='map')
            self.pc_pub.publish(pc_msg)
        except:
            pass

        self.rate_obj.sleep()

        ## I wanted to use transforms, but the update rate in RViz seems too slow... :-(
        # t = TransformStamped()
        # t.header.stamp = rospy.Time.now()
        # t.header.frame_id = 'map'
        # t.child_frame_id = 'pcrotate'
        # t.transform.translation.x = 0.0
        # t.transform.translation.y = 0.0
        # t.transform.translation.z = 0.0
        # t.transform.rotation      = q_from_yaw(self.rotation)

        #self.rotation = self.rotation + 0.01#(self.speed / self.RATE_NUM.get())

        #self.tf_bc.sendTransform(t)

def do_args():
    parser = ap.ArgumentParser(prog="pcrotate.py", 
                            description="Point Cloud Rotate",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser.add_argument('--log-level', '-V',  type=lambda x: check_enum(x, LogType), default=2,    help="Specify ROS log level (default: %(default)s).")
    parser.add_argument('--rate_num',  '-r',  type=check_positive_float,             default=10.0, help='Specify node rate (default: %(default)s).')
    parser.add_argument('--speed',     '-s',  type=check_positive_float,             default=0.18, help='Specify rotation speed (default: %(default)s).')
    
    # Parse args...
    args = vars(parser.parse_known_args()[0])
    args.update({'node_name': NODE_NAME, 'namespace': NAMESPACE, 'anon': True, 'reset': True})
    return args

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