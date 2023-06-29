#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point
from pyaarapsi.core.ros_tools import q_from_yaw

class mrc:
    def __init__(self):
        rospy.init_node('pc2grid_node', anonymous=False)
        self.rate                   = rospy.Rate(1) # 1 Hz
        self.map_res                = 0.1
        self.map_height             = 600
        self.map_width              = 800
        self.z_bounds               = [1,1.6]#[0.8, 1.2]
        
        self.pc_msg                 = PointCloud2()
        self.new_msg                = False

        self.grid_info              = MapMetaData()
        self.grid_info.resolution   = self.map_res
        self.grid_info.width        = self.map_width
        self.grid_info.height       = self.map_height
        self.grid_info.origin       = Pose(position=Point(x=-40,y=-25,z=0), orientation=q_from_yaw(0))

        self.pc_sub                 = rospy.Subscriber('/cloud_pcd', PointCloud2, self.pc_cb, queue_size=1)
        self.og_pub                 = rospy.Publisher('/grid', OccupancyGrid, queue_size=1)

        rospy.loginfo('Converting pointcloud to grid.')

        while not rospy.is_shutdown():
            self.rate.sleep()

            if not (self.new_msg):
                rospy.loginfo_throttle(5, "Waiting...")
                continue

            # read in point_cloud2 as an accessible numpy array:
            genxyz          = np.array(list(point_cloud2.read_points(self.pc_msg, skip_nans=True, field_names=("x", "y", "z"))))
            low_bound_inds  = genxyz[:,2] > self.z_bounds[0]
            high_bound_inds = genxyz[:,2] < self.z_bounds[1]
            total_inds = (np.array(low_bound_inds,dtype=int) + np.array(high_bound_inds,dtype=int)) == 2
            genxyz_bounded  = genxyz[total_inds,:]
            genxy           = np.delete(genxyz_bounded, 2, axis=1) # delete z values (all will be zero regardless)
            self.convert_pc_to_grid(genxy) # do alignment conversion
            self.new_msg    = False

        rospy.loginfo('Quit received.')

    def pc_cb(self, msg):
        self.pc_msg = msg
        self.new_msg = True

    def convert_pc_to_grid(self, genxy):
        # round to nearest multiple of MAP_RESOLUTION:
        genxyfloat = np.array(self.map_res * np.around(genxy/self.map_res, 0))

        # remove duplicate point entries (only unique rows; parse through integer to avoid floating point equalities):
        genxyfloatclean = np.array(np.unique(np.around(genxyfloat*100,0), axis=0)/100.0,dtype=float)

        # build empty map (note; number of rows -> map_height)
        #empty_grid = np.ones((self.map_height, self.map_width)) * -1
        empty_grid = np.zeros((self.map_height, self.map_width))

        # populate map with pointcloud at correct cartesian coordinates
        for i in range(np.shape(genxyfloatclean)[0]):
            ixp = genxyfloatclean[i,0] - self.grid_info.origin.position.x
            iyp = genxyfloatclean[i,1] - self.grid_info.origin.position.y

            ix = int( np.round(ixp / self.map_res ))
            iy = int( np.round(iyp / self.map_res ))
            try:
                empty_grid[iy][ix] = 100
            except:
                pass #print("Position (%d,%d) is off-grid. Skipping." % (ix,iy))

        # convert to (u)int8 for ROS message transmission
        populated_grid = np.array(empty_grid, dtype=np.int8).flatten()
    
        # the rest is building the OccupancyGrid() object and populating it for transmission
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id       = "map"
        grid_msg.header.stamp          = rospy.Time.now()
        grid_msg.info                  = self.grid_info
        grid_msg.info.map_load_time   = rospy.Time.now()
        grid_msg.data                  = populated_grid

        self.og_pub.publish(grid_msg)
        rospy.loginfo_throttle(5, 'Published new grid!')

if __name__ == '__main__':
    nmrc = mrc()
