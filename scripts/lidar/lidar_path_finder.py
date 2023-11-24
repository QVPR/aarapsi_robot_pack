#!/usr/bin/env python3
import rospy
import numpy as np
import cv2

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from pyaarapsi.core.helper_tools import plt_pause
from matplotlib import pyplot as plt

class mrc:
    def __init__(self, node_name='lidar2panorama', anon=True, rate_num=20.0):
        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        ## Parse all the inputs:
        self.rate_num       = rate_num # Hz
        self.rate_obj       = rospy.Rate(self.rate_num)

        self.lidar_topic    = '/velodyne_points'

        self.pix_x          = 800
        self.pix_y          = 90
        self.pix_s          = 128

        self.map_res        = 0.05
        self.map_width_m    = 30
        self.map_height_m   = 30
        self.map_height     = int(np.ceil(self.map_height_m / self.map_res)) + 2
        self.map_width      = int(np.ceil(self.map_width_m / self.map_res)) + 2

        if not self.map_height % 2:
            self.map_height += 1
        if not self.map_width % 2:
            self.map_width += 1

        self.lidar_msg      = PointCloud2()
        self.new_lidar      = False
    
        self.lidar_sub      = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1)

    def lidar_callback(self, msg):
        self.lidar_msg      = msg
        self.new_lidar      = True

    def convert_pc_to_grid(self, xy_arr):

        assert xy_arr.shape[1] == 2

        # round xy pairs to nearest multiple of MAP_RESOLUTION:
        xy_rounded = np.array(self.map_res * np.around(xy_arr/self.map_res, 0))
        
        # remove duplicate point entries (only unique rows; parse through integer to avoid floating point equalities):
        xy_clean = np.array(np.unique(np.around(xy_rounded*100,0), axis=0)/100.0,dtype=float)

        grid = np.zeros((self.map_height, self.map_width), dtype=int)

        x_offset = int(self.map_width / 2.0)
        y_offset = int(self.map_height / 2.0)

        # populate map with pointcloud at correct cartesian coordinates
        for xy_pair in xy_clean:

            ix = int( np.round(xy_pair[0] / self.map_res )) + x_offset
            iy = int( np.round(xy_pair[1] / self.map_res )) + y_offset

            grid[iy-1:iy+1,ix-1:ix+1] = 1
        
        return np.flipud(grid)

    def loop_contents(self):

        pointcloud_xyzi = np.array(list(
            point_cloud2.read_points(self.lidar_msg, skip_nans=True, field_names=("x", "y", "z", "intensity")) #, "ring", "time"
            ))
        pc_xyzi = pointcloud_xyzi[pointcloud_xyzi[:,2] > -0.5] # ignore erroneous rows when they appear
        x = pc_xyzi[:,0]
        y = pc_xyzi[:,1]
        z = pc_xyzi[:,2]
        r = pc_xyzi[:,3] # intensity, reflectance

        _range = np.sqrt(np.square(x) + np.square(y))

        [ax.clear() for ax in self.axes]

        _good = (z > 0.1) & (z < 1) & (_range<15)

        #self.axes[0].plot(x[_good], y[_good], 'g.')
        self.axes[0].imshow(self.convert_pc_to_grid(np.transpose(np.stack([x[_good], y[_good]],axis=0))), cmap='binary')
        self.axes[1].plot(_range)
        self.axes[2].plot(-np.arctan2(y[_good],x[_good]), _range[_good], 'b.')
        #self.axes[2].hist()


        #self.axes[0].set_xlim(-15,15)
        #self.axes[0].set_ylim(-15,15)
        plt_pause(0.01,self.fig)
       

    def main(self):

        self.fig, self.axes = plt.subplots(1,3,figsize=(15,5))
        self.axes[0].set_aspect('equal')
        plt.show(block=False)

        while not rospy.is_shutdown():
            if not self.new_lidar:
                rospy.sleep(0.005)
                continue

            self.rate_obj.sleep()
            self.new_lidar = False

            self.loop_contents()

if __name__ == '__main__':
    try:
        nmrc = mrc()
        nmrc.main()
            
        print("Exit state reached.")
    except rospy.ROSInterruptException:
        print("Node terminated.")
