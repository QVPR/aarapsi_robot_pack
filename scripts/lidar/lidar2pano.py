#!/usr/bin/env python3

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

import numpy as np
import cv2
import matplotlib
matplotlib.use("Qt5agg")
from matplotlib import pyplot as plt
from sensor_msgs.msg import CompressedImage
from pyaarapsi.core.missing_pixel_filler import fill_swath_fast 
from pyaarapsi.core.helper_tools import Timer

class mrc:
    def __init__(self, node_name='lidar2panorama', anon=True, rate_num=20.0):
        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        self.lidar_topic    = '/velodyne_points'
        self.image_topic    = '/ros_indigosdk_occam/image%d/compressed'

        self.lidar_sub      = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1)

        self.img_subs       = []
        self.img_msgs       = [None]*5
        self.new_imgs       = [False]*5
    
        for i in range(5):
            self.img_subs.append(rospy.Subscriber(self.image_topic % (i), CompressedImage, self.image_callback, (i)))

        ## Parse all the inputs:
        self.rate_num       = rate_num # Hz
        self.rate_obj       = rospy.Rate(self.rate_num)

        self.lidar_msg      = PointCloud2()
        self.new_lidar_msg  = False

    def image_callback(self, msg, id):
        self.img_msgs[id]   = msg
        self.new_imgs[id]   = True

    def lidar_callback(self, msg):
        self.lidar_msg      = msg
        self.new_lidar_msg  = True

    def main_loop(self, lidar_h, sight_h, tdva, lidar_plr_h, sight_plr_h, pix_x, pix_y, pix_s):

        timer = Timer()
        timer.add()
        if not self.new_lidar_msg and not all(self.new_imgs):
            return
        
        self.new_lidar_msg = False
        self.new_imgs = [False]*len(self.new_imgs)

        # Hacked for ROS + our LiDAR from:
        # https://adioshun.gitbooks.io/pcl_snippet/content/3D-Point-cloud-to-2D-Panorama-view.html
        def size(a, dims = [], ind=0):
            if ind == 0:
                dims = []
            if not isinstance(a, list):
                return dims# quick exit
            # if list:
            dims.append(len(a))
            if len(a) > 0:
                dims = size(a[0], dims, ind + 1) # pass in current element
            return dims

        def normalise_to_255(a):
            return (((a - min(a)) / float(max(a) - min(a))) * 255).astype(np.uint8)

        def lidar_to_surround_coords(x, y, z, d, pix_x, pix_y):
            u =   np.arctan2(x, y) *(180/np.pi)
            v = - np.arctan2(z, d) *(180/np.pi)
            u = (u + 90) % 360
            
            u = ((u - min(u)) / (max(u) - min(u))) * (pix_x - 1)
            v = ((v - min(v)) / (max(v) - min(v))) * (pix_y - 1)

            u = np.rint(u).astype(int)
            v = np.rint(v).astype(int)

            return u, v
        
        def lidar_to_surround_img(x, y, z, r, pix_x, pix_y):
            d = np.sqrt(x ** 2 + y ** 2)  # map distance relative to origin
            u,v = lidar_to_surround_coords(x, y, z, d, pix_x, 31) # 31 = num lasers + num lasers - 1 (number of rows due to lasers + number of empty rows)

            panorama                = np.zeros((31, pix_x, 3), dtype=np.uint8)
            panorama[v, u, 1]       = normalise_to_255(d)
            panorama[v, u, 2]       = normalise_to_255(z)
            panorama[v, u, 0]       = normalise_to_255(r)
            panorama                = panorama[::2]
            
            return panorama
                
        def surround_coords_to_polar_coords(u, v, pix_s, mode='linear'):
            LinTerm = ((pix_s-1)/ 2) - (((v - min(v)) / (max(v) - min(v))) * ((pix_s-1)/ 2))
            if mode == 'linear':
                R = LinTerm
            elif mode == 'log10':
                R = LinTerm * np.log10(R + 1) * pix_s / np.log10(pix_s + 1)

            t = (((u - min(u)) / (max(u) - min(u))) * (np.pi * 2)) + (np.pi/2)

            px = ((pix_s-1)/ 2) + (np.cos(t) * R)
            py = ((pix_s-1)/ 2) + (np.sin(t) * R)

            px = np.rint(px).astype(int)
            py = np.rint(py).astype(int)

            return px, py
        
        def surround_to_polar_coords(surround, pix_s, mode='linear'):
            u = np.repeat([np.arange(surround.shape[1])],surround.shape[0],0).flatten()
            v = np.repeat(np.arange(surround.shape[0]),surround.shape[1],0)

            px, py = surround_coords_to_polar_coords(u, v, pix_s, mode='linear')

            return px, py, u, v
        
        def surround_to_polar_img(surround, pix_s, mode='linear'):
            px,py,su,sv             = surround_to_polar_coords(surround, pix_s, mode='linear')
            polarimg                = np.zeros((pix_s, pix_s, 3), dtype=np.uint8)
            polarimg[py, px, :]     = surround[sv, su, :]
            return polarimg

        pointcloud_xyzi = np.array(list(point_cloud2.read_points(self.lidar_msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))))#, "ring", "time"
        pc_xyzi = pointcloud_xyzi[pointcloud_xyzi[:,2] > -0.5] # ignore erroneous rows when they appear
        x = np.array(pc_xyzi[:,0])
        y = np.array(pc_xyzi[:,1])
        z = np.array(pc_xyzi[:,2])
        r = np.array(pc_xyzi[:,3]) # intensity, reflectance

        lidar_pano = lidar_to_surround_img(x, y, z, r, pix_x, pix_y)
        lidar_pano_clean = fill_swath_fast(lidar_pano)
        lidar_pano_resize = cv2.resize(lidar_pano_clean, (pix_x, pix_y), interpolation = cv2.INTER_AREA)

        lidar_plr_img = surround_to_polar_img(lidar_pano_resize, pix_s, mode='log10')

        x0, y0 = np.where(np.sum(lidar_plr_img,2)==0)
        lidar_plr_img[x0,y0,:] = [255,255,255]

        #lidar_plr_img_clean = fill_swath_fast(lidar_plr_img)

        lidar_h.set_data(lidar_pano_resize)
        #sight_h.set_data()

        lidar_plr_h.set_data(lidar_plr_img)
        #sight_plr_h.set_data()

        tdva.cla()
        tdva.scatter(x, y, c=z)
        tdva.set_aspect('equal')
        tdva.set_xlim(-15, 15)
        tdva.set_ylim(-15, 15)

        timer.add()
        fig.canvas.draw()
        plt.pause(0.0001)

        timer.add()
        timer.addb()
        timer.show("main_loop")

if __name__ == '__main__':
    try:
        nmrc = mrc()

        fig, axes = plt.subplots(3, 3, figsize=(15,8))
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
        gs_top = axes[0, 0].get_gridspec()
        gs_mid = axes[1, 0].get_gridspec()
        for ax in axes[0:2,:].flatten():
            ax.remove()
        ax_top = fig.add_subplot(gs_top[0,:])
        ax_mid = fig.add_subplot(gs_mid[1,:])
        pix_x = 800
        pix_y = 90
        pix_s = 128

        image1 = np.zeros((pix_y, pix_x, 3))
        lidar_h = ax_top.imshow(image1, cmap='gist_rainbow', vmin=0, vmax=1)
        sight_h = ax_mid.imshow(image1, cmap='gist_rainbow', vmin=0, vmax=1)

        image2 = np.zeros((pix_s, pix_s, 3))
        lidar_plr_h = axes[2,1].imshow(image2, cmap='gist_rainbow', vmin=0, vmax=1)
        sight_plr_h = axes[2,2].imshow(image2, cmap='gist_rainbow', vmin=0, vmax=1)
        
        fig.show()
        timer = Timer()
        while not rospy.is_shutdown():
            nmrc.rate_obj.sleep()
            nmrc.main_loop(lidar_h, sight_h, axes[2,0], lidar_plr_h, sight_plr_h, pix_x, pix_y, pix_s)

            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass
