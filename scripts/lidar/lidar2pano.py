#!/usr/bin/env python3
import rospy
import numpy as np
import cv2

from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, CompressedImage

from pyaarapsi.core.missing_pixel_filler import fill_swath_fast 
from pyaarapsi.core.helper_tools import Timer
from pyaarapsi.core.ros_tools import np2compressed

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

    panorama                    = np.zeros((31, pix_x, 3), dtype=np.uint8)
    print('d:', min(d), max(d), 'z:', min(z), max(z), 'r:', min(r), max(r))
    panorama[v, u, 1]           = (((d - -0.5) / float(100 - -0.5)) * 255).astype(np.uint8) #normalise_to_255(d)
    panorama[v, u, 2]           = (((z - -0.5) / float( 20 - -0.5)) * 255).astype(np.uint8) #normalise_to_255(z)
    panorama[v, u, 0]           = (((r -    0) / float(255 -    0)) * 255).astype(np.uint8) #normalise_to_255(r)
    panorama[panorama > 255]    = 255
    panorama[panorama < 0]      = 0
    panorama                    = panorama[::2] # every second row
    
    return panorama
        
def surround_coords_to_polar_coords(u, v, pix_s, mode='linear'):
    LinTerm = ((pix_s-1)/ 2) - (((v - min(v)) / (max(v) - min(v))) * ((pix_s-1)/ 2))
    if mode == 'linear':
        R = LinTerm
    elif mode == 'log10':
        R = LinTerm * np.log10(LinTerm + 1) * pix_s / np.log10(pix_s + 1)
    else:
        raise Exception('Unknown mode "%s".' % mode)

    t = (((u - min(u)) / (max(u) - min(u))) * (np.pi * 2)) + (np.pi/2)

    px = ((pix_s-1)/ 2) + (np.cos(t) * R)
    py = ((pix_s-1)/ 2) + (np.sin(t) * R)

    px = np.rint(px).astype(int)
    py = np.rint(py).astype(int)

    return px, py

def surround_to_polar_coords(surround, pix_s, mode='linear'):
    u = np.repeat([np.arange(surround.shape[1])],surround.shape[0],0).flatten()
    v = np.repeat(np.arange(surround.shape[0]),surround.shape[1],0)

    px, py = surround_coords_to_polar_coords(u, v, pix_s, mode=mode)

    return px, py, u, v

def surround_to_polar_img(surround, pix_s, mode='linear'):
    px,py,su,sv             = surround_to_polar_coords(surround, pix_s, mode=mode)
    polarimg                = np.ones((pix_s, pix_s, 3), dtype=np.uint8) * 255
    polarimg[py, px, :]     = surround[sv, su, :]
    return polarimg

class mrc:
    def __init__(self, node_name='lidar2panorama', anon=True, rate_num=20.0):
        rospy.init_node(node_name, anonymous=anon)
        rospy.loginfo('Starting %s node.' % (node_name))

        ## Parse all the inputs:
        self.rate_num       = rate_num # Hz
        self.rate_obj       = rospy.Rate(self.rate_num)

        self.lidar_topic    = '/velodyne_points'
        self.front_topic    = '/lidar/front/compressed'
        self.pano_topic     = '/lidar/pano/compressed'
        self.polar_topic    = '/lidar/polar/compressed'

        self.pix_x          = 800
        self.pix_y          = 90
        self.pix_s          = 128

        self.lidar_msg      = PointCloud2()
        self.new_lidar      = False
    
        self.lidar_sub      = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1)
        self.front_pub      = rospy.Publisher(self.front_topic, CompressedImage, queue_size=1)
        self.pano_pub       = rospy.Publisher(self.pano_topic, CompressedImage, queue_size=1)
        self.polar_pub      = rospy.Publisher(self.polar_topic, CompressedImage, queue_size=1)

    def lidar_callback(self, msg):
        self.lidar_msg      = msg
        self.new_lidar      = True

    def loop_contents(self):

        pointcloud_xyzi = np.array(list(
            point_cloud2.read_points(self.lidar_msg, skip_nans=True, field_names=("x", "y", "z", "intensity")) #, "ring", "time"
            ))
        pc_xyzi = pointcloud_xyzi[pointcloud_xyzi[:,2] > -0.5] # ignore erroneous rows when they appear
        x = pc_xyzi[:,0]
        y = pc_xyzi[:,1]
        z = pc_xyzi[:,2]
        r = pc_xyzi[:,3] # intensity, reflectance

        lidar_pano          = lidar_to_surround_img(x, y, z, r, self.pix_x, self.pix_y)
        lidar_pano_clean    = fill_swath_fast(lidar_pano)
        lidar_pano_resize   = cv2.resize(lidar_pano_clean, (self.pix_x, self.pix_y), interpolation = cv2.INTER_AREA)
        lidar_plr_img       = surround_to_polar_img(lidar_pano_resize, self.pix_s, mode='linear')
        lidar_front         = lidar_pano_resize[:,int((self.pix_x/5)*2):int((self.pix_x/5)*3),:]

        _new_pano = np2compressed(lidar_pano_resize, True)
        _new_polar = np2compressed(lidar_plr_img, True)
        _new_front = np2compressed(lidar_front, True)
        _new_pano.header.frame_id = 'velodyne'
        _new_polar.header.frame_id = 'velodyne'
        _new_front.header.frame_id = 'velodyne'

        self.front_pub.publish(_new_front)
        self.pano_pub.publish(_new_pano)
        self.polar_pub.publish(_new_polar)

    def main(self):

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
