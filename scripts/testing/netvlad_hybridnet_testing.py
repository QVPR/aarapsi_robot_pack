#!/usr/bin/env python3

import rospkg
import numpy as np
import os
import sys
import time
import rospy
import torch
from pyaarapsi.vpr_simple.vpr_feature_tool import VPRImageProcessor, FeatureType
from pyaarapsi.core.helper_tools import vis_dict
from pyaarapsi.vpr_classes.netvlad import NetVLAD_Container
from pyaarapsi.vpr_classes.hybridnet import HybridNet_Container

gpu_exists      = torch.cuda.is_available()
cam             = 'forward'
do_netvlad      = True
do_hybridnet    = True

rospy.init_node('test', log_level=rospy.DEBUG)
rospy.loginfo("\tTest Settings: \n\t\t\t\t- GPU Detected: %r\n\t\t\t\t- NetVLAD Enabled: %r\n\t\t\t\t- HybridNet Enabled: %r" % (gpu_exists, do_netvlad, do_hybridnet))
#################### Open test data ###################################

PACKAGE_PATH        = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__)))

FEAT_TYPE           = [FeatureType.RAW, FeatureType.PATCHNORM, FeatureType.HYBRIDNET, FeatureType.NETVLAD] # Feature Type
REF_ROOT            = PACKAGE_PATH + "/data/"
DATABASE_PATH       = PACKAGE_PATH + "/data/compressed_sets/"
DATABASE_PATH_FILT  = PACKAGE_PATH + "/data/compressed_sets/filt"

#SET_NAMES           = [ 's1_ccw_o0_e0_a0', 's1_ccw_o0_e0_a1', 's1_ccw_o0_e0_a2', 's1_cw_o0_e0_a0', \
#                        's2_ccw_o0_e0_a0', 's2_ccw_o0_e1_a0', 's2_ccw_o1_e0_a0', 's2_cw_o0_e1_a0']
SET_NAMES           = ['test_library']
SIZES               = [ 32, 64, 128, 192]

REF_IMG_PATHS       = [ REF_ROOT + SET_NAMES[0] + "/forward", ]#\
                        #REF_ROOT + SET_NAMES[0] + "/left", \
                        #REF_ROOT + SET_NAMES[0] + "/right", \
                        #REF_ROOT + SET_NAMES[0] + "/panorama"]
REF_ODOM_PATH       = PACKAGE_PATH + "/data/" + SET_NAMES[0] + "/odometry.csv"
IMG_DIMS            = (SIZES[3],)*2

ip = VPRImageProcessor(cuda=True, init_hybridnet=True, init_netvlad=True, dims=IMG_DIMS) 
ip.npzDatabaseLoadSave(DATABASE_PATH, SET_NAMES[0], REF_IMG_PATHS, REF_ODOM_PATH, FEAT_TYPE, IMG_DIMS, do_save=False)
vis_dict(ip.SET_DICT)
#ip.npzLoader(DATABASE_PATH, SET_NAMES[0], IMG_DIMS)

ind                 = int(np.random.rand() * (len(ip.SET_DICT['times']) - 1))
ft_raw              = ip.SET_DICT['img_feats']['RAW'][cam]
img2d               = np.reshape(ft_raw[ind], IMG_DIMS).astype(np.uint8)
img3d               = np.dstack((img2d,)*3)

ref_path            = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/data/' + SET_NAMES[0] + '/' + cam
img_path            = ref_path + '/frame_id_000000.png'

#################### Prep test examples ##############################

torch.cuda.empty_cache()

type_test_data = [ref_path, img_path, [img_path, img_path], (ft_raw, IMG_DIMS), img2d, img3d, [img2d, img2d], [img3d, img3d]]
type_test_labs = ['dir path', 'img path', 'list img paths', 'features', '2d img', '3d img', '2d img list', '3d img list']
type_test_dict = dict(zip(type_test_labs, type_test_data))

#################### HybridNet test example ############################
if do_hybridnet:
    rospy.loginfo("HybridNet Tests")
    hybridnet_obj = HybridNet_Container(cuda=gpu_exists, logger=lambda x: (x), dims=IMG_DIMS)

    for key in type_test_dict:
        if isinstance(type_test_dict[key], np.ndarray): dims = str(type_test_dict[key].shape)
        elif isinstance(type_test_dict[key], list): dims = str(len(type_test_dict[key]))
        elif isinstance(type_test_dict[key], tuple): dims = str((type_test_dict[key][0].shape, type_test_dict[key][1]))
        else: dims = 'Object'
        t1 = rospy.Time.now().to_sec()
        if isinstance(type_test_dict[key], tuple):
            qry_ftrs = hybridnet_obj.getFeat(type_test_dict[key][0], dims=type_test_dict[key][1], use_tqdm=False)
        else:
            qry_ftrs = hybridnet_obj.getFeat(type_test_dict[key], use_tqdm=False)
        t2 = rospy.Time.now().to_sec()

        rospy.logdebug("Key: {%s} Class: %s Size: {%s} Execution Time: %0.4fs" % (key, str(type(type_test_dict[key])), dims, (t2-t1)))

#################### NetVLAD test example ##############################
if do_netvlad:
    rospy.loginfo("NetVLAD Tests")
    netvlad_obj = NetVLAD_Container(cuda=gpu_exists, ngpus=int(gpu_exists), logger=lambda x: (x), dims=IMG_DIMS)

    for key in type_test_dict:
        if isinstance(type_test_dict[key], np.ndarray): dims = str(type_test_dict[key].shape)
        elif isinstance(type_test_dict[key], list): dims = str(len(type_test_dict[key]))
        elif isinstance(type_test_dict[key], tuple): dims = str((type_test_dict[key][0].shape, type_test_dict[key][1]))
        else: dims = 'Object'

        t1 = rospy.Time.now().to_sec()
        if isinstance(type_test_dict[key], tuple):
            qry_ftrs = netvlad_obj.getFeat(type_test_dict[key][0], dims=type_test_dict[key][1], use_tqdm=False)
        else:
            qry_ftrs = netvlad_obj.getFeat(type_test_dict[key], use_tqdm=False)
        t2 = rospy.Time.now().to_sec()

        rospy.logdebug("Key: {%s} Class: %s Size: {%s} Execution Time: %0.4fs" % (key, str(type(type_test_dict[key])), dims, (t2-t1)))