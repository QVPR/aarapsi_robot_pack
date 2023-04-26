#!/usr/bin/env python3

# import rospkg
import rospkg
import rospy
import numpy as np
import copy
import cv2
import os
import sys
import torch
from tqdm.auto import tqdm

from pyaarapsi.core.helper_tools import vis_dict
from pyaarapsi.core.ros_tools import roslogger, LogType
from pyaarapsi.core.enum_tools import enum_name
from pyaarapsi.vpr_simple.new_vpr_feature_tool import FeatureType, VPRImageProcessor

### Example usage:
if __name__ == '__main__':

    rospy.init_node("test", log_level=rospy.DEBUG)

    FEAT_TYPES          = [FeatureType.RAW, FeatureType.PATCHNORM, FeatureType.HYBRIDNET, FeatureType.NETVLAD, FeatureType.ROLLNORM] # Feature Types
    REF_ROOT            = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/data/"
    VIDEO_SAVE_FOLDER   = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/data/videos/"
    NPZ_DBP             = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/data/compressed_sets/"
    NPZ_DBP_FILT        = NPZ_DBP + "/filt"
    try:
        BAG_DBP         = os.environ['AARAPSI_ONEDRIVE_ROOT'] + '/3.0 Data/3.1 rosbags/improved'
    except:
        roslogger("Environment variable AARAPSI_ONEDRIVE_ROOT not set, defaulting to HOME as root instead.", LogType.WARN, ros=True)
        roslogger("AARAPSI_ONEDRIVE_ROOT should specify directory to (and including) aarapsi_home, i.e. /path/example/aarapsi_home.", LogType.WARN, ros=True)
        BAG_DBP         = os.environ['HOME'] + '/OneDrive/aarapsi_home/3.0 Data/3.1 rosbags/improved'

    SET_NAMES           = [ 's1_ccw_o0_e0_a0', 's1_ccw_o0_e0_a1', 's1_ccw_o0_e0_a2', 's1_cw_o0_e0_a0',\
                            's2_ccw_o0_e0_a0', 's2_ccw_o0_e1_a0', 's2_ccw_o1_e0_a0', 's2_cw_o0_e1_a0']
    SIZES               = [ 32, 64, 128, 192, 400 ]


    BAG_NAME            = SET_NAMES[0] + '.bag'
    IMG_DIMS            = (SIZES[1],)*2
    FEAT_TYPES_IN       = FEAT_TYPES
    SAMPLE_RATE         = 10.0 # Hz
    ODOM_TOPIC          = '/odom/filtered'
    IMG_TOPICS          = ['/ros_indigosdk_occam/image0/compressed']

    dataset_params      = dict( bag_name=BAG_NAME, npz_dbp=NPZ_DBP, bag_dbp=BAG_DBP, \
                                odom_topic=ODOM_TOPIC, img_topics=IMG_TOPICS, \
                                sample_rate=SAMPLE_RATE, img_dims=IMG_DIMS, ft_types=FEAT_TYPES_IN, filters={})

    # Initialise ImageProcessor:
    ip                  = VPRImageProcessor(bag_dbp=BAG_DBP, npz_dbp=NPZ_DBP, dataset=dataset_params, \
                                            ros=True, init_netvlad=True, init_hybridnet=True, cuda=True, \
                                            try_gen=True, use_tqdm=True, autosave=True)
    
    vis_dict(ip.dataset)

    # Visualise forward camera feed:
    def prep_for_video(img, dims, dstack=True):
        _min      = np.min(img)
        _max      = np.max(img)
        _img_norm = (img - _min) / (_max - _min)
        _img_uint = np.array(_img_norm * 255, dtype=np.uint8)
        _img_dims = np.reshape(_img_uint, dims)
        if dstack: return np.dstack((_img_dims,)*3)
        return _img_dims

    def stack_frames(dict_in, sub_dict_key, index, dims):
        # assume enough to make a 2x2
        frames_at_index = []
        for key in dict_in:
            frames_at_index.append(prep_for_video(dict_in[key][sub_dict_key][index], dims, dstack=False))
        stacked_frame = np.concatenate((np.concatenate((frames_at_index[0],frames_at_index[1])),\
                                        np.concatenate((frames_at_index[2],frames_at_index[3]))), axis=1)
        dstack_heap = np.dstack((stacked_frame,)*3)
        return dstack_heap

    fps = 40.0
    data_for_vid = ip.dataset['dataset']
    for name in enum_name(FEAT_TYPES_IN):
        file_path = VIDEO_SAVE_FOLDER + name + "_feed.avi"
        if os.path.isfile(file_path): os.remove(file_path)
        vid_writer = cv2.VideoWriter(file_path, 0, fps, IMG_DIMS)
        for i in range(len(data_for_vid['time'])):
            img = prep_for_video(data_for_vid[name][i], IMG_DIMS)
            vid_writer.write(img)
        vid_writer.release()
