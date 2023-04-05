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
from pyaarapsi.vpr_simple.vpr_feature_tool import FeatureType, VPRImageProcessor

### Example usage:
if __name__ == '__main__':

    do_plotting = False
    make_videos = False
    gen_new_set = True

    rospy.init_node("test", log_level=rospy.DEBUG)

    PACKAGE_NAME        = 'aarapsi_robot_pack'

    FEAT_TYPE           = [FeatureType.RAW, FeatureType.PATCHNORM, FeatureType.HYBRIDNET, FeatureType.NETVLAD] # Feature Type
    REF_ROOT            = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/"
    DATABASE_PATH       = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/"
    DATABASE_PATH_FILT  = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/compressed_sets/filt"
    VIDEO_SAVE_FOLDER   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/src/aarapsi_robot_pack/vpr_simple/"

    SET_NAMES           = [ 's1_ccw_o0_e0_a0', 's1_ccw_o0_e0_a1', 's1_ccw_o0_e0_a2', 's1_cw_o0_e0_a0',\
                            's2_ccw_o0_e0_a0', 's2_ccw_o0_e1_a0', 's2_ccw_o1_e0_a0', 's2_cw_o0_e1_a0']
    SIZES               = [ 32, 64, 128, 192, 400 ]

    if do_plotting or make_videos:    
        # specific variables:
        SET_NAME        = SET_NAMES[0] #'test_library'
        IMG_DIMS        = (SIZES[3],)*2

        REF_IMG_PATHS   = [ REF_ROOT + SET_NAME + "/forward", ]#\
                                #REF_ROOT + SET_NAMES[0] + "/left", \
                                #REF_ROOT + SET_NAMES[0] + "/right", \
                                #REF_ROOT + SET_NAMES[0] + "/panorama"]
        REF_ODOM_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME + "/odometry.csv"

        # Load in dataset:
        ip          = VPRImageProcessor(ros=True, cuda=True, init_hybridnet=True, init_netvlad=True, dims=IMG_DIMS) 
        ip.npzDatabaseLoadSave(DATABASE_PATH, SET_NAMES[0], REF_IMG_PATHS, REF_ODOM_PATH, FEAT_TYPE, IMG_DIMS, do_save=True)

        vis_dict(ip.SET_DICT)
        sys.exit()
        # Perform discretisation operation:
        prefilt     = copy.deepcopy(ip.SET_DICT)
        filtered    = ip.discretise(keep='average', mode='position', metrics={'x': 0.1, 'y': 0.1, 'yaw': (10*2*np.pi/360)})
        ip.buildFullDictionary(dict_in=filtered)

    if do_plotting:
        # Visualise discretisation:
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(1,2)
        axes[0].plot(np.arange(len(filtered['times'])), filtered['times'], '.')
        axes[1].plot(filtered['odom']['position']['x'], filtered['odom']['position']['y'], '.', markersize=2)
        axes[1].plot(prefilt['odom']['position']['x'], prefilt['odom']['position']['y'], '.', markersize=2)
        axes[1].set_aspect('equal', 'box')
        plt.show()

    if make_videos:
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
        data_for_vid = prefilt # prefilt, filtered
        for name in ["RAW", "PATCHNORM", "HYBRIDNET", "NETVLAD"]:
            file_path = VIDEO_SAVE_FOLDER + name + "_feed.avi"
            if os.path.isfile(file_path): os.remove(file_path)
            vid_writer = cv2.VideoWriter(file_path, 0, fps, IMG_DIMS)
            for i in range(len(data_for_vid['times'])):
                img = prep_for_video(data_for_vid['img_feats'][name]['forward'][i], IMG_DIMS)
                if i == 0: print((img.shape, type(img.flatten()[0]), img.size))
                vid_writer.write(img)
            vid_writer.release()

        # Stack for fun:
        file_path = VIDEO_SAVE_FOLDER + "STACK_feed.avi"
        if os.path.isfile(file_path): os.remove(file_path)
        VID_DIMS = (200, 200)
        vid_writer = cv2.VideoWriter(file_path, 0, fps, VID_DIMS)
        for i in range(len(data_for_vid['times'])):
            img = stack_frames(data_for_vid['img_feats'], 'forward', i, IMG_DIMS)
            img = cv2.resize(img, VID_DIMS, interpolation=cv2.INTER_AREA)
            # BLACK BORDER:
            img = cv2.putText(img, 'R', ( 85,  95), cv2.FONT_HERSHEY_PLAIN, 1, (  0, 0, 0), 2)
            img = cv2.putText(img, 'H', (105,  95), cv2.FONT_HERSHEY_PLAIN, 1, (  0, 0, 0), 2)
            img = cv2.putText(img, 'P', ( 85, 113), cv2.FONT_HERSHEY_PLAIN, 1, (  0, 0, 0), 2)
            img = cv2.putText(img, 'N', (105, 113), cv2.FONT_HERSHEY_PLAIN, 1, (  0, 0, 0), 2)
            # GREEN FILL:
            img = cv2.putText(img, 'R', ( 85,  95), cv2.FONT_HERSHEY_PLAIN, 1, (40,250,40), 1)
            img = cv2.putText(img, 'H', (105,  95), cv2.FONT_HERSHEY_PLAIN, 1, (40,250,40), 1)
            img = cv2.putText(img, 'P', ( 85, 113), cv2.FONT_HERSHEY_PLAIN, 1, (40,250,40), 1)
            img = cv2.putText(img, 'N', (105, 113), cv2.FONT_HERSHEY_PLAIN, 1, (40,250,40), 1)
            if i == 0: print((img.shape, type(img.flatten()[0]), img.size, np.min(img), np.max(img)))
            vid_writer.write(img)
        vid_writer.release()

    if gen_new_set:
        ## Iterate through multiple:
        for SET_NAME in SET_NAMES:
            REF_IMG_PATHS   = [ REF_ROOT + SET_NAME + "/forward" ]
            REF_ODOM_PATH   = rospkg.RosPack().get_path(PACKAGE_NAME) + "/data/" + SET_NAME + "/odometry.csv"

            # generate seed_raw_image_data
            seed_raw_image_data = dict.fromkeys(REF_IMG_PATHS)
            for img_path in REF_IMG_PATHS:
                img_file_paths  = [os.path.join(img_path, f) for f in np.sort(os.listdir(img_path))]
                image_list      = []
                rospy.logdebug('Generating %s' % img_path)
                for img_file_path in tqdm(img_file_paths):
                    image_list.append(cv2.imread(img_file_path)[:, :, ::-1])
                seed_raw_image_data[img_path] = image_list

            torch.cuda.empty_cache()

            for i in SIZES:
                IMG_DIMS = (i, i)
                ip = VPRImageProcessor(ros=True, cuda=True, init_hybridnet=True, init_netvlad=True, dims=IMG_DIMS) # reinit to clean object
                ip.npzDatabaseLoadSave(DATABASE_PATH, SET_NAME, REF_IMG_PATHS, REF_ODOM_PATH, FEAT_TYPE, IMG_DIMS, do_save=True, seed_raw_image_data=seed_raw_image_data)
                
                # create filtered database:
                filtered = ip.discretise(keep='first', mode='position', metrics={'x': 0.1, 'y': 0.1, 'yaw': (10*2*np.pi/360)})
                ip.buildFullDictionary(dict_in=filtered)
                ip.save2npz(DATABASE_PATH_FILT, SET_NAME)

                ip.destroy()
                del ip

                torch.cuda.empty_cache()

            del seed_raw_image_data