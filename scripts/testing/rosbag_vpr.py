#!/usr/bin/env python3

import rospy
import rospkg
from fastdist import fastdist
import os, sys
import numpy as np
import signal
from pyaarapsi.vpr_simple.vpr_dataset_tool import VPRDatasetProcessor
from pyaarapsi.vpr_simple.svm_model_tool import SVMModelProcessor
from pyaarapsi.core.helper_tools import try_load_var, save_var

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

PYPATH = "/home/claxton/python_testing"

def sigint_cb(sig, frame):
    print("\nSIGINT Received. Exitting.")
    plt.close('all')
    sys.exit()

signal.signal(signal.SIGINT, sigint_cb)

if __name__ == "__main__":
    rospy.init_node("test_node", log_level=rospy.DEBUG)

    bag_name    = "run1_fix"
    npz_dbp     = "/data/compressed_sets"
    svm_dbp     = "/cfg/svm_models"
    bag_dbp     = "/data/rosbags"
    odom_topic  = "/odom/true"
    img_topics  = ["/ros_indigosdk_occam/stitched_image0/compressed"]
    sample_rate = 5.0
    ft_type     = "RAW"
    ft_types    = [ft_type]
    img_dims    = [32,32] #[64,64] #[128,128] #[192,192]

    run1_dataset_dict = dict(bag_name="run1_fix", npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                            odom_topic=odom_topic, img_topics=img_topics, sample_rate=sample_rate, \
                            ft_types=ft_types, img_dims=img_dims, filters='{}')
    run2_dataset_dict = dict(bag_name="run2_fix", npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                            odom_topic=odom_topic, img_topics=img_topics, sample_rate=sample_rate, \
                            ft_types=ft_types, img_dims=img_dims, filters='{}')
    run3_dataset_dict = dict(bag_name="run3_fix", npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                            odom_topic=odom_topic, img_topics=img_topics, sample_rate=sample_rate, \
                            ft_types=ft_types, img_dims=img_dims, filters='{}')
    run4_dataset_dict = dict(bag_name="run4_fix", npz_dbp=npz_dbp, bag_dbp=bag_dbp, \
                            odom_topic=odom_topic, img_topics=img_topics, sample_rate=sample_rate, \
                            ft_types=ft_types, img_dims=img_dims, filters='{}')
    
    svm1_model_dict   = dict(ref=run3_dataset_dict, qry=run4_dataset_dict, bag_dbp=bag_dbp, npz_dbp=npz_dbp, svm_dbp=svm_dbp)

    ip1 = VPRDatasetProcessor(run1_dataset_dict, try_gen=True, ros=True, autosave=True, init_netvlad=True,  init_hybridnet=True,  cuda=True, use_tqdm=True)
    ip2 = VPRDatasetProcessor(None,              try_gen=True, ros=True, autosave=True, init_netvlad=False, init_hybridnet=False, cuda=True, use_tqdm=True)

    ip2.pass_nns(ip1)

    ip2.load_dataset(run2_dataset_dict, try_gen=True)

    svm1 = SVMModelProcessor(ros=True)
    svm1.generate_model(**svm1_model_dict)

    #dvc1 = try_load_var(PYPATH, "dvc1")
    #if dvc1 is None:
    dvc1 = fastdist.matrix_to_matrix_distance(ip1.dataset['dataset'][ft_type], ip2.dataset['dataset'][ft_type], fastdist.euclidean, "euclidean")
    #save_var(PYPATH, dvc1, "dvc1")

    euc_dists = fastdist.matrix_to_matrix_distance( np.transpose(np.matrix([ip1.dataset['dataset']['px'], ip1.dataset['dataset']['py']])), np.transpose(np.matrix([ip2.dataset['dataset']['px'], ip2.dataset['dataset']['py']])), fastdist.euclidean, "euclidean")
    
    match_inds = np.argmin(dvc1, 1)
    true_inds = np.argmin(euc_dists, 1)
    true_dists = np.min(euc_dists, 1)

    error_inds = np.min(np.array([-1 * abs(match_inds - true_inds) + len(match_inds), abs(match_inds - true_inds)]), axis=0)

    error_dists = np.sqrt(np.square(ip2.dataset['dataset']['px'][match_inds] - ip2.dataset['dataset']['px'][true_inds]) + np.square(ip2.dataset['dataset']['py'][match_inds] - ip2.dataset['dataset']['py'][true_inds]))

    arr1 = []
    for i in dvc1:
        (y_pred_rt, y_zvalues_rt, [factor1_qry, factor2_qry], prob) = svm1.predict(i)
        arr1.append((y_pred_rt, y_zvalues_rt, factor1_qry, factor2_qry, prob))
    arr1 = np.array(arr1, dtype=object)

    colours = np.transpose(np.array([1-np.array(arr1[:,0], dtype=int), np.array(arr1[:,0], dtype=int), np.zeros(len(match_inds))]))

    print("Percent 'Good': %0.4f%%" % (100 * (np.sum(arr1[:,0])/len(match_inds))))

    good_points = np.array(arr1[:,0], dtype=bool)
    true_errors     = error_dists[good_points]
    false_errors    = error_dists[good_points == False]

    fig, axes = plt.subplots(1, 2)

    axes[0].plot(np.arange(len(match_inds)), true_dists,  'b',  linewidth=0.5)
    axes[0].plot(np.arange(len(match_inds)), error_dists,  'k',  linewidth=0.5, linestyle='dotted', alpha=0.5)
    axes[0].scatter(np.arange(len(match_inds))[good_points], error_dists[good_points], s=1.0, color=(0,1,0))
    axes[0].scatter(np.arange(len(match_inds))[good_points == False], error_dists[good_points == False], s=1.0, color=(1,0,0))
    axes[0].plot(np.arange(len(match_inds)), np.ones(len(match_inds))*0.2, 'm', linestyle='dotted')
    axes[0].plot(np.arange(len(match_inds)), np.ones(len(match_inds))*0.5, 'm', linestyle='dashdot')
    axes[0].plot(np.arange(len(match_inds)), np.ones(len(match_inds))*1.0, 'm', linestyle='dashed')
    axes[0].plot(np.arange(len(match_inds)), np.ones(len(match_inds))*2.0, 'm', linestyle='solid')
    axes[0].set_ylim(-0.1, 4)
    #axes[0].set_yscale('log')

    axes[1].hist(true_errors,  bins=50, range=[0, 10], color=( 85/255,217/255, 91/255), alpha=0.5)
    axes[1].hist(false_errors, bins=50, range=[0, 10], color=(219/255, 93/255, 85/255), alpha=0.5)
    plt.show()
