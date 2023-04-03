#!/usr/bin/env python3

import numpy as np
import os
from enum import Enum
from pathlib import Path
import datetime

try:
    import rospy # used for printing
except:
    pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearnex import patch_sklearn # Package for speeding up sklearn 
patch_sklearn()

from sklearn import svm
from sklearn.preprocessing import StandardScaler

from aarapsi_robot_pack.core.enum_tools import enum_name
from aarapsi_robot_pack.core.file_system_tools import scan_directory
from aarapsi_robot_pack.vpr_simple import VPRImageProcessor, FeatureType
from aarapsi_robot_pack.vpred import *

# For logging
class State(Enum):
    DEBUG = "[DEBUG]"
    INFO = "[INFO]"
    WARN = "[WARN]"
    ERROR = "[!ERROR!]"
    FATAL = "[!!FATAL!!]"

class SVMModelProcessor: # main ROS class
    def __init__(self, models_dir, model=None, try_gen=True, ros=False):
        self._clear()       # initialise variables in system
        self.models_dir     = models_dir
        self.ros            = ros
        # for making new models:
        self.cal_qry_ip     = VPRImageProcessor(ros=self.ros) # all else defaults to False or None which we want :)
        self.cal_ref_ip     = VPRImageProcessor(ros=self.ros) # all else defaults to False or None which we want :)

        if not (model is None):
            if isinstance(model, str):
                self.load_model(model)
            elif isinstance(model, dict):
                self.load_model_params(model)
                if try_gen and (not self.model_ready):
                    self.generate_model(model['database_path'], model['qry'], model['ref'], model['img_dims'], model['folder'], model['ft_type'])
            else:
                raise Exception("Model type not supported. Valid types: str, dict")
            if not self.model_ready:
                raise Exception("Model load failed.")
            self._print("[SVMModelProcessor] Model Ready.", State.INFO)

    def generate_model(self, database_path, cal_qry_dataset, cal_ref_dataset, img_dims, folder, ft_type, save=True):
        # store for access in saving operation:
        self.database_path      = database_path
        self.cal_qry_dataset    = cal_qry_dataset
        self.cal_ref_dataset    = cal_ref_dataset 
        self.img_dims           = img_dims
        self.folder             = folder
        self.feat_type          = ft_type
        self.model_ready        = False

        # generate:
        self._load_cal_data()
        self._calibrate()
        self._train()
        self._make()
        self.model_ready        = True

        if save:
            self.save_model(check_exists=True)

        return self

    def save_model(self, dir=None, name=None, check_exists=False):
        if dir is None:
            dir = self.models_dir
        if not self.model_ready:
            raise Exception("Model not loaded in system. Either call 'generate_model' or 'load_model(_params)' before using this method.")
        self._prep_directory(dir)
        if check_exists:
            existing_file = self._check(dir)
            if existing_file:
                self._print("[save_model] File exists with identical parameters: %s", State.INFO)
                return self
        
        # Ensure file name is of correct format, generate if not provided
        file_list, _, _, = scan_directory(dir, short_files=True)
        if (not name is None):
            if not (name.startswith('svmmodel')):
                name = "svmmodel_" + name
            if (name in file_list):
                raise Exception("Model with name %s already exists in directory." % name)
        else:
            name = datetime.datetime.today().strftime("svmmodel_%Y%m%d")

        # Check file_name won't overwrite existing models
        file_name = name
        count = 0
        while file_name in file_list:
            file_name = name + "_%d" % count
            count += 1
        
        separator = ""
        if not (dir[-1] == "/"):
            separator = "/"
        full_file_path = dir + separator + file_name
        np.savez(full_file_path, **self.model)
        self._print("[save_model] Saved file to %s" % full_file_path, State.INFO)
        return self

    def load_model_params(self, model_params, dir=None):
    # load via search for param match
        self._print("[load_model_params] Loading model.", State.DEBUG)
        self.model_ready = False
        if dir is None:
            dir = self.models_dir
        models = self._get_models(dir)
        self.model = {}
        for name in models:
            if models[name]['params'] == model_params:
                self._load(models[name])
                break
        return self

    def load_model(self, model_name, dir=None):
    # load via string matching name of model file
        self._print("[load_model] Loading %s" % (model_name), State.DEBUG)
        self.model_ready = False
        if dir is None:
            dir = self.models_dir
        models = self._get_models(dir)
        self.model = {}
        for name in models:
            if name == model_name:
                self._load(models[name])
                break
        return self
    
    def predict(self, dvc):
        if not self.model_ready:
            raise Exception("Model not loaded in system. Either call 'generate_model' or 'load_model(_params)' before using this method.")
        sequence        = (dvc - self.model['model']['rmean']) / self.model['model']['rstd'] # normalise using parameters from the reference set
        factor1_qry     = find_va_factor(np.c_[sequence])[0]
        factor2_qry     = find_grad_factor(np.c_[sequence])[0]
        # rt for realtime; still don't know what 'X' and 'y' mean! TODO
        Xrt             = np.c_[factor1_qry, factor2_qry]      # put the two factors into a 2-column vector
        Xrt_scaled      = self.model['model']['scaler'].transform(Xrt)  # perform scaling using same parameters as calibration set
        y_zvalues_rt    = self.model['model']['svm'].decision_function(Xrt_scaled)[0] # 'z' value; not probability but "related"...
        y_pred_rt       = self.model['model']['svm'].predict(Xrt_scaled)[0] # Make the prediction: predict whether this match is good or bad
        return (y_pred_rt, y_zvalues_rt, [factor1_qry, factor2_qry])
    
    def generate_svm_mat(self, array_dim=500):
        # Generate decision function matrix:
        f1          = np.linspace(0, self.model['model']['factors'][0].max(), array_dim)
        f2          = np.linspace(0, self.model['model']['factors'][1].max(), array_dim)
        F1, F2      = np.meshgrid(f1, f2)
        Fscaled     = self.model['model']['scaler'].transform(np.vstack([F1.ravel(), F2.ravel()]).T)
        y_zvalues_t = self.model['model']['svm'].decision_function(Fscaled).reshape([array_dim, array_dim])

        fig, ax = plt.subplots()
        ax.imshow(y_zvalues_t, origin='lower',extent=[0, f1[-1], 0, f2[-1]], aspect='auto')
        z_contour = ax.contour(F1, F2, y_zvalues_t, levels=[0], colors=['red','blue','green'])
        p_contour = ax.contour(F1, F2, y_zvalues_t, levels=[0.75])
        ax.clabel(p_contour, inline=True, fontsize=8)
        x_lim = [0, self.model['model']['factors'][0].max()]
        y_lim = [0, self.model['model']['factors'][1].max()]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_box_aspect(1)
        ax.axis('off')
        fig.canvas.draw()

        # extract matplotlib canvas as an rgb image:
        img_np_raw_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_np_raw = img_np_raw_flat.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close('all') # close matplotlib
        #plt.show()

        return (img_np_raw, (x_lim, y_lim))
    
    #### Private methods:
    def _check(self, dir):
        models = self._get_models(dir)
        for name in models:
            if models[name]['params'] == self.model['params']:
                return models[name]
        return ""

    def _load_cal_data(self):
        # Process calibration data (only needs to be done once)
        self._print("Loading calibration query data set...", State.DEBUG)
        if not self.cal_qry_ip.npzLoader(self.database_path, self.cal_qry_dataset, img_dims=self.img_dims): # dims is needed bc cal_xxx_ip initialised without specifying.
            raise Exception('Query load failed.')
        self._print("Loading calibration reference data set...", State.DEBUG)
        if not self.cal_ref_ip.npzLoader(self.database_path, self.cal_ref_dataset, img_dims=self.img_dims): # dims is needed bc cal_xxx_ip initialised without specifying.
            raise Exception('Reference load failed.')

    def _clean_cal_data(self):
        # Goals: 
        # 1. Reshape calref to match length of calqry
        # 2. Reorder calref to match 1:1 indices with calqry
        calqry_xy = np.transpose(np.stack((self.odom_calqry['position']['x'],self.odom_calqry['position']['y'])))
        calref_xy = np.transpose(np.stack((self.odom_calref['position']['x'],self.odom_calref['position']['y'])))
        match_mat = np.sum((calqry_xy[:,np.newaxis] - calref_xy)**2, 2)
        match_min = np.argmin(match_mat, 1) # should have the same number of rows as calqry (but as a vector)
        calref_xy = calref_xy[match_min, :]
        self.features_calref = self.features_calref[match_min, :]
        self.actual_match_cal = np.arange(len(self.features_calqry))

    def _calibrate(self):
        self.features_calqry                = np.array(self.cal_qry_ip.SET_DICT['img_feats'][enum_name(self.feat_type)][self.folder])
        self.features_calref                = np.array(self.cal_ref_ip.SET_DICT['img_feats'][enum_name(self.feat_type)][self.folder])
        self.odom_calqry                    = self.cal_qry_ip.SET_DICT['odom']
        self.odom_calref                    = self.cal_ref_ip.SET_DICT['odom']
        self._clean_cal_data()
        self.Scal, self.rmean, self.rstd    = create_normalised_similarity_matrix(self.features_calref, self.features_calqry)

    def _train(self):
        # We define the acceptable tolerance for a 'correct' match as +/- one image frame:
        self.tolerance      = 10

        # Extract factors that describe the "sharpness" of distance vectors
        self.factor1_cal    = find_va_factor(self.Scal)
        self.factor2_cal    = find_grad_factor(self.Scal)

        # Form input vector
        self.Xcal           = np.c_[self.factor1_cal, self.factor2_cal]
        self.scaler         = StandardScaler()
        self.Xcal_scaled    = self.scaler.fit_transform(self.Xcal)
        
        # Form desired output vector
        self.y_cal          = find_y(self.Scal, self.actual_match_cal, self.tolerance)

        # Define and train the Support Vector Machine
        self.svm_model      = svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True)
        self.svm_model.fit(self.Xcal_scaled, self.y_cal)

        # Make predictions on calibration set to assess performance
        self.y_pred_cal     = self.svm_model.predict(self.Xcal_scaled)
        self.y_zvalues_cal  = self.svm_model.decision_function(self.Xcal_scaled)

        [precision, recall, num_tp, num_fp, num_tn, num_fn] = \
            find_prediction_performance_metrics(self.y_pred_cal, self.y_cal, verbose=False)
        self._print('Performance of prediction on calibration set:\nTP={0}, TN={1}, FP={2}, FN={3}\nprecision={4:3.1f}% recall={5:3.1f}%\n' \
                   .format(num_tp,num_tn,num_fp,num_fn,precision*100,recall*100), State.INFO)
    
    def _prep_directory(self, dir):
        try:
            Path(dir).mkdir(parents=False, exist_ok=False) # throw both error states
        except FileNotFoundError:
            self._print("Error: parent directory does not exist.", State.ERROR)
        except FileExistsError:
            self._print("Directory already exists, no action needed.", State.INFO)

    def _get_models(self, dir=None):
        if dir is None:
            dir = self.models_dir
        models = {}
        try:
            entry_list = os.scandir(dir)
        except FileNotFoundError:
            self._print("Error: directory invalid.", State.ERROR)
            return models
        for entry in entry_list:
            if entry.is_file() and entry.name.startswith('svmmodel'):
                models[os.path.splitext(entry.name)[0]] = np.load(entry.path, allow_pickle=True)
        return models

    def _make(self):
        params_dict         = dict(ref=self.cal_ref_dataset, qry=self.cal_qry_dataset, \
                                    img_dims=self.img_dims, folder=self.folder, \
                                    database_path=self.database_path, ft_type=self.feat_type)
        model_dict          = dict(svm=self.svm_model, scaler=self.scaler, rstd=self.rstd, rmean=self.rmean, factors=[self.factor1_cal, self.factor2_cal])
        self.model          = dict(params=params_dict, model=model_dict)
        self.model_ready    = True

    def _clear(self):
    # Purge all variables
        self.database_path      = ""
        self.cal_qry_dataset    = ""
        self.cal_ref_dataset    = "" 
        self.folder             = ""
        self.models_dir         = ""
        self.tolerance          = None
        self.Xcal               = None
        self.Xcal_scaled        = None
        self.y_cal              = None
        self.y_pred_cal         = None
        self.y_zvalues_cal      = None
        self.svm_model          = None
        self.scaler             = None
        self.rmean              = None
        self.rstd               = None
        self.factor1_cal        = None
        self.factor2_cal        = None
        self.features_calqry    = []
        self.features_calref    = []
        self.feat_type          = FeatureType.NONE
        self.odom_calqry        = {}
        self.odom_calref        = {}
        self.model              = {}
        self.img_dims           = (-1, -1)
        self.model_ready        = False

    def _load(self, raw_model):
    # when loading objects inside dicts from .npz files, must extract with .item() each object
        self.model = dict(model=raw_model['model'].item(), params=raw_model['params'].item())
        self.model_ready = True

    def _print(self, text, state):
    # Print function helper
    # For use with integration with ROS
        try:
            if self.ros: # if used inside of a running ROS node
                if state == State.DEBUG:
                    rospy.logdebug(text)
                elif state == State.INFO:
                    rospy.loginfo(text)
                elif state == State.WARN:
                    rospy.logwarn(text)
                elif state == State.ERROR:
                    rospy.logerr(text)
                elif state == State.FATAL:
                    rospy.logfatal(text)
            else:
                raise Exception("running external to node environment.")
        except:
            print(state.value + " " + str(text))
        
# import rospkg

# if __name__ == '__main__':  # only execute if run as main script
#     root = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__)))
#     model_dir = root + "/cfg/svm_models"
#     dbp = root + "/data/compressed_sets"
#     qry = 's1_cw_z_230208_o0_e0'
#     ref = 's2_cw_z_230222_o0_e1'
#     img_dims = (64, 64)
#     folder = 'forward'
#     model = "svmmodel_20230308"
#     ft_type = FeatureType.RAW
#     model_params = dict(ref=ref, qry=qry, img_dims=img_dims, folder=folder, database_path=dbp, ft_type=ft_type)
#     #test = SVMModelProcessor(model_dir).generate_model(dbp, qry, ref, img_dims, folder, ft_type).save_model(check_exists=True)

#     # These are all valid methods to load a model:
#     test = SVMModelProcessor(model_dir, model=model) # init and load by name
#     #test = SVMModelProcessor(model_dir, model=model_params) # init and load by params match
#     #test = SVMModelProcessor(model_dir).load_model(model) # init, and then load by name
#     #test = SVMModelProcessor(model_dir).load_model_params(model_params) # init, and then load by params match

#     # Generate fake distance vector:
#     dvc = np.random.rand(100)*300 + 600
#     dvc[20:25] = [300, 250, 100, 220, 320]
#     (state, zval, [factor1, factor2]) = test.predict(dvc)
#     print((state, zval, [factor1, factor2]))

#     test.generate_svm_mat()

