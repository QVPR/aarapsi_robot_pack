#!/usr/bin/env python3

import numpy as np
import copy
try:
    import rospy # used for printing
except:
    pass
import os
import cv2
import sys
import csv
from enum import Enum
from tqdm import tqdm
try:
    from aarapsi_robot_pack.core.enum_tools import enum_name
    from aarapsi_robot_pack.vpr_classes import NetVLAD_Container, HybridNet_Container
    from aarapsi_robot_pack.core.helper_tools import formatException
    PYTHON_LIBRARY_STATUS = True
except:
    print("This file is running without knowledge of, or outside of, the aarapsi_robot_pack python library")
    print("This is OK, however feature extraction and building new compressed libraries is disabled.")
    print("Although best efforts have been made to ensure compatibility, issues may arise.")
    PYTHON_LIBRARY_STATUS = False

# For image processing type
class FeatureType(Enum):
    NONE        = 0
    RAW         = 1
    PATCHNORM   = 2
    NETVLAD     = 3
    HYBRIDNET   = 4

# For logging
class State(Enum):
    DEBUG       = "[DEBUG]"
    INFO        = "[INFO]"
    WARN        = "[WARN]"
    ERROR       = "[!ERROR!]"
    FATAL       = "[!!FATAL!!]"

class VPRImageProcessor: # main ROS class
    def __init__(self, ros=False, init_netvlad=False, init_hybridnet=False, cuda=False, dims=None):
        global PYTHON_LIBRARY_STATUS
        self.clearImageVariables()
        self.clearOdomVariables()
        self.ros            = ros
        self.cuda           = cuda
        self.init_netvlad   = init_netvlad
        self.init_hybridnet = init_hybridnet
        self.allow_build    = PYTHON_LIBRARY_STATUS

        if not dims is None:
            self.IMG_DIMS = dims

        if not self.allow_build and (self.init_hybridnet or self.init_hybridnet):
            self.print("Initialisation request for hybridnet or netvlad but generation of new libraries and feature extraction is disabled")
        elif self.allow_build:
            if self.init_netvlad:
                if dims is None: raise Exception("init_netvlad specified true but dims not provided")
                self.netvlad = NetVLAD_Container(cuda=self.cuda, ngpus=int(self.cuda), logger=lambda x: self.print(x, State.DEBUG), dims=self.IMG_DIMS)

            if self.init_hybridnet:
                if dims is None: raise Exception("init_hybridnet specified true but dims not provided")
                self.hybridnet = HybridNet_Container(cuda=self.cuda, logger=lambda x: self.print(x, State.DEBUG), dims=self.IMG_DIMS)

    def print(self, text, state):
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
                raise Exception
        except:
            print(state.value + " " + str(text))

    def buildFullDictionary(self, dict_in=None):
        if not (dict_in is None):
            self.IMG_FEATS = dict_in['img_feats']
            self.ODOM = dict_in['odom']
            self.TIMES = dict_in['times']
            self.ODOM_PATH = dict_in['odom_path']
            self.IMG_PATHS = dict_in['image_paths']
            self.IMG_DIMS = dict_in['img_dims']
        self.SET_DICT = {'img_feats': self.IMG_FEATS, \
                         'odom': self.ODOM, \
                         'times': self.TIMES,  \
                         'odom_path': self.ODOM_PATH, \
                         'image_paths': self.IMG_PATHS,  \
                         'img_dims': self.IMG_DIMS
                        }
    
    def loadFull(self, img_paths, odom_path, feat_type, img_dims, seed_raw_image_data=None):
    # Load in odometry and image library from raw files (.png and .csv)
    # Feed img_paths to directory containing directories of images
    # odom_path should always be full file path of an odometry.csv file
    # feat_type of enum type FeatureType or a list of enums of type FeatureType
    # img_dims a two-integer positive tuple containing image width and height (width, height) in pixels
    # Returns full dictionary; empty on fail.

        assert self.allow_build == True, "library generation is disabled"

        self.print("[loadFull] Attempting to load library.", State.DEBUG)
        if not self.loadImageFeatures(img_paths, feat_type, img_dims, seed_raw_image_data): raise Exception("Fatal")
        if not len(self.loadOdometry(odom_path)): raise Exception("Fatal")
        self.buildFullDictionary()
        if not (self.IMAGES_LOADED and self.ODOM_LOADED):
            self.print("[loadFull] Terminating; load procedure failed.", State.FATAL)
            sys.exit()
        return self.SET_DICT

    def loadImageFeatures(self, img_paths, fttype_in, img_dims, seed_raw_image_data=None):
    # Load in images

        assert self.allow_build == True, "library generation is disabled"

        if not isinstance(img_paths, list):
            if isinstance(img_paths, str):
                self.print("[loadImageFeatures] img_paths is of type str and not type list. Wrapping in a list to proceed.")
                img_paths = [img_paths]
            else:
                raise Exception("[loadImageFeatures] img_paths provided is not of type list.")

        if not isinstance(fttype_in, list):
            fttypes = [fttype_in]
        else:
            fttypes = fttype_in
        if not all([isinstance(fttype, FeatureType) for fttype in fttypes]):
            raise Exception("[loadImageFeatures] fttype_in provided contains elements that are not of type FeatureType")
        if any([fttype == FeatureType.NONE for fttype in fttypes]):
            raise Exception("[loadImageFeatures] fttype_in provided contains at least one FeatureType.NONE")
        
        self.IMG_DIMS = img_dims
        self.IMG_PATHS = img_paths
        img_set_names = [os.path.basename(img_path) for img_path in self.IMG_PATHS]

        try:
            self.IMG_FEATS = dict.fromkeys([enum_name(fttype) for fttype in fttypes], {}) # init empty #TODO remove fromkeys bc default value is shared memory
            for key in self.IMG_FEATS:
                self.IMG_FEATS[key] = dict.fromkeys(img_set_names)
            for img_path in self.IMG_PATHS:
                self.print("[loadImageFeatures] Loading set %s >>%s<<" % (str(self.IMG_PATHS), str(img_path)), State.INFO)
                self.processImageDataset(img_path, fttypes, seed_raw_image_data)
            self.IMAGES_LOADED = True
            return len(self.IMG_PATHS)
    
        except Exception as e:
            self.print("[loadImageFeatures] Unable to interpret, failed. Check variables.\nEnsure: img_paths is a valid string array, feat_type is a valid FeatureType value (not NONE), and image dimensions are a two-element integer tuple of valid dimensions (greater than zero).\nCode: %s" % (e), State.ERROR)
            self.print(formatException(), State.DEBUG)
            self.clearImageVariables()
            return 0

    def processImageDataset(self, img_path, fttype_in, seed_raw_image_data=None): 
    # Extract images and their features from path
    # Store in array and return them.

        assert self.allow_build == True, "library generation is disabled"
    
        img_set_name    = os.path.basename(img_path)
        if seed_raw_image_data is None:
            img_file_paths  = [os.path.join(img_path, f) for f in np.sort(os.listdir(img_path))]

            if not len(img_file_paths):
                raise Exception("[processImageDataset] No files at path - cannot continue.")
            
            self.print("[processImageDataset] Attempting to process images for directory: %s" % (img_set_name), State.DEBUG)
            image_list      = []
            for img_file_path in tqdm(img_file_paths):
                image_list.append(cv2.imread(img_file_path)[:, :, ::-1])
        else:
            self.print("[processImageDataset] Using seed_raw_image_data for directory: %s" % (img_set_name), State.DEBUG)
            image_list      = seed_raw_image_data[img_path]

        for fttype in fttype_in:
            self.print("[processImageDataset] Loading set %s >>%s<<" % (str(list(self.IMG_FEATS.keys())), enum_name(fttype)), State.DEBUG)
            self.IMG_FEATS[enum_name(fttype)][img_set_name] = copy.deepcopy(self.getFeat(image_list, fttype, use_tqdm=True))
    
    def getFeat(self, im, fttype_in, dims=None, use_tqdm=False):
    # Get features from im, using VPRImageProcessor's set image dimensions.
    # Specify type via fttype_in= (from FeatureType enum; list of FeatureType elements is also handled)
    # Can override the dimensions with dims= (two-element positive integer tuple)
    # Returns feature array, as a flattened array

        assert self.allow_build == True, "feature extraction is disabled"

        if dims is None: # should be almost always unless testing an override
            if not (self.IMG_DIMS[0] > 0 and self.IMG_DIMS[1] > 0):
                raise Exception("[getFeat] image dimensions are invalid")
            else:
                dims = self.IMG_DIMS
        if not isinstance(fttype_in, list):
            fttypes = [fttype_in]
        else:
            fttypes = fttype_in
        if not all([isinstance(fttype, FeatureType) for fttype in fttypes]):
            raise Exception("[getFeat] fttype_in provided contains elements that are not of type FeatureType")
        if any([fttype == FeatureType.NONE for fttype in fttypes]):
            raise Exception("[getFeat] fttype_in provided contains at least one FeatureType.NONE")
        try:
            ft_list     = []
            req_mode    = isinstance(im, list)

            for fttype in fttypes:
                if (fttype == FeatureType.RAW or fttype == FeatureType.PATCHNORM):
                    if not req_mode:
                        im = [im]
                    ft_ready_list = []
                    if use_tqdm: iter_obj = tqdm(im)
                    else: iter_obj = im
                    for i in iter_obj:
                        imr = cv2.resize(i, dims)
                        ft  = cv2.cvtColor(imr, cv2.COLOR_RGB2GRAY)
                        if fttype == FeatureType.PATCHNORM:
                            ft = self.patchNormaliseImage(ft, 8)
                        ft_ready_list.append(ft.flatten())
                    if len(ft_ready_list) == 1:
                        ft_ready = ft_ready_list[0]
                    else:
                        ft_ready = np.stack(ft_ready_list)
                elif fttype == FeatureType.HYBRIDNET:
                    if not self.init_hybridnet: 
                        raise Exception("[getFeat] FeatureType.HYBRIDNET provided but VPRImageProcessor not initialised with init_hybridnet=True")
                    ft_ready = self.hybridnet.getFeat(im, use_tqdm=use_tqdm)
                elif fttype == FeatureType.NETVLAD:
                    if not self.init_netvlad: 
                        raise Exception("[getFeat] FeatureType.NETVLAD provided but VPRImageProcessor not initialised with init_netvlad=True")
                    ft_ready = self.netvlad.getFeat(im, use_tqdm=use_tqdm)
                else:
                    raise Exception("[getFeat] fttype not recognised.")
                ft_list.append(ft_ready)
            if len(ft_list) == 1: 
                return ft_list[0]
            return ft_list
        except Exception as e:
            raise Exception("[getFeat] Feature vector could not be constructed.\nCode: %s" % (e))

    def loadOdometry(self, odom_path):
    # Load in odometry from path

        assert self.allow_build == True, "library generation is disabled"

        self.ODOM_PATH = odom_path
        try:
            self.processOdomDataset()
            self.ODOM = {'position': self._POSI, 'velocity': self._VELO}
            self.ODOM_LOADED = True
            return self.ODOM
        except Exception as e:
            self.print("[loadOdometry] Unable to interpret, failed. Check variables.\nEnsure: odom_path is a valid string.\nCode: %s" % (e), State.ERROR)
            self.clearOdomVariables()
            return {}
    
    def processOdomDataset(self):
    # Extract from position .csv at path, full odometry
    # Return these as nice lists/dicts

        assert self.allow_build == True, "library generation is disabled"

        self.print("[processOdomDataset] Attempting to process odometry from: %s" % (self.ODOM_PATH), State.DEBUG)
        self.TIMES = []
        self._POSI = {'x': [], 'y': [], 'yaw': []}
        self._VELO = {'x': [], 'y': [], 'yaw': []}
        if os.path.isfile(self.ODOM_PATH):
            with open(self.ODOM_PATH, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    self.TIMES.append(          float(row[0]))
                    self._POSI['x'].append(     float(row[1]))
                    self._POSI['y'].append(     float(row[2]))
                    self._POSI['yaw'].append(   float(row[3]))
                    self._VELO['x'].append(     float(row[4]))
                    self._VELO['y'].append(     float(row[5]))
                    self._VELO['yaw'].append(   float(row[6]))
            self.TIMES          = np.array(self.TIMES)       
            self._POSI['x']     = np.array(self._POSI['x'])  
            self._POSI['y']     = np.array(self._POSI['y'])  
            self._POSI['yaw']   = np.array(self._POSI['yaw'])
            self._VELO['x']     = np.array(self._VELO['x'])  
            self._VELO['y']     = np.array(self._VELO['y'])  
            self._VELO['yaw']   = np.array(self._VELO['yaw'])
        else:
            raise Exception("[processOdomDataset] No file at path - cannot continue.")
        return self.TIMES, self._POSI, self._VELO

    def save2npz(self, database_path, filename):
    # Public method. Handles saving the loaded/set attributes of the class instance as a fast loading .npz library/dict file.

        # check we have stuff to save:
        if not (self.IMAGES_LOADED):
            raise Exception("[save2npz] No images loaded. loadImageFeatures() must be performed before any save process can be performed.")
        if not (self.ODOM_LOADED):
            self.print("[save2npz] No odometry loaded: rows will be empty.", State.WARN)

        # ensure we don't double up on a ".npz":
        ext = os.path.splitext(filename)[-1].lower()
        if (ext == ".npz"):
            filename = filename[0:-4]
        # ensure we don't double up on a "/":
        separator = ""
        if not (database_path[-1] == "/"):
            separator = "/"
        # gemerate new full path to save to:
        filename_extended = database_path + separator + filename + "_%dx" % (self.IMG_DIMS[0])
        if not (self.IMG_DIMS[0] == self.IMG_DIMS[1]):
            filename_extended = filename_extended + "%d" % (self.IMG_DIMS[1])
        full_file_path = filename_extended + ".npz"
        # perform save to compressed numpy file:
        try:
            self.print("[save2npz] Saving data as '%s'." % (full_file_path), State.INFO)
            # Perform actual save operation, where each dict key is assigned as the variable name on the left-hand-side of equality 
            # i.e. filename, ft, x, y, and z will be dict keys, with lists or dicts assigned to them.
            np.savez(full_file_path, img_feats=self.IMG_FEATS, odom=self.ODOM, times=self.TIMES, \
                        odom_path=self.ODOM_PATH, image_paths=self.IMG_PATHS, img_dims=self.IMG_DIMS)
        except Exception as e:
            self.print("[save2npz] Unable to perform save operation. Check path.\nCode: %s" % (e), State.ERROR)

    def _npzLoader(self, database_path, filename):
    # Private method, embedded in public methods
        # https://stackoverflow.com/questions/5899497/how-can-i-check-the-extension-of-a-file
        ext = os.path.splitext(filename)[-1].lower()
        if (ext == ".npz"):
            filename = filename[0:-4]
        elif (ext == ""):
            pass
        elif not (ext == ".npz"):
            raise Exception("[_npzLoader] File is not of type .npz!")
        # gemerate new full path to save to:
        file_found = False
        for entry in os.scandir(database_path):
            if entry.is_file() and entry.name.startswith(filename):
                self.print("[_npzLoader] File found: %s" % (entry.name), State.INFO)
                data = np.load(entry.path, allow_pickle=True)
                file_found = True
                break
        if not file_found:
            raise Exception("[_npzLoader] No file found starting with %s in directory '%s'" % (filename, database_path))

        try:
            self.print("[_npzLoader] Data Main Keys: %s" % (str(np.fromiter(data.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')), State.DEBUG)
            self.print("[_npzLoader] Data Image Keys: %s" % (str(np.fromiter(data['img_feats'].item().keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')), State.DEBUG)
            self.print("[_npzLoader] Data Odom Keys: %s" % (str(np.fromiter(data['odom'].item().keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')), State.DEBUG)
            # update class attributes:
            self.IMG_FEATS = data['img_feats'].item()
            self.ODOM = data['odom'].item()
            self.TIMES = data['times']

            self.ODOM_PATH = data['odom_path']
            self.IMG_PATHS = data['image_paths']

            self.IMG_DIMS = tuple(data['img_dims'])

            self.ODOM_LOADED = True
            self.IMAGES_LOADED = True

            self.buildFullDictionary()
            dict_keys = str(np.fromiter(self.SET_DICT.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
            imgs_keys = str(np.fromiter(self.IMG_FEATS.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
            odom_keys = str(np.fromiter(self.ODOM.keys(), (str, 20))).replace('\'', '').replace('\n', '').replace(' ', ', ')
            return dict_keys, imgs_keys, odom_keys
        except Exception as e:
            raise Exception("[_npzLoader] Error: %s" % (e))

    def npzLoader(self, database_path, filename, img_dims=None):
    # Public method. Specify directory path containing .npz file (database_path) and the filename (or an incomplete but unique file prefix)
        if img_dims is None:
            if self.IMG_DIMS is None:
                raise Exception("img_dims not provided, and VPRImageProcessor was not initialised with dimensions")
            img_dims = self.IMG_DIMS
        try:
            while 'x' in filename.split('_')[-1]: # clean image size off end
                filename = filename[0:-1-len(filename.split('_')[-1])]
            filename_extended = filename + "_%dx" % (img_dims[0])
            if not (img_dims[0] == img_dims[1]):
                filename_extended = filename_extended + "%d" % (img_dims[1])

            dict_keys, imgs_keys, odom_keys = self._npzLoader(database_path, filename_extended)
            self.print("[npzLoader] Success. Data found with keys: \n\t%s" % (dict_keys), State.DEBUG)
            self.print("[npzLoader] Image dictionary contains keys: \n\t%s" % (imgs_keys), State.DEBUG)
            self.print("[npzLoader] Odometry dictionary contains keys: \n\t%s" % (odom_keys), State.DEBUG)
            return True
        except Exception as e:
            self.print("[npzLoader] Load failed. Check path and file name is correct.\nCode: %s" % (e), State.ERROR)
            return {}

    def npzDatabaseLoadSave(self, database_path, filename, img_paths, odom_path, feat_type, img_dims, do_save=False, seed_raw_image_data=None):
    # Public method. Handles fast loading and slow loading (in the latter, loaded data can be saved for fast loading next time)
    # database_path, filename as per npzLoader.
    # img_paths, odom_path, feat_type, and img_dims as per loadFull
    # do_save=True enables saving for fast loading (with directory/filename specified by database_path and filename respectively)

        if not self.npzLoader(database_path, filename, img_dims):
            self.print("[npzDatabaseLoadSave] Fast load failed. Building normally.", State.WARN)
            self.SET_DICT = self.loadFull(img_paths, odom_path, feat_type, img_dims, seed_raw_image_data)
            if not len(self.SET_DICT):
                raise Exception("[npzDatabaseLoadSave] Normal load failed, fatal error.")
        
            if do_save:
                self.print("[npzDatabaseLoadSave] Build success.", State.INFO)
                self.save2npz(database_path, filename)

        return self.SET_DICT
    
    def clearImageVariables(self):
        self.IMG_PATHS          = ""
        self.IMG_DIMS           = (0,0)
        self.IMG_FEATS          = {}
        self.IMAGES_LOADED      = False

    def clearOdomVariables(self):
        self.ODOM_PATH          = ""
        self.ODOM               = {}
        self.ODOM_LOADED        = False

    def patchNormaliseImage(self, img, patchLength):
    # TODO: vectorize
    # take input image, divide into regions, normalise
    # returns: patch normalised image

        img1 = img.astype(float)
        img2 = img1.copy()
        
        if patchLength == 1: # single pixel; already p-n'd
            return img2

        for i in range(img1.shape[0]//patchLength): # floor division -> number of rows
            iStart = i*patchLength
            iEnd = (i+1)*patchLength
            for j in range(img1.shape[1]//patchLength): # floor division -> number of cols
                jStart = j*patchLength
                jEnd = (j+1)*patchLength

                mean1 = np.mean(img1[iStart:iEnd, jStart:jEnd])
                std1 = np.std(img1[iStart:iEnd, jStart:jEnd])

                img2[iStart:iEnd, jStart:jEnd] = img1[iStart:iEnd, jStart:jEnd] - mean1 # offset remove mean
                if std1 == 0:
                    std1 = 0.1
                img2[iStart:iEnd, jStart:jEnd] /= std1 # crush by std

        return img2   

    def roundSpatial(self, spatial_vec, metrics=None):
        if metrics is None:
            metrics = {'x': 0.05, 'y': 0.05, 'yaw': (2*np.pi/360)}
        new_spatial_vec = {}
        for key in metrics:
            new_spatial_vec[key] = np.round(np.array(spatial_vec[key])/metrics[key],0) * metrics[key]
        new_spatial_matrix = np.transpose(np.stack([new_spatial_vec[key] for key in list(new_spatial_vec)]))
        groupings = []
        for arr in np.unique(new_spatial_matrix, axis=0): # for each unique row combination:
            groupings.append(list(np.array(np.where(np.all(new_spatial_matrix==arr,axis=1))).flatten())) # indices
            #groupings.append(list(np.array((np.all(new_spatial_matrix==arr,axis=1))).flatten())) # bools
        return new_spatial_vec, groupings
    
    def keep_operation(self, d_in, groupings, mode='first'):
    # Note: order of groupings within the original set can be lost depending on how groupings were generated
        
        # create 'table' where all rows are the same length with first entry as the 'label' (tuple)
        dict_to_list = []
        for bigkey in ['odom', 'img_feats']:
            for midkey in set(d_in[bigkey].keys()):
                for lowkey in set(d_in[bigkey][midkey].keys()):
                    base = [(bigkey, midkey, lowkey)]
                    base.extend(d_in[bigkey][midkey][lowkey])
                    dict_to_list.append(base)
        times = [('times',)]
        times.extend(d_in['times'])
        dict_to_list.append(times)
        np_dict_to_list = np.transpose(np.array(dict_to_list, dtype=object))

        # extract rows
        groups_store = []
        for group in groupings:
            if len(group) < 1:
                continue
            if mode=='average':
                groups_store.append(np.mean(np_dict_to_list[1:, :][group,:], axis=0))
                continue
            elif mode=='first': 
                index = 0
            elif mode=='random': 
                index = int(np.random.rand() * (len(group) - 1))
            # for first and random modes:
            index_to_keep = group[index] + 1 # +1 accounts for label
            groups_store.append(np_dict_to_list[index_to_keep, :])

        # restructure and reorder
        cropped_store = np.array(groups_store)
        ind = -1
        for c, i in enumerate(np_dict_to_list[0,:]):
            if i[0] == 'times':
                ind = c
                break
        if ind == -1: raise Exception("Fatal")
        cropped_reorder = cropped_store[cropped_store[:,-1].argsort()]
        d_in.pop('times')
        d_in['times'] = cropped_reorder[:,c]

        # convert back to dictionary and update old dictionary entries
        for bigkey in ['odom', 'img_feats']:
            for midkey in set(d_in[bigkey].keys()):
                for lowkey in set(d_in[bigkey][midkey].keys()):
                    for c, i in enumerate(np_dict_to_list[0,:]):
                        if (bigkey,midkey,lowkey) == i:
                            d_in[bigkey][midkey].pop(lowkey)
                            d_in[bigkey][midkey][i[2]] = np.stack(cropped_reorder[:,c],axis=0)
        return d_in
    
    def discretise(self, metrics=None, mode=None, keep='first'):
        if not len(self.SET_DICT):
            raise Exception("[filter] Full dictionary not yet built.")
        filtered = copy.deepcopy(self.SET_DICT) # ensure we don't change the original dictionary

        if mode is None:
            return filtered
        valid_keeps = ['first', 'random', 'average']
        if not keep in valid_keeps: # ensure valid
            raise Exception('[filter] Unsupported keep style %s. Valid keep styles: %s' % (str(keep), str(valid_keeps)))
        valid_modes = ['position', 'velocity']
        if not mode in valid_modes: # ensure valid
            raise Exception('[filter] Unsupported mode %s. Valid modes: %s' % (str(mode), str(valid_modes)))
        
        # Perform filter step:
        if mode in ['position', 'velocity']: # valid inputs to roundSpatial()
            (filtered['odom'][mode], groupings) = self.roundSpatial(filtered['odom'][mode], metrics)
        #elif mode in []: #TODO
        #    pass
        filtered = self.keep_operation(filtered, groupings, keep) # remove duplications
        return filtered
    
    def destroy(self):
        del self.ODOM_PATH     
        del self.ODOM      
        del self.ODOM_LOADED   
        del self.IMG_PATHS     
        del self.IMG_DIMS      
        del self.IMG_FEATS     
        del self.IMAGES_LOADED 
        del self.TIMES
        del self.SET_DICT
        if self.allow_build:
            if self.init_hybridnet:
                self.hybridnet.destroy()
                del self.hybridnet
            if self.init_netvlad:
                self.netvlad.destroy()
                del self.netvlad
        del self.init_netvlad
        del self.init_hybridnet

