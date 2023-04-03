#!/usr/bin/env python3

import rospkg
import os
from os.path import join, exists, isfile
from os import makedirs
os.environ['GLOG_minloglevel'] = '2' # must be done prior to importing caffe to suppress excessive logging.
import caffe
import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
from PIL import Image

class PlaceDataset(torch.utils.data.Dataset):
    def __init__(self, image_data, dims=None):
        super().__init__()
    
        self.images         = image_data
        self.dims           = dims

        if isinstance(self.images, list):
            # either contains PIL Images, str paths to images, or np.ndarray images
            self.len = lambda: len(self.images)
            if isinstance(self.images[0], Image.Image): # type list of PIL Images
                self.getitem = lambda index: np.array(self.images[index])
            elif isinstance(self.images[0], str): # type list of strs
                self.getitem = lambda index: cv2.imread(self.images[index])
            elif isinstance(self.images[0], np.ndarray): # type list of np.ndarray images
                if len(self.images[0].shape) == 2: # greyscale image
                    self.getitem = lambda index: np.dstack((self.images[index],)*3).astype(np.uint8)
                else: # rgb image:
                    self.getitem = lambda index: self.images[index].astype(np.uint8)
            else:
                raise Exception("Input of type list but contains elements of unknown type. Type: %s" % (str(type(self.images[0]))))
        elif isinstance(self.images, np.ndarray):
            self.len = lambda: self.images.shape[0]
            if self.dims is None: # type np.ndarray of np.ndarray images
                if len(self.images[0].shape) == 2: # grayscale
                    self.getitem = lambda index: np.dstack((self.images[index],)*3).astype(np.uint8)
                else:
                    self.getitem = lambda index: self.images[index].astype(np.uint8)
            else: # type np.ndarray of np.ndarray flattened images
                self.getitem = lambda index: np.dstack((np.reshape(self.images[index], self.dims),)*3).astype(np.uint8)
        else:
            raise Exception("Input not of type list or np.ndarray. Type: %s" % (str(type(self.images))))
    
    def __getitem__(self, index):
        return self.getitem(index), index
    
    def __len__(self):
        return self.len()
    
    def destroy(self):
        del self.len
        del self.getitem
        del self.images
        del self.dims

class HybridNet_Container:
    def __init__(self, logger=print, cuda=False, 
                 dims=None, target_layer='fc7_new'):
        
        self.cuda           = cuda
        self.logger         = logger
        self.target_layer   = target_layer
        self.dims           = dims
        # keep these features dim fixed as they need to match the network architecture inside "HybridNet"
        self.layerLabs      = ['conv3', 'conv4', 'conv5', 'conv6' ,'pool1', 'pool2', 'fc7_new', 'fc8_new']
        self.layerDims      = [64896, 64896, 43264, 43264, 69984, 43264, 4096, 2543]
        self.layerDict      = dict(zip(self.layerLabs, self.layerDims))

        if self.cuda:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.load()

    def load(self):

        model_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/src/aarapsi_robot_pack/HybridNet/"
        model_file = os.path.join(model_path,'HybridNet.caffemodel')

        self.logger('Loading HybridNet model')

        self.net = caffe.Net(os.path.join(model_path,'deploy.prototxt'), model_file, caffe.TEST)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', np.load(os.path.join(model_path,'amosnet_mean.npy')).mean(1).mean(1)) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        self.net.blobs['data'].reshape(1,3,227,227)

        self.logger('HybridNet loaded')

    def destroy(self):
        del self.logger
        del self.net
        del self.transformer
        del self.cuda
        del self.target_layer
        del self.dims
        del self.layerLabs
        del self.layerDims
        del self.layerDict

    def clean_data_input(self, dataset_input, dims):
        if isinstance(dataset_input, str): # either a single image path or a set (or a bad path has been provided)
            try:
                img = cv2.imread(dataset_input)
                if img is None: raise Exception('Image path is invalid')
                dataset_clean = PlaceDataset([dataset_input]) # handle single image
                # INPUTS: type list of strs
            except:
                try:
                    ref_filenames = [os.path.join(dataset_input, filename) 
                                    for filename in sorted(os.listdir(dataset_input)) 
                                    if os.path.isfile(os.path.join(dataset_input, filename))]
                    dataset_clean = PlaceDataset(ref_filenames) # handle directory of images
                    # INPUTS: type list of strs
                except:
                    raise Exception("Bad path specified; input is not an image or directory path")
        elif isinstance(dataset_input, Image.Image):
            dataset_clean = PlaceDataset([dataset_input]) # handle PIL image
            # INPUTS: type list of PIL Images
        elif isinstance(dataset_input, np.ndarray):
            if dims is None:
                dataset_clean = PlaceDataset([dataset_input]) # handle single greyscale or rgb image
                # INPUTS: type list of np.ndarray greyscale or rgb images
            else:
                dataset_clean = PlaceDataset(dataset_input, dims) # handle compressed npz input
                # INPUTS: type np.ndarray of np.ndarray flattened images
        elif isinstance(dataset_input, list): # np.ndarrays, PIL images, or strs:
            if any([isinstance(dataset_input[0], i) for i in [str, Image.Image, np.ndarray]]):
                dataset_clean = PlaceDataset(dataset_input)
                # INPUTS: type list of strs, PIL Image.Images, or np.ndarray images
            else:
                raise Exception('Bad input type.')
        else:
            raise Exception('Bad input type.')
        return dataset_clean

    def getFeat(self, dataset_input, dims=None, use_tqdm=True, save_dir=None):
    # Accepts:
    # Strings for path to a single image, or a directory containing other images
    # list of np.ndarrays or PIL images or image paths
    # A 2D np.ndarray (assumed single image)
    # A 3D np.ndarray (first index is assumed to specify individual images)
    # A 2D np.ndarray of flattened images (first index is assumed to specify individual images) (required dims to be specified)
    # a PIL image
        dataset_clean = self.clean_data_input(dataset_input, dims)

        feats = []
        if use_tqdm: iteration_obj = tqdm(dataset_clean)
        else: iteration_obj = dataset_clean
        
        for (input_data, _) in iteration_obj:
            self.net.blobs['data'].data[...] = self.transformer.preprocess('data', input_data)
            self.net.forward()

            feat = np.squeeze(self.net.blobs[self.target_layer].data).flatten()
            feats.append(feat)

        dataset_clean.destroy()
            
        if len(feats) == 1:
            db_feat = feats[0]
        else:
            db_feat = np.array(feats)

        if not (save_dir is None):
            if not exists(save_dir):
                makedirs(save_dir)
            output_global_features_filename = join(save_dir, 'NetVLAD_feats.npy')
            np.save(output_global_features_filename, db_feat)

        return db_feat
