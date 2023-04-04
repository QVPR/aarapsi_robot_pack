#!/usr/bin/env python3

import os
import urllib.request
import gdown
import sys
import os
import imp # TODO: deprecated.
AARAPSI_ROBOT_PACK_ROOT_DIR = imp.find_module("aarapsi_robot_pack")[1]
PATCHNETVLAD_ROOT_DIR       = imp.find_module("patchnetvlad")[1]
HYBRIDNET_ROOT_DIR          = os.path.join(AARAPSI_ROBOT_PACK_ROOT_DIR, 'HybridNet')

def ask_yesnoexit(question):
    """
    Helper to get yes / no / exit answer from user.
    """
    _yes = {'yes', 'y'}
    _no = {'no', 'n'}
    _exit = {'q', 'quit', 'e', 'exit'}

    done = False
    print(question)
    while not done:
        choice = input().lower()
        if choice in _yes:
            return True
        elif choice in _no:
            return False
        elif choice in _exit:
            sys.exit()
        else:
            print("Please respond 'yes', 'no', or 'quit'.")

def download_netvlad_models(force=False):
    global PATCHNETVLAD_ROOT_DIR
    dest_dir = os.path.join(PATCHNETVLAD_ROOT_DIR, 'pretrained_models')
    print("Path to netvlad model/s:\n\t<%s>" % dest_dir)
    if ask_yesnoexit("Auto-download pretrained NetVLAD model/s (takes around 2GB of space)? (y)es/(n)o/(q)uit."):
        if force:
            try:
                os.remove(os.path.join(dest_dir, "mapillary_WPCA128.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA128.pth.tar")):
            print('Downloading mapillary_WPCA128.pth.tar')
            gdown.download("https://drive.google.com/uc?id=1v-MDvga__tblvZ9-m4Zfk6iem4c4Gjyh", os.path.join(dest_dir, "mapillary_WPCA128.pth.tar"), quiet=False)
        
        if force:
            try:
                os.remove(os.path.join(dest_dir, "mapillary_WPCA512.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA512.pth.tar")):
            print('Downloading mapillary_WPCA512.pth.tar')
            gdown.download("https://drive.google.com/uc?id=1D9V_k7VoTCbq6L0B8hVIqniV1vGFpCB8", os.path.join(dest_dir, "mapillary_WPCA512.pth.tar"), quiet=False)
        
        if force:
            try:
                os.remove(os.path.join(dest_dir, "mapillary_WPCA4096.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "mapillary_WPCA4096.pth.tar")):
            print('Downloading mapillary_WPCA4096.pth.tar')
            gdown.download("https://drive.google.com/uc?id=14fe_v2OYVJDa8qCiG1gcgf9I81qFRCGv", os.path.join(dest_dir, "mapillary_WPCA4096.pth.tar"), quiet=False)
       
        if force:
            try:
                os.remove(os.path.join(dest_dir, "pittsburgh_WPCA128.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA128.pth.tar")):
            print('Downloading pittsburgh_WPCA128.pth.tar')
            gdown.download("https://drive.google.com/uc?id=1pnqUoREiTk-UJzU4N4n5doUm4dOIPxIU", os.path.join(dest_dir, "pittsburgh_WPCA128.pth.tar"), quiet=False)
        
        if force:
            try:
                os.remove(os.path.join(dest_dir, "pittsburgh_WPCA512.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA512.pth.tar")):
            print('Downloading pittsburgh_WPCA512.pth.tar')
            gdown.download("https://drive.google.com/uc?id=1FGk77XsRo5ZRHHaxcCSmH_rhOdl0ByFs", os.path.join(dest_dir, "pittsburgh_WPCA512.pth.tar"), quiet=False)
       
        if force:
            try:
                os.remove(os.path.join(dest_dir, "pittsburgh_WPCA4096.pth.tar"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "pittsburgh_WPCA4096.pth.tar")):
            print('Downloading pittsburgh_WPCA4096.pth.tar')
            gdown.download("https://drive.google.com/uc?id=1TcF6Z2n7lxkf_9pXnEbJXmOzCMEljhjO", os.path.join(dest_dir, "pittsburgh_WPCA4096.pth.tar"), quiet=False)
        print('Downloaded all pretrained models.')

def download_hybridnet_models(force=False):
    global HYBRIDNET_ROOT_DIR
    dest_dir = os.path.join(HYBRIDNET_ROOT_DIR, 'pretrained_models')
    print("Path to hybridnet model/s:\n\t<%s>" % dest_dir)
    if ask_yesnoexit("Auto-download pretrained HybridNet model/s (takes around 300MB of space)? (y)es/(n)o/(q)uit."):
        
        if force:
            try:
                os.remove(os.path.join(dest_dir, "HybridNet.caffemodel"))
            except:
                pass
        if not os.path.isfile(os.path.join(dest_dir, "HybridNet.caffemodel")):
            print('Downloading HybridNet.caffemodel')
            gdown.download("https://drive.google.com/uc?id=1_tzdDkM5TeyB37TEduyxWMi_T4CqQ4u4", os.path.join(dest_dir, "HybridNet.caffemodel"), quiet=False)

def download_all_models(force=False):
    
    download_netvlad_models(force)
    print("Path to hybridnet model/s:\n\t<%s>" % os.path.join(HYBRIDNET_ROOT_DIR, 'pretrained_models'))
    download_hybridnet_models(force)

if __name__ == "__main__":
    download_all_models()