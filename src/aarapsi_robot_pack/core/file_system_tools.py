#!/usr/bin/env python3

import os
import numpy as np


def scan_directory(path, short_files=False):
    dir_scan = os.scandir(path) # https://docs.python.org/3/library/os.html#os.scandir
    fil_names = []
    dir_names = []
    file_exts = []
    for entry in dir_scan:
        if entry.is_file():
            if entry.name.startswith('.'):
                continue
            if short_files:
                fil_names.append(os.path.splitext(entry.name)[0].lower())
            else:
                fil_names.append(entry.name)
            file_exts.append(os.path.splitext(entry)[-1].lower())
        elif entry.is_dir(): 
            dir_names.append(entry.name)
        else: raise Exception("Unknown file type detected.")
    return fil_names, dir_names, list(np.unique(file_exts))

def check_dir_type(path, filetype=None, alltype=False):
    fs, ds, exts = scan_directory(path)
    if filetype == '': filetype = None
    if (len(fs) > 0) and not (filetype is None): # contains a file and we want a file
        if (filetype in exts): # we want a file and it exists
            if (alltype == False): # we don't care if there are other file types present:
                return True
            elif (len(exts) == 1): # there is only one and it is only what we want
                return True
        return False # we wanted a file and it didn't exist, or there were multiple types and we didn't want that
    if (len(ds) > 0) and (filetype is None): # contains a folder and we only want folders
        return True
    return False # either: 
                 # 1. the directory has no files but we want a filetype
                 # 2. the directory has files but we want no files
                 # 3. the directory has no files and we want no files (good), but it has no folders either (and we want folders)

def check_structure(root, ft, at=False, skip=[]):
    if not check_dir_type(root):  # root must be only directories
        return False, [] 
    dir_paths = []
    for dir in os.scandir(root):
        if (dir.name in skip) or (dir.path in skip): 
            continue
        if not check_dir_type(dir.path, filetype=ft, alltype=at): 
            return False, []
        else: dir_paths.append(dir.path)
    return True, dir_paths

def find_shared_root(dirs):
    paths       = []
    lengths     = []
    depth       = 0
    built_root  = ""

    # clean paths, find depth of each
    for dir in dirs:
        path = os.path.normpath(dir).split(os.sep)
        if path[0] == "": path.pop(0)
        paths.append(path)
        lengths.append(len(path))

    # find shared root and depth shared:
    for dir_level in range(min(lengths)):
        path_init = paths[0][dir_level]
        add = True
        for path in paths:
            if not (path[dir_level] == path_init):
                add = False
                break

        if add:
            built_root += ("/" + path[dir_level])
            depth += 1
    
    dist = max(lengths) - depth
    return depth, dist, built_root
