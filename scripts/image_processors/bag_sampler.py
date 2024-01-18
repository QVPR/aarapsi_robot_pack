#!/usr/bin/env python3

import rospkg
import sys
import os
import cv2
import shutil

import argparse as ap
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from pathlib import Path
from tqdm.auto import tqdm

from pyaarapsi.core.argparse_tools import check_positive_float, check_bool, check_string, check_string_list
from pyaarapsi.core.ros_tools import roslogger, LogType, rip_bag, compressed2np, raw2np
from pyaarapsi.core.helper_tools import vis_dict, formatException, ask_yesnoexit

'''

ROS Bag Sampler

Open up a bag, grab out images from img_topics at a constant rate, and save to a directory

'''        

class mrc:
    def __init__(self, img_topics, bag_path, bag_name, rate, compress, save_path, save_name, overwrite):
        if compress:
            self.img_topics = [i + "/compressed" if not i.endswith("/compressed") else i for i in img_topics]
        else:
            self.img_topics = img_topics
        self.img_types      = [CompressedImage if i.endswith("/compressed") else Image for i in self.img_topics]

        self.bag_path       = bag_path
        self.bag_name       = bag_name
        self.rate           = rate
        self.compress       = compress
        self.save_path      = save_path
        self.save_name      = save_name

        Path(self.bag_path).mkdir(parents=False, exist_ok=True)
        Path(self.save_path).mkdir(parents=False, exist_ok=True)
        _dir = self.save_path + '/' + self.bag_name
        if overwrite and os.path.exists(_dir):
            if ask_yesnoexit("Delete directory %s? (y)es/(n)o/(q)uit." % _dir):
                shutil.rmtree(_dir)

        Path(_dir).mkdir(parents=False, exist_ok=False)
        for i in self.img_topics:
            Path(self.save_path + '/' + self.bag_name + '/' + (i.replace('/', '_'))).mkdir(parents=False, exist_ok=False)

    def main(self):
        data = rip_bag(self.bag_path + '/' + self.bag_name + '.bag', self.rate, self.img_topics, use_tqdm=True)

        _len = len(data)
        if _len < 1:
            raise Exception('No usable data!')

        print("Converting stored messages (%s)" % (str(_len)))
        new_dict        = {key: [] for key in self.img_topics}
        none_rows       =   0

        for row in tqdm(data):
            if None in row:
                none_rows = none_rows + 1
                continue
                
            for topic in self.img_topics:
                if "/compressed" in topic:
                    new_dict[topic].append(compressed2np(row[1 + self.img_topics.index(topic)]))
                else:
                    new_dict[topic].append(raw2np(row[1 + self.img_topics.index(topic)]))
        
        print("%0.2f%% of %d rows contained NoneType; these were ignored." % (100 * none_rows / _len, _len))
        output = {key: np.array(new_dict[key]) for key in new_dict}
        _len = np.ceil(np.log10(len(output[list(output.keys())[0]]))).astype(int) + 1
        for _key in output:
            print('Saving output, topic: ' + _key)
            _path = self.save_path + '/' + self.bag_name + '/' + (_key.replace('/', '_'))
            for c,_img in tqdm(enumerate(output[_key])):
                cv2.imwrite((_path + '/image_%0' + str(_len) + 'i.png') % c, _img[:,:,-1::-1])
        
if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="Bag Ripper", 
                            description="ROS Bag Ripper Tool",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    sp_default = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/data/image_libraries"
    bp_default = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + "/data/rosbags"
    it_default = ['/ros_indigosdk_occam/image0']

    parser.add_argument('bag_name',             type=check_string,                             help='Set input .bag file name.')
    parser.add_argument('--img-topics', '-It',  type=check_string_list,    default=it_default, help='Set image topics (/compressed can be added to each to mix compressed and uncompressed topics) (default: %(default)s).')
    parser.add_argument('--bag-path',   '-Bp',  type=check_string,         default=bp_default, help='Set system directory (excluding file name) for bag to read (default: %(default)s).')
    parser.add_argument('--save-path',  '-Sp',  type=check_string,         default=sp_default, help='Set output .npz file path (excluding file name) (default: %(default)s).')
    parser.add_argument('--save-name',  '-Sn',  type=check_string,         default='rip.npz',  help='Set output .npz file name (default: %(default)s).')
    parser.add_argument('--rate',       '-r',   type=check_positive_float, default=10.0,       help='Set sample rate (default: %(default)s).')
    parser.add_argument('--compress',   '-C',   type=check_bool,           default=True,       help='Force use of only compressed image topics (default: %(default)s).')
    parser.add_argument('--overwrite',  '-o',   type=check_bool,           default=True,       help='Force overwrite if images already exist (default: %(default)s).')

    try:
        nmrc = mrc(**vars(parser.parse_known_args()[0]))
        nmrc.main()
        roslogger("Operation complete.", LogType.INFO, ros=False) # False as rosnode likely terminated
        sys.exit()
    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)
