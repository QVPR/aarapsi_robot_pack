#!/usr/bin/env python3

# How I did this:
# http://docs.ros.org/en/jade/api/catkin/html/user_guide/setup_dot_py.html
# http://docs.ros.org/en/jade/api/catkin/html/howto/format2/installing_python.html

# An example I looked at:
# https://github.com/ros-perception/vision_opencv/tree/rolling/cv_bridge/python/cv_bridge

# Steps:
# 1. in CMakeLists.txt, uncommented catkin_python_setup()
# 2. Created a directory (name becomes package name) inside the ROS package src folder (which becomes package_dir)
# 3. Made this file, and set packages/package_dir variables to the python package information
# 4. Moved my first .py file into the new to-be-python-package directory
# 5. Created a __init__.py file in the new to-be-python-package directory

# How to use:
# - Remake the catkin workspace
# - Refresh/resource terminal
# -> But ensure PYTHONPATH is correct, with reference to the catkin_workspace (must have: /path/to/catkin/workspace/devel/lib/python3/dist-packages)
# - import and go!

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup
#import subprocess
import os

aarapsi_setup_location      = os.path.abspath(os.path.dirname(__file__))
#patchnetvlad_setup_location = os.path.join(aarapsi_setup_location, 'src/Patch-NetVLAD')
pyaarapsi_setup_location    = os.path.join(aarapsi_setup_location, 'src/pyaarapsi')

# subprocess.call(
#             "pip3 install --no-deps -e %s" % str(patchnetvlad_setup_location), shell=True
#         )

# This 'equivalent' is handled below in setup()
# subprocess.call(
#             "pip3 install --no-deps -e %s" % str(pyaarapsi_setup_location), shell=True
#         )

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['pyaarapsi'],
    package_dir={'': 'src'})

setup(**setup_args)
