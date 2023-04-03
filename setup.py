#!/usr/bin/env python3

# How I did this:
# http://docs.ros.org/en/jade/api/catkin/html/user_guide/setup_dot_py.html
# http://docs.ros.org/en/jade/api/catkin/html/howto/format2/installing_python.html

# An example I looked at:
# https://github.com/ros-perception/vision_opencv/tree/rolling/cv_bridge/python/cv_bridge

# Steps:
# 1. in CMakeLists.txt, uncommented catkin_python_setup()
# 2. Created a package with the same name as the ROS package, inside the ROS package src folder
# 3. Made this file, and set packages/package_dir variables to the python package information
# 4. Moved my first .py file into the new to-be-python-package directory
# 5. Created a __init__.py file in the new to-be-python-package directory

# How to use:
# - Remake the catkin workspace
# - Refresh/resource terminal
# -> But ensure PYTHONPATH is correct, with reference to the catkin_workspace (must have: /path/to/catkin/workspace/devel/lib/python3/dist-packages)
# - import and go!

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['aarapsi_robot_pack'],
    package_dir={'': 'src'})

setup(**setup_args)
