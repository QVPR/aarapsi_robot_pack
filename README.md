# aarapsi_robot_pack
ROS package for the AARAPSI project ([github](https://github.com/QVPR/aarapsiproject)) ([website](https://research.qut.edu.au/qcr/Projects/adversity%E2%80%90-and-adversary%E2%80%90robust-adaptive-positioning-systems-with-integrity/))

## Installation
This assumes you have a valid ```ROS noetic``` (ROS1 for Ubuntu 20.04 with python 3.8.10) environment. If not, please visit the [ROS wiki for noetic installation](https://wiki.ros.org/noetic/Installation/Ubuntu) and follow the tutorials to install and configure `ROS`.
Once ROS is configured, install [our catkin workspace](https://github.com/QVPR/aarapsi_offrobot_ws) or build one from scratch with the equivalent packages. For more information, visit the [wiki page for ROS catkin workspaces](https://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment).
- In the root of the catkin workspace, execute ```rosdep install --from-paths src --ignore-src -r -y```
- Install missing python libraries: ```pip3 install numexpr fastdist bokeh gdown nvidia-ml-py3 pyexiv2==2.8.0```

## Package Contents
- **cfg**: Configuration files including yaml files, reference documentation, SVMs, urdf files
- **data**: Generated or utilised data sets, maps, images
- **exe**: Executables
- **launch**: ```roslaunch``` files for nodes in this package as well as custom configurations for other packages such as [HDL_graph_slam](https://github.com/koide3/hdl_graph_slam)
- **media**: Images for using in GUIs or plots
- **msg**: ```rosmsg``` files
- **nodes**: ```rosnode``` python files
- **rviz**: ```RViz``` configuration files
- **scripts**: Helper nodes, testing scripts, and experiments
- **servers**: ```Bokeh``` server files (Broken)
- **src**: Where [pyaarapsi](https://github.com/QVPR/pyaarapsi) lives
- **srv** ```rosservice``` files

   
## Weird things to note for aarch64 (AgileX Scout Mini)
- Xavier AGX Error: libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
  - FIX: export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1  
- Xavier AGX Torch:
  - https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
    - https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl
    - pip install ./torch-2....
  - I learnt the hard way: do not try to custom build or install CUDA/CUDNN/anything. Sort out all of these things first; you must use the CUDA and CUDNN version installed with jetpack otherwise tears will be shed and days will be wasted. There are prebuilt PyTorch and torchvision wheels you can install if you need them.
