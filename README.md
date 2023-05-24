# aarapsi_robot_pack

- Installation:
  - In the catkin_ws folder: rosdep install --from-paths src --ignore-src -r -y
  - Install additionally items:
    - pip instal numexpr (sudo apt-get install python3-numexpr)
    - pip install fastdist
    - pip install bokeh
    - pip install gdown
    - pip install nvidia-ml-py3
- Weird things:
  - Xavier AGX Error: libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
    - FIX: export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1  
  - Xavier AGX Torch:
    - https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
      - https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl
      - pip install ./torch-2....
      - 
