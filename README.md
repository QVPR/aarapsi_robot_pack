# aarapsi_robot_pack

- Installation:
  - In the catkin_ws folder: rosdep install --from-paths src --ignore-src -r -y
  - Install additionally items:
    - pip instal numexpr (sudo apt-get install python3-numexpr)
    - pip install fastdist
    - pip install bokeh
- Weird things:
  - Error: libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
    - FIX: export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1  
