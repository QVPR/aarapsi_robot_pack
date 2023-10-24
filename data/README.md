# Data Inputs and Outputs
## compressed_sets
This folder is used by pyaarapsi's vpr_dataset_tool.py for generating and storing ```rosbag``` contents and extracted features for VPR.
### root:
Compressed ```numpy``` dictionaries, where each file is a reference data set for VPR
### params:
Small files for vpr_dataset_tool that uniquely describe each data set. Names correspond between the folders.
### filt:
Filtered versions of the data sets. No longer actively used.
## image_libraries:
Legacy, no longer actively used. Old style uncompressed image libraries.
## maps:
Pointclouds of environments for ground truth localization
## paths:
Legacy, no longer actively used. CSV files for paths to follow.
## rosbags:
Whole rosbags to extract data sets from, to go into compressed_sets
## videos:
Videos generated in testing.
