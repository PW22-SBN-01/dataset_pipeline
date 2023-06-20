#!/bin/bash

python3 generate_depth_dataset.py --depth_dataset_dir /home/aditya/Datasets/Depth_Dataset_Bengaluru_DENSE/ --gen_rgb --gen_depth
python3 generate_instance_segmentation_pointrend_dataset.py --seg_dataset_dir /home/aditya/Datasets/Depth_Dataset_Bengaluru_DENSE/ --gen_seg
python3 generate_trajectory_dataset.py --trajectory_dataset_dir /home/aditya/Datasets/Depth_Dataset_Bengaluru_DENSE/ --gen_traj
