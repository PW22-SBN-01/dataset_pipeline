#!/bin/bash

DEPTH_FRAMES=$(ls -CF ~/Datasets/Depth_Dataset_Bengaluru_DENSE/1658384707877/depth_img/ | wc -l)
RGB_FRAMES=$(ls -CF ~/Datasets/Depth_Dataset_Bengaluru_DENSE/1658384707877/rgb_img/ | wc -l)

python3 -c "from tqdm import tqdm; p = tqdm(total=$RGB_FRAMES); p.update($DEPTH_FRAMES)"
