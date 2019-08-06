# -*- coding: utf-8 -*-
"""
How to solve dims and oritation of pedestrian ?
"""

import os
import numpy as np

from utils.orient_eval import read_labels, label_dir
total_3ds_label = np.empty((0, 7))

# collect
for lbl_file in os.listdir(label_dir):
    sample_name = os.path.splitext(lbl_file)[0]
    box_2ds_label, box_3ds_label = read_labels(
        label_dir, sample_name, classes=['Cyclist'])
    total_3ds_label = np.append(total_3ds_label, box_3ds_label, axis=0)

dim = total_3ds_label[:, :3]
pos = total_3ds_label[:, 3:6]
ry = total_3ds_label[:, 6:]

print(dim.mean(axis=0))

from utils.analysis import data_vis

data_vis(dim[:, 0])
data_vis(dim[:, 1])
data_vis(dim[:, 2])

print(dim.std(axis=0))
