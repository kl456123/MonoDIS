# -*- coding: utf-8 -*-

import os

command_template = 'python utils/visualize.py --label_file {} --pkl_path {} --kitti_file {}'

label_dir = '/data/object/liangxiong/results/label'
bev_dir = '/data/object/liangxiong/results/bev'
dets_dir = 'results/data'

# visualize all dets in dets_dir

for dets_file in os.listdir(dets_dir):
    sample_idx = dets_file[:6]
    # label
    label_file = os.path.join(label_dir, dets_file)

    # bev
    pkl_path = os.path.join(bev_dir, sample_idx + '.pkl')

    # dets
    kitti_file = os.path.join(dets_dir, dets_file)

    command = command_template.format(label_file, pkl_path, kitti_file)

    os.system(command)
