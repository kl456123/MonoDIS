# -*- coding: utf-8 -*-
"""
As for fns or fns_bst,just display gt boxes,
As for fps or fps_bst,display gt boxes and dets
"""

from analysis.postprocess import postprocess
from utils.visualize import visualize_bbox

import os
false_case_dir = './results/analysis'
#  difficulty = 'moderate'
#  false_case_dir = os.path.join(false_case_dir, difficulty)
img_dir = '/data/object/training/image_2/'
label_dir = '/data/object/training/label_2/'
kitti_dir = './results/data'

false_cases = ['fns', 'fps']

fns_case = ['fns']
fps_case = ['fps_bst', 'fps']

title = 'false_case'


def vis_false(false_case_dir, case):
    for kitti in os.listdir(false_case_dir):
        sample_idx = os.path.splitext(kitti)[0]
        print('current image idx: {}'.format(sample_idx))
        img_path = os.path.join(img_dir, '{}.png'.format(sample_idx))
        kitti_path = os.path.join(false_case_dir, '{}.txt'.format(sample_idx))
        if case == 'fns':
            label_path = kitti_path
            kitti_path = os.path.join(kitti_dir, '{}.txt'.format(sample_idx))
            command = 'python utils/visualize.py --img {} --kitti {} --title {} --label {}'.format(
                img_path, kitti_path, label_path, label_path)
        else:
            label_path = os.path.join(label_dir, '{}.txt'.format(sample_idx))
            command = 'python utils/visualize.py --img {} --kitti {} --title {} --label {}'.format(
                img_path, kitti_path, kitti_path, label_path)
        os.system(command)
        input("Press Enter to continue...")


def main():
    for false_case in false_cases:
        dirpath = os.path.join(false_case_dir, false_case)
        if false_case == 'fns':
            reverse = True
        else:
            reverse = False
        # reverse = True
        fnames = os.listdir(dirpath)
        fnames_num = sorted([int(fname) for fname in fnames], reverse=reverse)
        fnames = [str(fname_num) for fname_num in fnames_num]
        for dirpath_per_thresh in fnames:
            vis_false(os.path.join(dirpath, dirpath_per_thresh), false_case)


if __name__ == '__main__':
    main()
