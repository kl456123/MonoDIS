# -*- coding: utf-8 -*-
import os
import sys
from utils.box_vis import mainv2 as box_3d_vis
import time

data_dir = '/data/object/training'
# data_dir = '/data/pengwu//seq/keyframes'
# data_dir = '/data/dm202_3w/left_img'
# data_dir = '/data/liangxiong/yizhuang'
# data_dir = '/data/2011_09_26/2011_09_26_drive_0011_sync/image_02/data/'
# data_dir = '/data/liangxiong/training/image_02/0000/'
# data_dir = '/home/pengwu/mono3d/kitti/0006/'
#  data_dir = '/data/liangxiong/yizhuang/2019_0107_090221/keyframes'
#  data_dir = '/data/liangxiong/yizhuang/2019_0107_140749/keyframes/'
# data_dir = '/data/hw/image_2'
# data_dir = '/data/liangxiong/pedestrian_data/data/'

result_dir = './results/data/'
# result_dir = '/data/liangxiong/detection/results/data'


def read_dir(result_dir):
    pass


def main():
    # data_file = 'val.txt'
    # data_file_path = os.path.join(data_dir, data_file)
    # with open(data_file_path, 'r') as f:
        # lines = f.readlines()
        # lines = [line.strip() for line in lines]
    start = time.time()

    for ind, det_file in enumerate(sorted(os.listdir(result_dir))):
        sample_name = os.path.splitext(det_file)[0]

        kitti_path = os.path.join(result_dir, '{}.txt'.format(sample_name))
        img_path = os.path.join(data_dir, 'image_2/{}.png'.format(sample_name))
        # img_path = os.path.join(data_dir, '{}.png'.format(sample_name))
        calib_path = os.path.join(data_dir, 'calib/{}.txt'.format(sample_name))
        # calib_path = './000000.txt'
        # calib_path = '/data/liangxiong/training/calib/0000.txt'
        save_path = '{}.jpg'.format(sample_name)
        #  command = 'python utils/box_vis.py --kitti {} --img {} --calib {} --save_path {}'.format(
        #  kitti_path, img_path, calib_path, save_path)
        box_3d_vis(img_path, kitti_path, calib_path, save_path, display_label=True)
        duration = time.time() - start
        sys.stdout.write(
            '\r{}/{} duration: {:.4f}'.format(ind, 3679, duration))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
