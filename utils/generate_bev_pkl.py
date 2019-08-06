# -*- coding: utf-8 -*-




"""
generate bev map pkl file and bev label kitti file
"""

import os

from builder.dataloader_builders.kitti_bev_dataloader_builder import KITTIBEVDataLoaderBuilder
from configs import kitti_bev_config
from utils.visualize import save_pkl
import sys


def save_bev_map(bev_map, label_info, cache_dir):
    label_idx = os.path.splitext(label_info)[0][-6:]
    label_file = label_idx + '.pkl'
    pkl_path = os.path.join(cache_dir, label_file)
    save_pkl(bev_map.numpy(), pkl_path)


def save_bev_kitti(dets, label_info, output_dir):
    class_name = 'Car'
    label_idx = os.path.splitext(label_info)[0][-6:]
    label_file = label_idx + '.txt'
    label_path = os.path.join(output_dir, label_file)
    res_str = []
    kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} -1 -1 -1 -1000 -1000 -1000 -10 {:.8f}'
    with open(label_path, 'w') as f:
        for det in dets:
            xmin, ymin, xmax, ymax, class_idx = det
            res_str.append(
                kitti_template.format(class_name, xmin, ymin, xmax, ymax,
                                      class_idx))
        f.write('\n'.join(res_str))


def check_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        print('dir {} exist already!'.format(target_dir))


def main():
    config = kitti_bev_config
    data_config = config.eval_data_config
    eval_config = config.eval_config
    # assert data_config['batch_size'] == 1

    # check dir
    check_dir(eval_config['cache_gt_dir'])
    check_dir(eval_config['cache_img_dir'])

    print('label will be save to {}'.format(eval_config['cache_gt_dir']))
    print('bev map will be save to {}'.format(eval_config['cache_img_dir']))

    dataloader_builder = KITTIBEVDataLoaderBuilder(data_config, training=False)
    data_loader = dataloader_builder.build()

    num_samples = len(data_loader)

    for i, data in enumerate(data_loader):
        img_orig = data['img_orig']
        img_file = data['img_name']
        gt_boxes = data['bbox']

        # generate bev map pkl
        save_bev_map(img_orig[0], img_file[0], eval_config['cache_img_dir'])

        # generate kitti text file
        save_bev_kitti(gt_boxes.numpy()[0], img_file[0],
                       eval_config['cache_gt_dir'])
        sys.stdout.write('\r{}/{}'.format(i + 1, num_samples))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
