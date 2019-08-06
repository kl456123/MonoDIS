# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------




import _init_paths
import os

import numpy as np
import argparse

import torch
import torch.nn as nn

# will depercated in the future

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from configs import kitti_bev_config
from builder.dataloader_builders.kitti_bev_dataloader_builder import KITTIBEVDataLoaderBuilder
from builder.optimizer_builder import OptimizerBuilder
from builder.scheduler_builder import SchedulerBuilder
from builder import model_builder
from core import trainer
from core.saver import Saver


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--epochs',
        dest='max_epochs',
        help='number of epochs to train',
        default=20,
        type=int)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='directory to save models',
        type=str)
    parser.add_argument(
        '--net', dest='net', help='which arch to use', type=str)
    parser.add_argument(
        '--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument(
        '--mGPUs',
        dest='mGPUs',
        help='whether use multiple GPUs',
        action='store_true')

    # resume trained model
    parser.add_argument(
        '--r',
        dest='resume',
        help='resume checkpoint or not',
        default=False,
        type=bool)
    parser.add_argument(
        '--checkepoch',
        dest='checkepoch',
        help='checkepoch to load model',
        default=1,
        type=int)
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        help='checkpoint to load model',
        default=0,
        type=int)
    # log and diaplay
    parser.add_argument(
        '--use_tfboard',
        dest='use_tfboard',
        help='whether use tensorflow tensorboard',
        default=False,
        type=bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse config of scripts
    args = parse_args()

    config = kitti_bev_config
    data_config = config.data_config
    model_config = config.model_config
    train_config = config.train_config

    if args.resume:
        model_config['pretrained'] = False

    np.random.seed(train_config['rng_seed'])

    torch.backends.cudnn.benchmark = True

    if args.save_dir is not None:
        train_config['save_dir'] = args.save_dir

    assert args.net is not None, 'please select a base model'
    model_config['net'] = args.net

    output_dir = train_config['save_dir'] + "/" + model_config[
        'net'] + "/" + data_config['name']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print('output_dir is already exist')
    print('checkpoint will be saved to {}'.format(output_dir))

    # model
    fasterRCNN = model_builder.build(model_config)

    # saver
    saver = Saver(output_dir)

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN, train_config['device_ids'])

    if args.cuda:
        fasterRCNN.cuda()

    data_loader_builder = KITTIBEVDataLoaderBuilder(data_config, training=True)
    data_loader = data_loader_builder.build()

    # optimizer
    optimizer_builder = OptimizerBuilder(fasterRCNN,
                                         train_config['optimizer_config'])
    optimizer = optimizer_builder.build()

    scheduler_config = train_config['scheduler_config']
    if args.resume:
        checkpoint_name = 'faster_rcnn_{}_{}.pth'.format(args.checkepoch,
                                                         args.checkpoint)
        params_dict = {
            'start_epoch': None,
            'model': fasterRCNN,
            'optimizer': optimizer,
        }
        saver.load(params_dict, checkpoint_name)
        train_config['start_epoch'] = params_dict['start_epoch']
        scheduler_config['last_epoch'] = params_dict['start_epoch'] - 1

    # scheduler(after resume)
    scheduler_builder = SchedulerBuilder(optimizer, scheduler_config)
    scheduler = scheduler_builder.build()

    trainer.train(train_config, data_loader, fasterRCNN, optimizer, scheduler,
                  saver)
