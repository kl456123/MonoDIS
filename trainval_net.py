# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------

import os

import numpy as np
import argparse
import shutil
import json

import torch
import torch.nn as nn

# will depercated in the future
import sys
sys.path.append('./lib')

from builder import dataloader_builder
from builder.optimizer_builder import OptimizerBuilder
from builder.scheduler_builder import SchedulerBuilder
from builder import model_builder
from core import trainer
from core.saver import Saver
from core.summary_writer import SummaryWriter


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
        '--net', dest='net', help='which base mode to use', type=str)
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
    parser.add_argument(
        '--config', dest='config', help='config file(.json)', type=str)
    parser.add_argument('--lr', dest='lr', help='learning rate', type=float)
    parser.add_argument(
        '--model', dest='model', help='path to pretrained model', type=str)

    parser.add_argument(
        '--in_path', default=None, type=str, help='Input directory.')
    parser.add_argument(
        '--out_path', default=None, type=str, help='Output directory.')
    parser.add_argument(
        '--pretrained_path',
        default=None,
        type=str,
        help='Path to pretained model')
    parser.add_argument('--job_name', default='', type=str, help='name of job')
    args = parser.parse_args()
    return args


def change_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        # used for scheduler
        param_group['initial_lr'] = lr


if __name__ == '__main__':
    # parse config of scripts
    args = parse_args()
    print(args.pretrained_path)
    print(args.in_path)
    print(args.out_path)
    with open(args.config) as f:
        config = json.load(f)

    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['train_config']
    if args.in_path is not None:
        # overwrite the data root path
        data_config['dataset_config']['root_path'] = args.in_path
        # data_config['dataset_config']['root_path'] = os.path.join(
    # args.in_path, 'object/training')
    if args.pretrained_path is not None:
        model_config['feature_extractor_config'][
            'pretrained_path'] = args.pretrained_path

    if args.resume:
        model_config['pretrained'] = False

    assert args.net is not None, 'please select a base model'
    model_config['net'] = args.net

    np.random.seed(train_config['rng_seed'])

    torch.backends.cudnn.benchmark = True

    train_config['save_dir'] = args.out_path

    output_dir = train_config['save_dir'] + "/" + model_config[
        'net'] + "/" + data_config['name']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o777)

    else:
        print('output_dir is already exist')

    # copy config to output dir
    shutil.copy2(args.config, output_dir)

    print('checkpoint will be saved to {}'.format(output_dir))

    # model
    fasterRCNN = model_builder.build(model_config)

    # saver
    saver = Saver(output_dir)

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN, train_config['device_ids'])

    if args.cuda:
        fasterRCNN.cuda()

    #  data_loader_builder = Mono3DKittiDataLoaderBuilder(data_config, training=True)
    data_loader = dataloader_builder.build(data_config, training=True)

    # optimizer
    optimizer_builder = OptimizerBuilder(fasterRCNN,
                                         train_config['optimizer_config'])
    optimizer = optimizer_builder.build()

    scheduler_config = train_config['scheduler_config']
    if args.resume:
        # resume mode
        checkpoint_name = 'faster_rcnn_{}_{}.pth'.format(args.checkepoch,
                                                         args.checkpoint)
        params_dict = {
            'start_epoch': None,
            'model': fasterRCNN,
            'optimizer': optimizer,
            'last_step': None
        }
        saver.load(params_dict, checkpoint_name)
        train_config['start_epoch'] = params_dict['start_epoch']
        scheduler_config['last_step'] = params_dict['last_step'] - 1

    if args.model is not None:
        # pretrain mode
        # just load pretrained model
        params_dict = {'model': fasterRCNN, }
        saver.load(params_dict, args.model)

    if args.lr is not None:
        change_lr(args.lr, optimizer)

    # scheduler(after resume)
    scheduler_builder = SchedulerBuilder(optimizer, scheduler_config)
    scheduler = scheduler_builder.build()

    # summary writer
    summary_path = os.path.join(output_dir, './summary')
    summary_writer = SummaryWriter(summary_path)

    trainer.train(train_config, data_loader, fasterRCNN, optimizer, scheduler,
                  saver, summary_writer)
