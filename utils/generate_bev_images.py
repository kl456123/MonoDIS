# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------




import _init_paths
import os

import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import pickle

import torch
from data.kitti_bev_loader import KITTIBEVLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import data.data_loader as data
from data.kitti_loader import KittiLoader
from utils.bev_encoder import DataEncoder

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from configs.kitti_bev_config import MODEL_CONFIG, OBJ_CLASSES, data_config

from model.faster_rcnn.vgg16 import vgg16

from utils.visualize import visualize_bbox


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='training dataset',
        default='pascal_voc',
        type=str)
    parser.add_argument(
        '--net', dest='net', help='vgg16, res101', default='vgg16', type=str)
    parser.add_argument(
        '--start_epoch',
        dest='start_epoch',
        help='starting epoch',
        default=1,
        type=int)
    parser.add_argument(
        '--epochs',
        dest='max_epochs',
        help='number of epochs to train',
        default=20,
        type=int)
    parser.add_argument(
        '--disp_interval',
        dest='disp_interval',
        help='number of iterations to display',
        default=100,
        type=int)
    parser.add_argument(
        '--checkpoint_interval',
        dest='checkpoint_interval',
        help='number of iterations to display',
        default=10000,
        type=int)

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='directory to save models',
        default="./weights",
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--nw',
        dest='num_workers',
        help='number of worker to load data',
        default=0,
        type=int)
    parser.add_argument(
        '--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument(
        '--ls',
        dest='large_scale',
        help='whether use large imag scale',
        action='store_true')
    parser.add_argument(
        '--mGPUs',
        dest='mGPUs',
        help='whether use multiple GPUs',
        action='store_true')
    parser.add_argument(
        '--bs', dest='batch_size', help='batch_size', default=1, type=int)
    parser.add_argument(
        '--cag',
        dest='class_agnostic',
        help='whether perform class_agnostic bbox regression',
        action='store_true')

    # config optimization
    parser.add_argument(
        '--o',
        dest='optimizer',
        help='training optimizer',
        default="sgd",
        type=str)
    parser.add_argument(
        '--lr',
        dest='lr',
        help='starting learning rate',
        default=0.001,
        type=float)
    parser.add_argument(
        '--lr_decay_step',
        dest='lr_decay_step',
        help='step to do learning rate decay, unit is epoch',
        default=5,
        type=int)
    parser.add_argument(
        '--lr_decay_gamma',
        dest='lr_decay_gamma',
        help='learning rate decay ratio',
        default=0.1,
        type=float)

    # set training session
    parser.add_argument(
        '--s', dest='session', help='training session', default=1, type=int)

    # resume trained model
    parser.add_argument(
        '--r',
        dest='resume',
        help='resume checkpoint or not',
        default=False,
        type=bool)
    parser.add_argument(
        '--checksession',
        dest='checksession',
        help='checksession to load model',
        default=1,
        type=int)
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
        '--dataset_file',
        dest='dataset_file',
        help='which dataset file to used',
        type=str,
        default='train.txt')

    args = parser.parse_args()
    return args


# MODEL_CONFIG = {
# 'num_classes': 2,
# 'output_stride': [8., 16., 32., 64., 128., 192., 384.],
# 'input_shape': (384, 1300),
# }

data_root_path = '/data/object/training'
save_path = './weights'
eval_out = './eval'
test_fig = './toy.png'

train_encoder = DataEncoder(MODEL_CONFIG)

DATA_AUG_CFG = {
    'resize_range': [0.2, 0.4],
    'random_brightness': 10,
    'crop_size': MODEL_CONFIG['input_shape'],
    'random_blur': 0,
    'dataset_file': 'new.txt'
}

BATCH_SIZE = 1
BEV_IMAGE_DIR = './bev_images'


def __change_into_variable(elems, use_gpu=True):
    if use_gpu:
        return [Variable(elem.cuda()) for elem in elems]
    else:
        return [Variable(elem) for elem in elems]


def visualize(img, bbox):
    """visualize the augumentated image for validate the correction"""
    # recover img
    # img = img * DATA_AUG_CFG['normal_van'] + DATA_AUG_CFG['normal_mean']
    img *= 255
    visualize_bbox(img, bbox, save=True)


def save_bev_image(img_data, gt_boxes, sample_idx):
    box_file = os.path.join(BEV_IMAGE_DIR, '{:06d}_box.pkl'.format(sample_idx))
    img_file = os.path.join(BEV_IMAGE_DIR, '{:06d}_img.pkl'.format(sample_idx))
    with open(box_file, 'wb') as f:
        pickle.dump(gt_boxes, f, pickle.HIGHEST_PROTOCOL)

    with open(img_file, 'wb') as f:
        pickle.dump(img_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(cfg.RNG_SEED)

    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    if type(args.save_dir) is str:
        save_dir = args.save_dir
    else:
        save_dir = args.save_dir[0]
    output_dir = save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    fasterRCNN = vgg16(
        ['bg', 'Car'],
        pretrained=False,
        class_agnostic=args.class_agnostic,
        img_channels=6)
    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in list(dict(fasterRCNN.named_parameters()).items()):
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{
                    'params': [value],
                    'lr': lr,
                    'weight_decay': cfg.TRAIN.WEIGHT_DECAY
                }]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(
            args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in list(checkpoint.keys()):
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN, device_ids=[5])

    if args.cuda:
        fasterRCNN.cuda()
    DATA_AUG_CFG['dataset_file'] = args.dataset_file

    # data_loader = data.load_data(data_root_path, args.batch_size, DATA_AUG_CFG, KittiLoader)
    # data_loader = data.load_data(data_root_path, args.batch_size, data_config,
    # KITTIBEVLoader)
    data_loader = data.load_data(BATCH_SIZE, data_config, train_encoder,
                                 KITTIBEVLoader)
    train_size = len(data_loader)
    iters_per_epoch = int(train_size / args.batch_size)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        for step, _data in enumerate(data_loader):
            im_data, im_info, gt_boxes, num_boxes, ry_target, sample_names = _data
            sample_idx = sample_names.numpy()[0]
            save_bev_image(
                im_data.numpy()[0].transpose((1, 2, 0)),
                gt_boxes.numpy()[0, :, :4],
                sample_idx=sample_idx)
            import ipdb
            ipdb.set_trace()
        visualize(im_data.numpy()[0].transpose((1, 2, 0)),
                  gt_boxes.numpy()[0, :, :4])
        im_data, im_info, gt_boxes, num_boxes, ry_target = __change_into_variable(
            [im_data, im_info, gt_boxes, num_boxes, ry_target])

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                    ry_target)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
            + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        # + RCNN_loss_ry.mean()
        loss_temp += loss.data[0]

        # backward
        optimizer.zero_grad()
        loss.backward()
        if args.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().data[0]
                    loss_rpn_box = rpn_loss_box.mean().data[0]
                    loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                    loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
                    # loss_rcnn_ry = RCNN_loss_ry.mean().data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.data[0]
                    loss_rpn_box = rpn_loss_box.data[0]
                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    loss_rcnn_box = RCNN_loss_bbox.data[0]
                    # loss_rcnn_ry = RCNN_loss_ry.mean().data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" %
                      (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                loss_temp = 0
                start = time.time()

        if args.mGPUs:
            save_name = os.path.join(
                output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session,
                                                              epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(
                output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session,
                                                              epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print(end - start)
