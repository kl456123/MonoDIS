# -*- coding: utf-8 -*-
"""
tools for visualize box for selecting the best anchors scale
and ratios
"""

import cv2
import numpy as np
import argparse
import pickle
from utils.generate_anchors import generate_anchors
# import seaborn as sns
import matplotlib.pyplot as plt


def expand_anchors(anchors, feat_size=(24, 79), feat_stride=16):
    # initialize some params
    num_anchors = anchors.shape[0]
    feat_height, feat_width = feat_size

    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel())).transpose()
    A = num_anchors
    K = shifts.shape[0]

    anchors = anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    anchors = anchors.reshape(K * A, 4)
    return anchors


def visualize_bbox(img,
                   bboxes,
                   gt_bboxes=[],
                   class_names=[],
                   size=None,
                   save=False,
                   title='test',
                   keypoints=None,
                   display=True):
    """
    Args:
        bboxes: non-normalized(N,4)
        img: non-noramlized (H,W,C)(bgr)
    """

    print(("img shape: ", img.shape))
    #################################
    # Image
    ################################

    # do resize first
    if size is not None:
        img = cv2.resize(img)

    # do something visualization according to the num of channels of images
    num_channles = img.shape[-1]

    # all image in imgs_batch should be 3-channels,3-dims
    imgs_batch = []
    h, w = img.shape[:2]
    if num_channles == 1 or num_channles > 3:
        # gray image
        for idx in range(num_channles):
            blob = np.zeros((h, w, 3))
            blob[:, :, 0] = img[:, :, idx]
            imgs_batch.append(blob)
    elif num_channles == 3:
        # color image
        imgs_batch.append(img)
    # make array contiguous for used by cv2
    imgs_batch = [
        np.ascontiguousarray(
            img, dtype=np.uint8) for img in imgs_batch
    ]

    #####################################
    # BOX
    #####################################
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.asarray(bboxes)
    #  bboxes = bboxes.astype(np.int)

    # display
    for idx, img in enumerate(imgs_batch):
        for i, box in enumerate(bboxes):
            cv2.rectangle(
                img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                color=(55, 255, 155),
                thickness=2)
            if len(box) == 5:
                text = class_names[i] + ' ' + str(box[4])
                cv2.putText(
                    img,
                    text, (int(box[0]), int(box[1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0))
        for i, box in enumerate(gt_bboxes):
            cv2.rectangle(
                img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                color=(255, 255, 255),
                thickness=2)

        if keypoints is not None:
            keypoints = keypoints.reshape((-1, 2))
            point_size = 1
            point_color = (0, 0, 255)
            for point in keypoints:
                cv2.circle(
                    img, (point[0], point[1]),
                    point_size,
                    point_color,
                    thickness=2)
        if display:
            cv2.imshow(title, img)
            cv2.waitKey(0)

        if save:
            img_path = 'res_%d.jpg' % idx
            cv2.imwrite(img_path, img)


def visualize_points(img, keypoints):
    pass


def vis_featmap(featmap):
    """
    Args:
        featmap:Tensor(H,W,C)
    """

    if len(featmap.shape) == 3:
        featmap = featmap.sum(axis=-1)

    assert len(featmap.shape) == 2
    sns.set()
    ax = sns.heatmap(featmap)
    plt.show()
    return ax


def read_kitti(label_file, classes=['Car'], pred=True, use_3d=False):
    """
    Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()

    boxes = []
    class_names = []
    for line in lines:
        obj = line.strip().split(' ')
        obj_name = obj[0]
        #  if obj_name not in classes:
        #  continue
        xmin = int(float(obj[4]))
        ymin = int(float(obj[5]))
        xmax = int(float(obj[6]))
        ymax = int(float(obj[7]))
        if pred:
            conf = float(obj[-1])
        else:
            conf = 1.0
        if use_3d:
            h = float(obj[8])
            w = float(obj[9])
            l = float(obj[10])
            boxes.append([xmin, ymin, xmax, ymax, h, w, l, conf])
        else:
            boxes.append([xmin, ymin, xmax, ymax, conf])
        class_names.append(obj_name)
    return np.asarray(boxes), class_names


def save_pkl(pkl_data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f, pickle.HIGHEST_PROTOCOL)


def shift_bbox(bbox, translation):
    """
    Args:
        translation:(x_delta,y_delta)
    """
    bbox[:, 0] += translation[0]
    bbox[:, 1] += translation[1]
    bbox[:, 2] += translation[0]
    bbox[:, 3] += translation[1]
    return bbox


def read_img(img_name):
    return cv2.imread(img_name)


def read_pkl(pkl_name):
    with open(pkl_name) as f:
        pkl_array = pickle.loads(f.read())
    pkl_data = np.asarray(pkl_array)
    return pkl_data


def read_keypoints(keypoint_file):
    keypoints = np.loadtxt(keypoint_file).astype(np.float32)
    return keypoints


def test():
    img_name = './img3.jpg'
    img = read_img(img_name)
    cv2.imshow('test', img)
    cv2.waitKey(0)


def analysis_boxes(boxes):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    print(('h: ', h))
    print(('w: ', w))


def parser_args():
    parser = argparse.ArgumentParser(
        description='Visualize bbox in one image from txt file')
    parser.add_argument('--img', dest='img', help='path to image', type=str)
    parser.add_argument('--pkl', dest='pkl', help='path to pkl', type=str)
    parser.add_argument(
        '--label', dest='label', help='path to label', type=str)
    parser.add_argument(
        '--kitti',
        dest='kitti',
        help='path to bbox file in kitti format',
        type=str)
    parser.add_argument(
        '--save',
        dest='save',
        help='whether image should be saved',
        action='store_true')
    parser.add_argument(
        '--title',
        dest='title',
        help='title of display window',
        type=str,
        default='test')
    parser.add_argument(
        '--keypoint', dest='keypoint', help='file of keypoints')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # test()
    #  img_name = '000000.png'
    args = parser_args()
    if args.img is not None:
        img = read_img(args.img)
    elif args.pkl is not None:
        img = read_pkl(args.pkl)
    # scales = np.array([2, 3, 4])
    # ratios = np.array([0.5, 1, 2])
    # anchors = generate_anchors(base_size=16, scales=scales, ratios=ratios)
    # anchors = expand_anchors(anchors)
    # print(anchors)
    # import ipdb
    # ipdb.set_trace()
    # anchors = shift_bbox(anchors,translation=(200,200))
    # analysis_boxes(anchors)
    # anchors = [[100,100,300,300]]
    # visualize_bbox(img, anchors)

    # read from kitti result file
    if args.label is not None:
        gt_boxes, class_names = read_kitti(args.label)
    else:
        gt_boxes = []
        class_names = []
    #  import ipdb
    #  ipdb.set_trace()
    boxes, class_names = read_kitti(args.kitti)

    if args.keypoint is not None:
        keypoints = read_keypoints(args.keypoint)
    else:
        keypoints = None
    visualize_bbox(
        img,
        boxes,
        gt_boxes,
        class_names,
        save=True,
        title=args.title,
        keypoints=keypoints)
