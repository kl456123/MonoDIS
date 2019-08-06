# -*- coding: utf-8 -*-

import time
from torch.autograd import Variable
from lib.model.roi_layers import nms
from utils.visualize import save_pkl, visualize_bbox
# from utils.postprocess import mono_3d_postprocess_angle, mono_3d_postprocess_bbox, mono_3d_postprocess_depth
from utils.parallel_postprocess import mono_3d_postprocess_bbox
from utils.kitti_util import proj_3dTo2d
import numpy as np
import torch
import os
import sys


def test(eval_config, data_loader, model):
    #  oft_test(eval_config, data_loader, model)
    #  mono_test_keypoint(eval_config, data_loader, model)
    mono_test(eval_config, data_loader, model)
    # test_2d(eval_config, data_loader, model)


def to_cuda(target):
    if isinstance(target, list):
        return [to_cuda(e) for e in target]
    elif isinstance(target, dict):
        return {key: to_cuda(target[key]) for key in target}
    elif isinstance(target, torch.Tensor):
        return target.cuda()


def oft_test(eval_config, data_loader, model):
    """
    Only one image in batch is supported
    """
    # import ipdb
    # ipdb.set_trace()
    num_samples = len(data_loader)
    end_time = 0
    for i, data in enumerate(data_loader):
        img_file = data['img_name']
        start_time = time.time()

        with torch.no_grad():
            data = to_cuda(data)
            prediction = model(data)

        if eval_config.get('feat_vis'):
            featmaps_dict = model.get_feat()
            from utils.visualizer import FeatVisualizer
            feat_visualizer = FeatVisualizer()
            feat_visualizer.visualize_maps(featmaps_dict)

        pred_probs_3d = prediction['pred_probs_3d']
        pred_boxes_3d = prediction['pred_boxes_3d']

        duration_time = time.time() - start_time

        scores = pred_probs_3d.squeeze()
        pred_boxes_3d = pred_boxes_3d.squeeze()

        classes = eval_config['classes']
        thresh = eval_config['thresh']
        thresh = 0.1
        #  import ipdb
        #  ipdb.set_trace()

        dets = []
        # nms
        for j in range(1, len(classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                cls_scores, order = torch.sort(cls_scores, 0, True)
                if eval_config['class_agnostic']:
                    pred_boxes_3d = pred_boxes_3d[inds, :]
                else:
                    pred_boxes_3d = pred_boxes_3d[inds][:, j * 4:(j + 1) * 4]

                pred_boxes_3d = pred_boxes_3d[order]

                # keep = nms(pred_boxes_3d, eval_config['nms'])

                # pred_boxes_3d = pred_boxes_3d[keep.view(-1).long()]

                pred_boxes_3d = pred_boxes_3d.detach().cpu().numpy()
                p2 = data['orig_p2'][0].detach().cpu().numpy()
                cls_scores = cls_scores.cpu().numpy()

                cls_boxes = proj_3dTo2d(pred_boxes_3d, p2)

                # import ipdb
                # ipdb.set_trace()
                cls_dets = np.concatenate(
                    (cls_boxes, cls_scores[..., np.newaxis]), 1)

                # img filter(ignore outside of image)
                img_filter = get_img_filter(cls_dets)
                final_dets = np.concatenate([cls_dets, pred_boxes_3d], axis=-1)
                final_dets = final_dets[img_filter]
                dets.append(final_dets)

            else:
                dets.append([])

        save_dets(dets, img_file[0], 'kitti', eval_config['eval_out'])

        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


def get_img_filter(cls_dets):
    xmin = cls_dets[:, 0]
    ymin = cls_dets[:, 1]
    xmax = cls_dets[:, 0]
    ymax = cls_dets[:, 1]
    width_range = [-150, 2200]
    height_range = [-150, 1100]

    img_filter = (xmin > width_range[0]) & (xmax < width_range[1]) & (
        ymin > height_range[0]) & (ymax < height_range[1])
    return img_filter


def test_2d(eval_config, data_loader, model):
    """
    Only one image in batch is supported
    """
    num_samples = len(data_loader)
    for i, data in enumerate(data_loader):
        img_file = data['img_name']
        start_time = time.time()
        pred_boxes, scores, rois, anchors = im_detect_2d(
            model, to_cuda(data), eval_config, im_orig=data['img_orig'])
        duration_time = time.time() - start_time

        # import ipdb
        # ipdb.set_trace()
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        rois = rois.squeeze()

        classes = eval_config['classes']
        thresh = eval_config['thresh']
        # thresh = 0.1
        # import ipdb
        # ipdb.set_trace()

        dets = []
        res_rois = []
        res_anchors = []
        dets_3d = []

        n_classes = (len(classes) + 1)
        # nms
        #  import ipdb
        #  ipdb.set_trace()
        for j in range(1, len(classes) + 1):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[inds, j]

                if not eval_config['class_agnostic']:
                    pred_boxes_per_class = pred_boxes.contiguous().view(
                        -1, 4 * n_classes)[:, j * 4:(j + 1) * 4]
                    cls_boxes = pred_boxes_per_class[inds, :]
                else:
                    cls_boxes = pred_boxes[inds, :]
                #  rois_boxes = rois[inds, :]
                #  anchors_boxes = anchors[inds, :]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                #  rois_dets = torch.cat((rois_boxes, cls_scores.unsqueeze(1)), 1)
                #  anchors_dets = torch.cat(
                #  (anchors_boxes, cls_scores.unsqueeze(1)), 1)

                # sort
                _, order = torch.sort(cls_scores, 0, True)

                cls_dets = cls_dets[order]
                #  rois_dets = rois_dets[order]
                #  anchors_dets = anchors_dets[order]

                keep = nms(cls_dets, eval_config['nms'])

                cls_dets = cls_dets[keep.view(-1).long()]
                #  rois_dets = rois_dets[keep.view(-1).long()]
                #  anchors = anchors_dets[keep.view(-1).long()]

                #  res_rois.append(rois_dets.detach().cpu().numpy())
                #  res_anchors.append(anchors.detach().cpu().numpy())

                rcnn_3d = np.zeros((cls_dets.shape[0], 7))
                dets.append(np.concatenate([cls_dets, rcnn_3d], axis=-1))

            else:
                dets.append([])
                res_rois.append([])
                res_anchors.append([])
                dets_3d.append([])

        #  import ipdb
        #  ipdb.set_trace()
        save_dets(
            dets,
            img_file[0],
            'kitti',
            eval_config['eval_out'],
            classes_name=eval_config['classes'])

        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


def mono_test(eval_config, data_loader, model):
    """
    Only one image in batch is supported
    """
    num_samples = len(data_loader)
    end_time = 0
    for i, data in enumerate(data_loader):
        data_time = time.time() - end_time
        img_file = data['img_name']
        start_time = time.time()
        pred_boxes, scores, rois, anchors, rcnn_3d = im_detect(
            model, to_cuda(data), eval_config, im_orig=data['img_orig'])
        det_time = time.time() - start_time

        # import ipdb
        # ipdb.set_trace()
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        rois = rois.squeeze()
        rcnn_3d = rcnn_3d.squeeze()
        # anchors = anchors.squeeze()

        classes = eval_config['classes']
        thresh = eval_config['thresh']
        # print(thresh)
        # thresh = 0.3

        dets = []
        res_rois = []
        res_anchors = []
        dets_3d = []
        # import ipdb
        # ipdb.set_trace()
        # nms
        # new_scores = torch.zeros_like(scores)
        # _, scores_argmax = scores.max(dim=-1)
        # row = torch.arange(0, scores.shape[0]).type_as(scores_argmax)
        # new_scores[row, scores_argmax] = scores[row, scores_argmax]
        for j in range(1, len(classes) + 1):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            post_start_time = time.time()
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]

                cls_boxes = pred_boxes[inds, :]
                #  rois_boxes = rois[inds, :]
                #  anchors_boxes = anchors[inds, :]
                # if not eval_config['class_agnostic_3d']:
                #  rcnn_3d_dets = torch.cat(
                    #  [rcnn_3d[inds, j * 3:j * 3 + 3], rcnn_3d[inds, -4:]],
                    #  dim=-1)
                # else:
                rcnn_3d_dets = rcnn_3d[inds]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                #  rois_dets = torch.cat((rois_boxes, cls_scores.unsqueeze(1)), 1)
                #  anchors_dets = torch.cat(
                #  (anchors_boxes, cls_scores.unsqueeze(1)), 1)

                # sort
                _, order = torch.sort(cls_scores, 0, True)

                cls_dets = cls_dets[order]
                #  rois_dets = rois_dets[order]
                #  anchors_dets = anchors_dets[order]
                rcnn_3d_dets = rcnn_3d_dets[order]

                keep = nms(cls_dets[:, :4], cls_dets[:, -1],
                           eval_config['nms'])

                cls_dets = cls_dets[keep.view(-1).long()]
                #  rois_dets = rois_dets[keep.view(-1).long()]
                #  anchors = anchors_dets[keep.view(-1).long()]
                rcnn_3d_dets = rcnn_3d_dets[keep.view(-1).long()]

                cls_dets = cls_dets.detach().cpu().numpy()
                #  res_rois.append(rois_dets.detach().cpu().numpy())
                #  res_anchors.append(anchors.detach().cpu().numpy())

                coords = data['coords'][0].detach().cpu().numpy()
                gt_boxes = data['gt_boxes'][0].detach().cpu().numpy()
                gt_boxes_2d_proj = data['gt_boxes_proj'][0].detach().cpu(
                ).numpy()
                gt_boxes_3d = data['gt_boxes_3d'][0].detach().cpu().numpy()
                points_3d = data['points_3d'][0].detach().cpu().numpy()
                local_angles_gt = data['local_angle'][0].detach().cpu().numpy()
                local_angle_oritation_gt = data['local_angle_oritation'][
                    0].detach().cpu().numpy()
                encoded_side_points = data['encoded_side_points'][0].detach(
                ).cpu().numpy()
                points_3d = points_3d.T

                p2 = data['orig_p2'][0].detach().cpu().numpy()
                rcnn_3d_dets = rcnn_3d_dets.detach().cpu().numpy()
                cls_orient_gt = data['cls_orient'][0].detach().cpu().numpy()
                reg_orient_gt = data['reg_orient'][0].detach().cpu().numpy()
                # rcnn_3d_gt = rcnn_3d_gt.detach().cpu().numpy()

                # use gt
                use_gt = False
                post_time = 0

                if use_gt:
                    #  import ipdb
                    #  ipdb.set_trace()

                    #  center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
                    #  center_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
                    #  gt_boxes_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
                    #  gt_boxes_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
                    #  center = np.stack([center_x, center_y], axis=-1)
                    #  gt_boxes_dims = np.stack([gt_boxes_w, gt_boxes_h], axis=-1)

                    #  point1 = encoded_side_points[:, :2] * gt_boxes_dims + center
                    #  point2 = encoded_side_points[:, 2:] * gt_boxes_dims + center

                    #  global_angles_gt = gt_boxes_3d[:, -1:]

                    rcnn_3d_gt = np.concatenate(
                        [
                            gt_boxes_3d[:, :3], cls_orient_gt[..., np.newaxis],
                            reg_orient_gt
                        ],
                        axis=-1)
                    # just for debug
                    if len(gt_boxes):
                        cls_dets_gt = np.concatenate(
                            [gt_boxes, np.zeros_like(gt_boxes[:, -1:])],
                            axis=-1)
                        cls_dets_2d_proj_gt = np.concatenate(
                            [
                                gt_boxes_2d_proj,
                                np.zeros_like(gt_boxes[:, -1:])
                            ],
                            axis=-1)
                        rcnn_3d_gt, _ = mono_3d_postprocess_bbox(
                            rcnn_3d_gt, cls_dets_2d_proj_gt, p2)

                        dets.append(
                            np.concatenate(
                                [cls_dets_2d_proj_gt, rcnn_3d_gt], axis=-1))
                    else:
                        dets.append([])
                        res_rois.append([])
                        res_anchors.append([])
                        dets_3d.append([])
                else:
                    # import ipdb
                    # ipdb.set_trace()
                    # sample_name = os.path.splitext(os.path.basename(data['img_name'][0]))[0]
                    # if sample_name=='000031':
                    # import ipdb
                    # ipdb.set_trace()
                    #  rcnn_3d[:, :-1] = gt_boxes_3d[:, :3]
                    # global_angles_gt = gt_boxes_3d[:, -1:]
                    # rcnn_3d = np.concatenate(
                    # [gt_boxes_3d[:, :3], global_angles_gt], axis=-1)
                    # rcnn_3d[:,3] = 1-rcnn_3d[:,3]
                    # rcnn_3d_dets, location = mono_3d_postprocess_bbox(
                        # rcnn_3d_dets, cls_dets, p2)

                    post_time = time.time() - post_start_time
                    # rcnn_3d = mono_3d_postprocess_angle(rcnn_3d, cls_dets, p2)
                    # rcnn_3d = mono_3d_postprocess_depth(rcnn_3d, cls_dets, p2)
                    # rcnn_3d[:, 3:6] = location
                    # rcnn_3d = np.zeros((cls_dets.shape[0], 7))
                    dets.append(
                        np.concatenate(
                            [cls_dets, rcnn_3d_dets], axis=-1))

            else:
                dets.append([])
                res_rois.append([])
                res_anchors.append([])
                dets_3d.append([])
                post_time = 0


        duration_time = time.time() - end_time

        # import ipdb
        # ipdb.set_trace()
        save_dets(
            dets,
            img_file[0],
            'kitti',
            eval_config['eval_out'],
            classes_name=eval_config['classes'])
        # save_dets(res_rois[0], img_file[0], 'kitti',
        # eval_config['eval_out_rois'])
        # save_dets(res_anchors[0], img_file[0], 'kitti',
        # eval_config['eval_out_anchors'])

        sys.stdout.write(
            '\r{}/{},duration: {}, det_time: {}, post_time: {}, data_time: {}'.
            format(i + 1, num_samples, duration_time, det_time, post_time,
                   data_time))
        sys.stdout.flush()

        end_time = time.time()


def mono_test_keypoint(eval_config, data_loader, model):
    """
    Only one image in batch is supported
    """
    num_samples = len(data_loader)
    for i, data in enumerate(data_loader):
        img_file = data['img_name']
        start_time = time.time()
        pred_boxes, scores, rois, anchors, rcnn_3d, keypoints = im_detect(
            model, to_cuda(data), eval_config, im_orig=data['img_orig'])
        duration_time = time.time() - start_time

        # import ipdb
        # ipdb.set_trace()
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        rois = rois.squeeze()
        rcnn_3d = rcnn_3d.squeeze()
        keypoints = keypoints.squeeze()
        # anchors = anchors.squeeze()

        classes = eval_config['classes']
        thresh = eval_config['thresh']

        dets = []
        res_rois = []
        res_anchors = []
        dets_3d = []
        keypoint_dets = []
        # import ipdb
        # ipdb.set_trace()
        # nms
        for j in range(1, len(classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if eval_config['class_agnostic']:
                    cls_boxes = pred_boxes[inds, :]
                    rois_boxes = rois[inds, :]
                    anchors_boxes = anchors[inds, :]
                    rcnn_3d = rcnn_3d[inds]
                    keypoints = keypoints[inds]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                rois_dets = torch.cat((rois_boxes, cls_scores.unsqueeze(1)), 1)
                anchors_dets = torch.cat(
                    (anchors_boxes, cls_scores.unsqueeze(1)), 1)

                cls_dets = cls_dets[order]
                rois_dets = rois_dets[order]
                anchors_dets = anchors_dets[order]
                rcnn_3d = rcnn_3d[order]
                keypoints = keypoints[order]

                keep = nms(cls_dets, eval_config['nms'])

                cls_dets = cls_dets[keep.view(-1).long()]
                rois_dets = rois_dets[keep.view(-1).long()]
                anchors = anchors_dets[keep.view(-1).long()]
                rcnn_3d = rcnn_3d[keep.view(-1).long()]
                keypoints = keypoints[keep.view(-1).long()]

                cls_dets = cls_dets.detach().cpu().numpy()
                res_rois.append(rois_dets.detach().cpu().numpy())
                res_anchors.append(anchors.detach().cpu().numpy())

                coords = data['coords'][0].detach().cpu().numpy()
                gt_boxes = data['gt_boxes'][0].detach().cpu().numpy()
                gt_boxes_3d = data['gt_boxes_3d'][0].detach().cpu().numpy()
                points_3d = data['points_3d'][0].detach().cpu().numpy()
                local_angles_gt = data['local_angle'][0].detach().cpu().numpy()
                local_angle_oritation_gt = data['local_angle_oritation'][
                    0].detach().cpu().numpy()
                encoded_side_points = data['encoded_side_points'][0].detach(
                ).cpu().numpy()
                points_3d = points_3d.T

                p2 = data['p2'][0].detach().cpu().numpy()
                rcnn_3d = rcnn_3d.detach().cpu().numpy()
                keypoints = keypoints.detach().cpu().numpy()
                # rcnn_3d_gt = rcnn_3d_gt.detach().cpu().numpy()

                # use gt
                use_gt = False

                if use_gt:
                    keypoints_gt = data['keypoint_gt'][0].detach().cpu().numpy()
                    #  import ipdb
                    #  ipdb.set_trace()

                    center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
                    center_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
                    gt_boxes_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
                    gt_boxes_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
                    center = np.stack([center_x, center_y], axis=-1)
                    gt_boxes_dims = np.stack([gt_boxes_w, gt_boxes_h], axis=-1)

                    point1 = encoded_side_points[:, :2] * gt_boxes_dims + center
                    point2 = encoded_side_points[:, 2:] * gt_boxes_dims + center

                    global_angles_gt = gt_boxes_3d[:, -1:]

                    rcnn_3d_gt = np.concatenate(
                        [gt_boxes_3d[:, :3], point1, point2], axis=-1)
                    # just for debug
                    if len(rcnn_3d_gt):
                        cls_dets_gt = np.concatenate(
                            [gt_boxes, np.zeros_like(gt_boxes[:, -1:])],
                            axis=-1)
                        rcnn_3d_gt, _ = mono_3d_postprocess_bbox(
                            rcnn_3d_gt, cls_dets_gt, p2)

                        dets.append(
                            np.concatenate(
                                [cls_dets_gt, rcnn_3d_gt], axis=-1))
                        keypoint_dets.append(keypoints_gt)
                    else:
                        dets.append([])
                        res_rois.append([])
                        res_anchors.append([])
                        dets_3d.append([])
                        keypoint_dets.append([])
                else:
                    # import ipdb
                    # ipdb.set_trace()
                    # sample_name = os.path.splitext(os.path.basename(data['img_name'][0]))[0]
                    # if sample_name=='000031':
                    # import ipdb
                    # ipdb.set_trace()
                    #  rcnn_3d[:, :-1] = gt_boxes_3d[:, :3]
                    # global_angles_gt = gt_boxes_3d[:, -1:]
                    # rcnn_3d = np.concatenate(
                    # [gt_boxes_3d[:, :3], global_angles_gt], axis=-1)
                    # rcnn_3d[:,3] = 1-rcnn_3d[:,3]
                    rcnn_3d, location = mono_3d_postprocess_bbox(rcnn_3d,
                                                                 cls_dets, p2)
                    # rcnn_3d = mono_3d_postprocess_angle(rcnn_3d, cls_dets, p2)
                    # rcnn_3d = mono_3d_postprocess_depth(rcnn_3d, cls_dets, p2)
                    # rcnn_3d[:, 3:6] = location
                    # rcnn_3d = np.zeros((cls_dets.shape[0], 7))
                    dets.append(np.concatenate([cls_dets, rcnn_3d], axis=-1))
                    keypoints = keypoints.reshape((keypoints.shape[0], -1))
                    keypoint_dets.append(keypoints)

            else:
                dets.append([])
                res_rois.append([])
                res_anchors.append([])
                dets_3d.append([])
                keypoint_dets.append([])

        # import ipdb
        # ipdb.set_trace()
        save_dets(dets, img_file[0], 'kitti', eval_config['eval_out'])
        save_keypoints(keypoint_dets[0], img_file[0])
        # save_dets(res_rois[0], img_file[0], 'kitti',
        # eval_config['eval_out_rois'])
        # save_dets(res_anchors[0], img_file[0], 'kitti',
        # eval_config['eval_out_anchors'])

        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


def save_keypoints(keypoint_dets, label_info,
                   output_dir='./results/keypoints'):
    # import ipdb
    # ipdb.set_trace()

    class_name = 'Car'
    label_info = os.path.basename(label_info)
    label_idx = os.path.splitext(label_info)[0]
    label_file = label_idx + '.txt'
    label_path = os.path.join(output_dir, label_file)
    res_str = []
    # kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.8f}'
    # keypoint_dets = keypoint_dets.reshape((keypoint_dets.shape[0], -1))
    with open(label_path, 'w') as f:
        for det in keypoint_dets:
            det = det.tolist()
            det = [str(item) for item in det]
            res_str.append(' '.join(det))
        f.write('\n'.join(res_str))


def decode_3d(rcnn_3ds, boxes_2d):
    """
    Args:
        rcnn_3ds: shape(N,7)
    """
    center_x = (boxes_2d[:, 2] + boxes_2d[:, 0]) / 2
    center_y = (boxes_2d[:, 3] + boxes_2d[:, 1]) / 2
    center = np.expand_dims(np.stack([center_x, center_y], axis=-1), axis=1)
    w = (boxes_2d[:, 2] - boxes_2d[:, 0] + 1)
    h = (boxes_2d[:, 3] - boxes_2d[:, 1] + 1)
    dims = np.expand_dims(np.stack([w, h], axis=-1), axis=1)
    rcnn_coords = rcnn_3ds[:, :-1].reshape((-1, 3, 2))
    rcnn_coords = rcnn_coords * dims + center

    y = rcnn_3ds[:, -1:] * dims[:, 0, 1:] + center[:, 0, 1:]
    return np.concatenate([rcnn_coords.reshape((-1, 6)), y], axis=-1)


def im_detect_2d(model, data, eval_config, im_orig=None):
    im_info = data['im_info']
    with torch.no_grad():
        prediction = model(data)

    if eval_config.get('feat_vis'):
        featmaps_dict = model.get_feat()
        from utils.visualizer import FeatVisualizer
        feat_visualizer = FeatVisualizer()
        feat_visualizer.visualize_maps(featmaps_dict)

    cls_prob = prediction['rcnn_cls_probs']
    rois = prediction['rois_batch']
    bbox_pred = prediction['rcnn_bbox_preds']
    anchors = prediction['second_rpn_anchors'][0]

    scores = cls_prob
    im_scale = im_info[0][2]
    boxes = rois.data[:, :, 1:5]
    if prediction.get('rois_scores') is not None:
        rois_scores = prediction['rois_scores']
        boxes = torch.cat([boxes, rois_scores], dim=2)

    # visualize rois
    if im_orig is not None and eval_config['rois_vis']:
        visualize_bbox(im_orig.numpy()[0], boxes.cpu().numpy()[0], save=True)
        # visualize_bbox(im_orig.numpy()[0], anchors[0].cpu().numpy()[:100], save=True)

    if eval_config['bbox_reg']:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        #  if eval_config['bbox_normalize_targets_precomputed']:
        #  # Optionally normalize targets by a precomputed mean and stdev
        if not eval_config['class_agnostic']:
            boxes = boxes.repeat(1, 1, len(eval_config['classes']) + 1)
        #  box_deltas = box_deltas.view(
        #  -1, 4) * torch.FloatTensor(eval_config[
        #  'bbox_normalize_stds']).cuda() + torch.FloatTensor(
        #  eval_config['bbox_normalize_means']).cuda()
        #  box_deltas = box_deltas.view(eval_config['batch_size'], -1, 4)
        #  else:
        #  box_deltas = box_deltas.view(
        #  -1, 4) * torch.FloatTensor(eval_config[
        #  'bbox_normalize_stds']).cuda() + torch.FloatTensor(
        #  eval_config['bbox_normalize_means']).cuda()
        #  box_deltas = box_deltas.view(eval_config['batch_size'], -1,
        #  4 * len(eval_config['classes']))

        #  pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)

        pred_boxes = model.target_assigner.bbox_coder.decode_batch(
            box_deltas.view(eval_config['batch_size'], -1, 4),
            boxes.view(eval_config['batch_size'], -1, 4))
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    pred_boxes /= im_scale

    return pred_boxes, scores, rois[:, :, 1:5], anchors


def im_detect(model, data, eval_config, im_orig=None):
    im_info = data['im_info']
    with torch.no_grad():
        prediction = model(data)

    if eval_config.get('feat_vis'):
        featmaps_dict = model.get_feat()
        from utils.visualizer import FeatVisualizer
        feat_visualizer = FeatVisualizer()
        feat_visualizer.visualize_maps(featmaps_dict)

    cls_prob = prediction['rcnn_cls_probs']
    rois = prediction['rois_batch']
    bbox_pred = prediction['rcnn_bbox_preds']
    anchors = prediction['second_rpn_anchors'][0]
    rcnn_3d = prediction['rcnn_3d']

    scores = cls_prob
    im_scale = im_info[0][2]
    boxes = rois.data[:, :, 1:5]
    if prediction.get('rois_scores') is not None:
        rois_scores = prediction['rois_scores']
        boxes = torch.cat([boxes, rois_scores], dim=2)

    # visualize rois
    if im_orig is not None and eval_config['rois_vis']:
        visualize_bbox(im_orig.numpy()[0], boxes.cpu().numpy()[0], save=True)
        # visualize_bbox(im_orig.numpy()[0], anchors[0].cpu().numpy()[:100], save=True)

    if eval_config['bbox_reg']:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        #  if eval_config['bbox_normalize_targets_precomputed']:
        #  # Optionally normalize targets by a precomputed mean and stdev
        #  if eval_config['class_agnostic']:
        #  box_deltas = box_deltas.view(
        #  -1, 4) * torch.FloatTensor(eval_config[
        #  'bbox_normalize_stds']).cuda() + torch.FloatTensor(
        #  eval_config['bbox_normalize_means']).cuda()
        #  box_deltas = box_deltas.view(eval_config['batch_size'], -1, 4)
        #  else:
        #  box_deltas = box_deltas.view(
        #  -1, 4) * torch.FloatTensor(eval_config[
        #  'bbox_normalize_stds']).cuda() + torch.FloatTensor(
        #  eval_config['bbox_normalize_means']).cuda()
        #  box_deltas = box_deltas.view(eval_config['batch_size'], -1,
        #  4 * len(eval_config['classes']))

        #  pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = model.target_assigner.bbox_coder.decode_batch(
            box_deltas.view(eval_config['batch_size'], -1, 4), boxes)
        # pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        #  pred_boxes_2d_proj = model.target_assigner.bbox_coder.decode_batch(
    #  rcnn_3d[:, -4:].view(eval_config['batch_size'], -1, 4), boxes)

    pred_boxes /= im_scale
    #  pred_boxes_2d_proj /= im_scale

    if prediction.get('keypoints') is not None:
        keypoints = prediction['keypoints']
        return pred_boxes, scores, rois[:, :, 1:5], anchors, rcnn_3d, keypoints
    return pred_boxes, scores, rois[:, :, 1:5], anchors, rcnn_3d


def save_dets(dets,
              label_info,
              data_format='kitti',
              output_dir='',
              classes_name=['Car']):
    if data_format == 'kitti':
        save_dets_kitti(dets, label_info, output_dir, classes_name)
    else:
        raise ValueError('data format is not ')


def save_dets_kitti(dets, label_info, output_dir, classes_name=['Car']):
    #  class_name = 'Car'
    # import ipdb
    # ipdb.set_trace()
    label_info = os.path.basename(label_info)
    label_idx = os.path.splitext(label_info)[0]
    label_file = label_idx + '.txt'
    label_path = os.path.join(output_dir, label_file)
    res_str = []
    kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.8f}'
    with open(label_path, 'w') as f:
        for cls_ind, dets_per_classes in enumerate(dets):
            for det in dets_per_classes:
                xmin, ymin, xmax, ymax, cf, h, w, l, x, y, z, alpha = det
                res_str.append(
                    kitti_template.format(classes_name[cls_ind], xmin, ymin,
                                          xmax, ymax, h, w, l, x, y, z, alpha,
                                          cf))
        f.write('\n'.join(res_str))


def save_bev_map(bev_map, label_info, cache_dir):
    label_idx = os.path.splitext(label_info)[0][-6:]
    label_file = label_idx + '.pkl'
    pkl_path = os.path.join(cache_dir, label_file)
    save_pkl(bev_map.numpy(), pkl_path)
