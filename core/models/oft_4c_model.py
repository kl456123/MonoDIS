# -*- coding: utf-8 -*-
"""
use one stage detector as the framework to detect 3d object
in OFT feature map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model import Model
from core.models.feature_extractors.oft import OFTNetFeatureExtractor
from core.models.multibin_loss import MultiBinLoss
from core.voxel_generator import VoxelGenerator
from core.oft_4c_target_assigner import TargetAssigner as OFTargetAssigner
from core.target_assigner import TargetAssigner
from core.models.focal_loss import FocalLoss
from core.samplers.detection_sampler import DetectionSampler
from utils.integral_map import IntegralMapGenerator
from core.profiler import Profiler
from core.projector import Projector

from core import ops


class OFT4CModel(Model):
    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()

        self.profiler.start('1')
        self.voxel_generator.proj_voxels_3dTo2d(feed_dict['p2'],
                                                feed_dict['im_info'])
        self.profiler.end('1')

        self.profiler.start('2')
        img_feat_maps = self.feature_extractor.forward(feed_dict['img'])
        self.profiler.end('2')

        self.profiler.start('3')
        img_feat_maps = self.feature_preprocess(img_feat_maps)
        self.profiler.end('3')

        self.profiler.start('4')
        integral_maps = self.generate_integral_maps(img_feat_maps)
        self.profiler.end('4')

        # import ipdb
        # ipdb.set_trace()
        self.profiler.start('5')
        oft_maps = self.generate_oft_maps(integral_maps)
        self.profiler.end('5')

        self.profiler.start('6')
        bev_feat_maps = self.feature_extractor.bev_feature(oft_maps)
        self.profiler.end('6')

        # pred output
        # shape (NCHW)
        self.profiler.start('7')
        output_maps = self.output_head(bev_feat_maps)
        self.profiler.end('7')

        # shape(N,M,out_channels)
        pred_3d = output_maps.permute(0, 2, 3, 1).contiguous().view(
            self.batch_size, -1, self.output_channels)

        pred_boxes_3d = pred_3d[:, :, self.n_classes:]
        pred_scores_3d = pred_3d[:, :, :self.n_classes]

        pred_probs_3d = F.softmax(pred_scores_3d, dim=-1)
        # import ipdb
        # ipdb.set_trace()
        self.add_feat('pred_scores_3d', output_maps[:, 1:2, :, :])
        self.add_feat('bev_feat_map', bev_feat_maps)

        if not self.training:
            # import ipdb
            # ipdb.set_trace()
            voxel_centers = self.voxel_generator.voxel_centers
            D = self.voxel_generator.lattice_dims[1]
            voxel_centers = voxel_centers.view(-1, D, 3)[:, 0, :]
            # pred_boxes_3d = self.bbox_coder.decode_batch_bbox(voxel_centers,
            # pred_boxes_3d)
            # decode angle
            # angles_oritations = self.bbox_coder.decode_batch_angle(
            # pred_boxes_3d[:, :, 6:], self.angle_loss.bin_centers,
            # self.num_bins)

            pred_boxes_3d = self.bbox_coder.decode_batch_bbox(
                voxel_centers, pred_3d[:, :, 2:8])
            # import ipdb
            # ipdb.set_trace()
            angles_oritations = self.bbox_coder.decode_batch_angle_box_4c(
                pred_3d[:, :, 8:])
            # import ipdb
            # ipdb.set_trace()
            # random_value = torch.rand(angles_oritations.shape)
            # angles_oritations = random_value.type_as(
            # angles_oritations) * angles_oritations

            pred_boxes_3d = torch.cat([pred_boxes_3d, angles_oritations],
                                      dim=-1)

            # gussian filter probs map
            # reshape first
            shape = output_maps.shape[-2:]
            fg_mask = pred_probs_3d[0, :, 1].view(shape).detach().cpu().numpy()

            # then smooth
            from scipy.ndimage import gaussian_filter
            smoothed_fg_mask = gaussian_filter(fg_mask, sigma=self.nms_deltas)

            smoothed_fg_mask = torch.tensor(smoothed_fg_mask).type_as(
                pred_probs_3d)

            # nms
            smoothed_fg_mask = self.nms_map(smoothed_fg_mask)

            # assign back to tensor
            pred_probs_3d[0, :, 1] = smoothed_fg_mask.view(-1)

            # reset bg according to fg
            pred_probs_3d[0, :, 0] = 1 - pred_probs_3d[0, :, 1]

        prediction_dict = {}
        prediction_dict['pred_boxes_3d'] = pred_boxes_3d
        # prediction_dict['pred_scores_3d'] = pred_scores_3d
        prediction_dict['pred_probs_3d'] = pred_probs_3d

        return prediction_dict

    def nms_map(self, smoothed_fg_mask):
        """
        supress the neibor
        """

        directions = [-1, 0, 1]
        shape = smoothed_fg_mask.shape
        orig_index = (torch.arange(shape[0]).cuda().long(),
                      torch.arange(shape[1]).cuda().long())
        orig_index = ops.meshgrid(orig_index[1], orig_index[0])
        orig_index = [orig_index[1], orig_index[0]]
        dest_indexes = []
        for i in directions:
            for j in directions:
                dest_index = (orig_index[0] + directions[i],
                              orig_index[1] + directions[j])
                dest_indexes.append(dest_index)

        nms_filter = torch.ones_like(smoothed_fg_mask).byte()
        orig_fg_mask = smoothed_fg_mask

        # pad fg mask first to prevent out of boundary
        padded_smoothed_fg_mask = torch.zeros(
            (shape[0] + 1, shape[1] + 1)).type_as(smoothed_fg_mask)
        padded_smoothed_fg_mask[:-1, :-1] = smoothed_fg_mask

        # import ipdb
        # ipdb.set_trace()
        for dest_index in dest_indexes:
            nms_filter = nms_filter & (
                orig_fg_mask >=
                padded_smoothed_fg_mask[dest_index].view_as(orig_fg_mask))

        # surpress
        smoothed_fg_mask[~nms_filter] = 0
        return smoothed_fg_mask

    def feature_preprocess(self, feat_maps):
        # import ipdb
        # ipdb.set_trace()
        reduced_feat_maps = []
        for ind, feat_map in enumerate(feat_maps):
            reduced_feat_map = self.feats_reduces[ind](feat_map)
            reduced_feat_maps.append(reduced_feat_map)
        return reduced_feat_maps

    def generate_integral_maps(self, img_feat_maps):
        integral_maps = []
        for img_feat_map in img_feat_maps:
            integral_maps.append(
                self.integral_map_generator.generate(img_feat_map))

        return integral_maps

    def generate_oft_maps(self, integral_maps):
        # shape(N,4)
        normalized_voxel_proj_2d = self.voxel_generator.normalized_voxel_proj_2d
        # for i in range(voxel_proj_2d.shape[0]):
        multiscale_img_feat = []
        for integral_map in integral_maps:
            multiscale_img_feat.append(
                self.integral_map_generator.calc(integral_map,
                                                 normalized_voxel_proj_2d))

        # shape(N,C,HWD)
        fusion_feat = multiscale_img_feat[0] + multiscale_img_feat[
            1] + multiscale_img_feat[2]
        depth_dim = self.voxel_generator.lattice_dims[1]
        height_dim = self.voxel_generator.lattice_dims[0]

        fusion_feat = fusion_feat.view(
            self.batch_size, self.feat_size, -1,
            depth_dim).permute(0, 3, 1, 2).contiguous()
        # shape(N,C,HW)
        oft_maps = self.feat_collapse(fusion_feat).view(
            self.batch_size, self.feat_size, height_dim, -1)

        return oft_maps

    def init_param(self, model_config):

        self.feat_size = model_config['common_feat_size']
        self.batch_size = model_config['batch_size']
        self.sample_size = model_config['sample_size']
        self.n_classes = model_config['num_classes']
        self.use_focal_loss = model_config['use_focal_loss']
        self.feature_extractor_config = model_config['feature_extractor_config']

        self.voxel_generator = VoxelGenerator(
            model_config['voxel_generator_config'])
        self.voxel_generator.init_voxels()

        self.integral_map_generator = IntegralMapGenerator()

        self.oft_target_assigner = OFTargetAssigner(
            model_config['target_assigner_config'])

        self.target_assigner = TargetAssigner(
            model_config['eval_target_assigner_config'])
        self.target_assigner.analyzer.append_gt = False

        self.sampler = DetectionSampler(model_config['sampler_config'])

        self.bbox_coder = self.oft_target_assigner.bbox_coder

        # find the most expensive operators
        self.profiler = Profiler()

        # self.multibin = model_config['multibin']
        self.num_bins = model_config['num_bins']

        # box_4c encode format
        self.reg_channels = 3 + 3 + 10

        # score, pos, dim, ang
        self.output_channels = self.n_classes + self.reg_channels

        nms_deltas = model_config.get('nms_deltas')
        if nms_deltas is None:
            nms_deltas = 1
        self.nms_deltas = nms_deltas

    def init_modules(self):
        """
        some modules
        """

        self.feature_extractor = OFTNetFeatureExtractor(
            self.feature_extractor_config)

        feats_reduce_1 = nn.Conv2d(128, self.feat_size, 1, 1, 0)
        feats_reduce_2 = nn.Conv2d(256, self.feat_size, 1, 1, 0)
        feats_reduce_3 = nn.Conv2d(512, self.feat_size, 1, 1, 0)
        self.feats_reduces = nn.ModuleList(
            [feats_reduce_1, feats_reduce_2, feats_reduce_3])

        self.feat_collapse = nn.Conv2d(8, 1, 1, 1, 0)

        self.output_head = nn.Conv2d(256 * 4, self.output_channels, 1, 1, 0)

        # loss
        self.reg_loss = nn.L1Loss(reduce=False)
        # self.reg_loss = nn.SmoothL1Loss(reduce=False)
        # if self.use_focal_loss:
        # self.conf_loss = FocalLoss(
        # self.n_classes, alpha=0.2, gamma=2, auto_alpha=False)
        # else:
        # self.conf_loss = nn.CrossEntropyLoss(reduce=False)
        self.conf_loss = nn.L1Loss(reduce=False)

        self.angle_loss = MultiBinLoss(num_bins=self.num_bins)

    def init_weights(self):
        self.feature_extractor.init_weights()

    def loss(self, prediction_dict, feed_dict):
        self.profiler.start('8')
        gt_boxes_3d = feed_dict['gt_boxes_3d']
        gt_labels = feed_dict['gt_labels']
        gt_boxes_ground_2d_rect = feed_dict['gt_boxes_ground_2d_rect']

        voxels_ground_2d = self.voxel_generator.proj_voxels_to_ground()
        voxel_centers = self.voxel_generator.voxel_centers
        D = self.voxel_generator.lattice_dims[1]
        voxel_centers = voxel_centers.view(-1, D, 3)[:, 0, :]

        # gt_boxes_3d = torch.cat([gt_boxes_3d[:,:,:3],],dim=-1)

        cls_weights, reg_weights, cls_targets, reg_targets = self.oft_target_assigner.assign(
            voxels_ground_2d, gt_boxes_ground_2d_rect, voxel_centers,
            gt_boxes_3d, gt_labels)

        # pred_boxes_3d = prediction_dict['pred_boxes_3d']
        ################################
        # subsample
        ################################

        # pos_indicator = reg_weights > 0
        # indicator = cls_weights > 0

        # rpn_cls_probs = prediction_dict['pred_probs_3d'][:, :, 1]
        # cls_criterion = rpn_cls_probs

        # batch_sampled_mask = self.sampler.subsample_batch(
        # self.sample_size,
        # pos_indicator,
        # criterion=cls_criterion,
        # indicator=indicator)

        # import ipdb
        # ipdb.set_trace()
        # batch_sampled_mask = batch_sampled_mask.type_as(cls_weights)
        # rpn_cls_weights = cls_weights[batch_sampled_mask]
        # rpn_reg_weights = reg_weights[batch_sampled_mask]
        # cls_targets = cls_targets[batch_sampled_mask]
        # reg_targets = reg_targets[batch_sampled_mask]

        # num_cls_coeff = (rpn_cls_weights > 0).sum(dim=-1)
        # import ipdb
        # ipdb.set_trace()
        num_reg_coeff = (reg_weights > 0).sum(dim=-1)
        # # check
        # #  assert num_cls_coeff, 'bug happens'
        # #  assert num_reg_coeff, 'bug happens'
        # if num_cls_coeff == 0:
        # num_cls_coeff = torch.ones([]).type_as(num_cls_coeff)
        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones([]).type_as(num_reg_coeff)

        # import ipdb
        # ipdb.set_trace()
        # cls loss
        rpn_cls_probs = prediction_dict['pred_probs_3d'][:, :, -1]
        rpn_cls_loss = self.conf_loss(rpn_cls_probs, cls_targets)
        rpn_cls_loss = rpn_cls_loss.view_as(cls_weights)
        rpn_cls_loss = rpn_cls_loss * cls_weights
        rpn_cls_loss = rpn_cls_loss.mean(dim=-1)

        # bbox loss
        rpn_bbox_preds = prediction_dict['pred_boxes_3d']
        rpn_reg_loss = self.reg_loss(rpn_bbox_preds,
                                     reg_targets)
        rpn_reg_loss = rpn_reg_loss * reg_weights.unsqueeze(-1)
        num_reg_coeff = num_reg_coeff.type_as(reg_weights)

        # angle_loss
        # angle_loss, angle_tp_mask = self.angle_loss(rpn_bbox_preds[:, :, 6:],
        # reg_targets[:, :, -1:])
        # rpn_angle_loss = angle_loss * reg_weights

        # split reg loss
        dim_loss = rpn_reg_loss[:, :, :3].sum(dim=-1).sum(
            dim=-1) / num_reg_coeff
        pos_loss = rpn_reg_loss[:, :, 3:6].sum(dim=-1).sum(
            dim=-1) / num_reg_coeff
        angle_loss = rpn_reg_loss[:, :, 6:].sum(dim=-1).sum(
            dim=-1) / num_reg_coeff

        prediction_dict['rcnn_reg_weights'] = reg_weights

        loss_dict = {}

        loss_dict['rpn_cls_loss'] = rpn_cls_loss
        # loss_dict['rpn_bbox_loss'] = rpn_reg_loss
        # split bbox loss instead of fusing them
        loss_dict['dim_loss'] = dim_loss
        loss_dict['pos_loss'] = pos_loss
        loss_dict['angle_loss'] = angle_loss

        self.profiler.end('8')

        # recall
        # final_boxes = self.bbox_coder.decode_batch(rpn_bbox_preds, )
        # self.target_assigner.assign(final_boxes, gt_boxes)

        # import ipdb
        # ipdb.set_trace()
        voxel_centers = self.voxel_generator.voxel_centers
        D = self.voxel_generator.lattice_dims[1]
        voxel_centers = voxel_centers.view(-1, D, 3)[:, 0, :]
        # import ipdb
        # ipdb.set_trace()
        # decode bbox
        pred_boxes_3d = self.bbox_coder.decode_batch_bbox(
            voxel_centers, rpn_bbox_preds[:, :, :6])
        # decode angle
        angles_oritations = self.bbox_coder.decode_batch_angle_box_4c(
            rpn_bbox_preds[:, :, 6:])
        pred_boxes_3d = torch.cat([pred_boxes_3d, angles_oritations], dim=-1)

        # import ipdb
        # ipdb.set_trace()
        # select the top n
        order = torch.sort(rpn_cls_probs, descending=True)[1]
        topn = 1000
        order = order[:, :topn]
        rpn_cls_probs = rpn_cls_probs[0][order[0]].unsqueeze(0)
        pred_boxes_3d = pred_boxes_3d[0][order[0]].unsqueeze(0)

        target = {
            'dimension': pred_boxes_3d[0, :, :3],
            'location': pred_boxes_3d[0, :, 3:6],
            'ry': pred_boxes_3d[0, :, 6]
        }

        boxes_2d = Projector.proj_box_3to2img(target, feed_dict['p2'])
        gt_boxes = feed_dict['gt_boxes']
        num_gt = gt_labels.numel()
        self.target_assigner.assign(boxes_2d, gt_boxes, eval_thresh=0.7)

        fake_match = self.target_assigner.analyzer.match
        # import ipdb
        # ipdb.set_trace()
        self.target_assigner.analyzer.analyze_ap(
            fake_match, rpn_cls_probs, num_gt, thresh=0.1)

        # import ipdb
        # ipdb.set_trace()
        # angle stats

        return loss_dict
