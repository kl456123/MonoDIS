# -*- coding: utf-8 -*-

from core.model import Model
from core.models.feature_extractors.pyramid_vggnet import PyramidVggnetExtractor
from core.anchor_generators.anchor_generator import AnchorGenerator
from core.target_assigner import TargetAssigner
from core.samplers.detection_sampler import DetectionSampler
from core.prior_box import PriorBox
from core.models.focal_loss import FocalLoss
import torch.nn as nn
import torch
import torch.nn.functional as F


class SSDModel(Model):
    def init_param(self, model_config):
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.multibox_cfg = [3, 3, 3, 3, 3, 3]
        self.n_classes = len(model_config['classes'])
        self.sampler = DetectionSampler(model_config['sampler_config'])
        self.batch_size = model_config['batch_size']
        self.use_focal_loss = model_config['use_focal_loss']
        # self.multibox_cfg = model_config['multibox_config']

        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        # import ipdb
        # ipdb.set_trace()
        self.anchor_generator = AnchorGenerator(
            model_config['anchor_generator_config'])

        self.bbox_coder = self.target_assigner.bbox_coder

        # self.priorsbox = PriorBox(model_config['anchor_generator_config'])

    def init_modules(self):
        self.feature_extractor = PyramidVggnetExtractor(
            self.feature_extractor_config)

        # loc layers and conf layers
        base_feat = self.feature_extractor.base_feat
        extra_layers = self.feature_extractor.extras_layers
        loc_layers, conf_layers = self.make_multibox(base_feat, extra_layers)
        self.loc_layers = loc_layers
        self.conf_layers = conf_layers

        # self.rcnn_3d_preds = nn.Linear()

        # loss layers
        self.loc_loss = nn.SmoothL1Loss(reduce=False)

        if self.use_focal_loss:
            self.conf_loss = FocalLoss(
                self.n_classes, alpha=0.2, gamma=2, auto_alpha=False)
        else:
            self.conf_loss = nn.CrossEntropyLoss(reduce=False)

    def make_multibox(self, vgg, extra_layers):
        cfg = self.multibox_cfg
        num_classes = self.n_classes
        loc_layers = []
        conf_layers = []
        vgg_source = [21, -2]
        for k, v in enumerate(vgg_source):
            loc_layers += [
                nn.Conv2d(
                    vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)
            ]
            conf_layers += [
                nn.Conv2d(
                    vgg[v].out_channels,
                    cfg[k] * num_classes,
                    kernel_size=3,
                    padding=1)
            ]
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [
                nn.Conv2d(
                    v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)
            ]
            conf_layers += [
                nn.Conv2d(
                    v.out_channels,
                    cfg[k] * num_classes,
                    kernel_size=3,
                    padding=1)
            ]
        return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

    def init_weights(self):
        pass

    def forward(self, feed_dict):
        img = feed_dict['img']
        source_feats = self.feature_extractor(img)
        loc_preds = []
        conf_preds = []

        featmap_shapes = []

        # apply multibox head to source layers
        for (x, l, c) in zip(source_feats, self.loc_layers, self.conf_layers):
            loc_preds.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_preds.append(c(x).permute(0, 2, 3, 1).contiguous())
            featmap_shapes.append(x.size()[-2:])

        # import ipdb
        # ipdb.set_trace()
        loc_preds = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf_preds = torch.cat([o.view(o.size(0), -1) for o in conf_preds], 1)
        probs = F.softmax(
            conf_preds.view(conf_preds.size(0), -1, self.n_classes), dim=-1)
        loc_preds = loc_preds.view(loc_preds.size(0), -1, 4)

        # import ipdb
        # ipdb.set_trace()
        anchors = self.anchor_generator.generate_pyramid(featmap_shapes)
        # anchors = self.priorsbox.forward(featmap_shapes)

        # import ipdb
        # ipdb.set_trace()
        rois_batch_inds = torch.zeros_like(loc_preds[:, :, -1:])
        rois_batch = torch.cat([rois_batch_inds, anchors.unsqueeze(0)], dim=-1)
        second_rpn_anchors = anchors.unsqueeze(0)

        rcnn_3d = torch.zeros_like(loc_preds)

        prediction_dict = {
            'rcnn_bbox_preds': loc_preds,
            'rcnn_cls_scores': conf_preds,
            'anchors': anchors,
            'rcnn_cls_probs': probs,
            'rois_batch': rois_batch,
            'second_rpn_anchors': second_rpn_anchors,
            'rcnn_3d': rcnn_3d
        }
        return prediction_dict

    def loss(self, prediction_dict, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        # loss for cls
        loss_dict = {}

        gt_boxes = feed_dict['gt_boxes']

        anchors = prediction_dict['anchors']

        #################################
        # target assigner
        ################################
        # no need gt labels here,it just a binary classifcation problem
        # import ipdb
        # ipdb.set_trace()
        rpn_cls_targets, rpn_reg_targets, \
            rpn_cls_weights, rpn_reg_weights = \
            self.target_assigner.assign(anchors, gt_boxes, gt_labels=None)

        ################################
        # subsample
        ################################

        pos_indicator = rpn_reg_weights > 0
        indicator = rpn_cls_weights > 0

        rpn_cls_probs = prediction_dict['rcnn_cls_probs'][:, :, 1]
        cls_criterion = rpn_cls_probs

        batch_sampled_mask = self.sampler.subsample_batch(
            self.batch_size,
            pos_indicator,
            criterion=cls_criterion,
            indicator=indicator)
        batch_sampled_mask = batch_sampled_mask.type_as(rpn_cls_weights)
        rpn_cls_weights = rpn_cls_weights * batch_sampled_mask
        rpn_reg_weights = rpn_reg_weights * batch_sampled_mask
        num_cls_coeff = (rpn_cls_weights > 0).sum(dim=1)
        num_reg_coeff = (rpn_reg_weights > 0).sum(dim=1)
        # check
        #  assert num_cls_coeff, 'bug happens'
        #  assert num_reg_coeff, 'bug happens'
        if num_cls_coeff == 0:
            num_cls_coeff = torch.ones([]).type_as(num_cls_coeff)
        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones([]).type_as(num_reg_coeff)

        # cls loss
        rpn_cls_score = prediction_dict['rcnn_cls_scores']
        # rpn_cls_loss = self.rpn_cls_loss(rpn_cls_score, rpn_cls_targets)
        rpn_cls_loss = self.conf_loss(
            rpn_cls_score.view(-1, 2), rpn_cls_targets.view(-1))
        rpn_cls_loss = rpn_cls_loss.view_as(rpn_cls_weights)
        rpn_cls_loss *= rpn_cls_weights
        rpn_cls_loss = rpn_cls_loss.sum(dim=1) / num_cls_coeff.float()

        # bbox loss
        # shape(N,num,4)
        rpn_bbox_preds = prediction_dict['rcnn_bbox_preds']
        # rpn_bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous()
        # shape(N,H*W*num_anchors,4)
        # rpn_bbox_preds = rpn_bbox_preds.view(rpn_bbox_preds.shape[0], -1, 4)
        # import ipdb
        # ipdb.set_trace()
        rpn_reg_loss = self.loc_loss(rpn_bbox_preds, rpn_reg_targets)
        rpn_reg_loss *= rpn_reg_weights.unsqueeze(-1).expand(-1, -1, 4)
        rpn_reg_loss = rpn_reg_loss.view(rpn_reg_loss.shape[0], -1).sum(
            dim=1) / num_reg_coeff.float()

        prediction_dict['rcnn_reg_weights'] = rpn_reg_weights[
            batch_sampled_mask > 0]

        loss_dict['rpn_cls_loss'] = rpn_cls_loss
        loss_dict['rpn_bbox_loss'] = rpn_reg_loss

        # recall
        final_boxes = self.bbox_coder.decode_batch(rpn_bbox_preds, anchors)
        self.target_assigner.assign(final_boxes, gt_boxes)
        return loss_dict
