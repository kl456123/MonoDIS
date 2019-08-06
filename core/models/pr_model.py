# -*- coding: utf-8 -*-

from core.model import Model
from BDD_inf.models.PRNet import PRNet
from BDD_inf.cfgs.model_cfgs.retina_dla_bdd_cfg import ModelCFG
from BDD_inf.models.encoder import DataEncoder
from core.models.multibin_loss import MultiBinLoss
from core.pr_target_assigner import TargetAssigner
from core.samplers.balanced_sampler import BalancedSampler

from core.profiler import Profiler

import torch.nn as nn
import torch


class PRModel(Model):
    def init_weights(self):
        print("loading pre-trained weight")
        weight = torch.load(
            self.model_path, map_location=lambda storage, loc: storage)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        module_dict = self.det_model.state_dict()
        for k, v in weight.items():
            if k not in module_dict:
                continue
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        module_dict.update(new_state_dict)
        self.det_model.load_state_dict(module_dict)
        # else:

    # new_state_dict = OrderedDict()
    # for k, v in weight.items():
    # name = k[7:]      # remove `module.`
    # new_state_dict[name] = v
    # self.det_model.load_state_dict(new_state_dict)

    def pre_forward(self):
        self.freeze_modules()
        for param in self.det_model.multibox.box_3d_feature.parameters():
            param.requires_grad = True

        for param in self.det_model.multibox.orients_out.parameters():
            param.requires_grad = True

        for param in self.det_model.multibox.dims_3d_out.parameters():
            param.requires_grad = True

        self.freeze_bn(self)
        self.unfreeze_bn(self.det_model.multibox.box_3d_feature)
        self.unfreeze_bn(self.det_model.multibox.orients_out)
        self.unfreeze_bn(self.det_model.multibox.dims_3d_out)

    def init_param(self, model_config):
        self.n_classes = len(model_config['classes']) + 1
        self.rcnn_batch_size = model_config['rcnn_batch_size']
        self.profiler = Profiler()
        self.encoder = DataEncoder(
            ModelCFG, anchor_type=ModelCFG['anchor_type'], infer_mode=True)
        self.num_bins = 2

        self.model_path = model_config['model_path']

        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        self.sampler = BalancedSampler(model_config['sampler_config'])

    def init_modules(self):
        self.det_model = PRNet(ModelCFG)

        # dims loss
        self.dims_loss = nn.SmoothL1Loss(reduce=False)

        # multibin loss
        self.multibin_loss = MultiBinLoss(self.num_bins)

    def forward(self, feed_dict):
        self.target_assigner.bbox_coder_3d.mean_dims = feed_dict['mean_dims']
        image = feed_dict['img']
        loc1_preds, loc2_preds, os_preds, cls_preds,\
            dims_3d_out, orients_out = self.det_model.forward(
            image)

        # if not self.training:
            # boxes, lbls, scores, has_obj = self.encoder.decode(
                # loc2_preds.data.squeeze(0), F.softmax(cls_preds.squeeze(0), dim=1).data, os_preds.squeeze(0), Nt=0.5)

        prediction_dict = {}
        prediction_dict['dims_3d_out'] = dims_3d_out
        prediction_dict['orients_out'] = orients_out

        # prediction_dict['rcnn_cls_probs'] = scores
        # prediction_dict['rcnn_bbox_pred'] =



        return prediction_dict

    def generate_anchors(self, im_shape):
        default_boxes = self.encoder.default_boxes
        xymin = default_boxes[:, :2] - 0.5 * default_boxes[:, 2:]
        xymax = default_boxes[:, :2] + 0.5 * default_boxes[:, 2:]

        xymin = xymin
        xymax = xymax

        normalized_anchors = torch.cat([xymin, xymax], dim=-1)
        anchors = torch.zeros_like(normalized_anchors)
        anchors[:, ::2] = normalized_anchors[:, ::2] * im_shape[1]
        anchors[:, 1::2] = normalized_anchors[:, 1::2] * im_shape[0]

        return anchors

    def squeeze_bbox_preds(self, rcnn_bbox_preds, rcnn_cls_targets, out_c=4):
        """
        squeeze rcnn_bbox_preds from shape (N, 4 * num_classes) to shape (N, 4)
        Args:
            rcnn_bbox_preds: shape(N, num_classes, 4)
            rcnn_cls_targets: shape(N, 1)
        """
        rcnn_bbox_preds = rcnn_bbox_preds.view(-1, self.n_classes, out_c)
        batch_size = rcnn_bbox_preds.shape[0]
        offset = torch.arange(0, batch_size) * rcnn_bbox_preds.size(1)
        rcnn_cls_targets = rcnn_cls_targets + offset.type_as(rcnn_cls_targets)
        rcnn_bbox_preds = rcnn_bbox_preds.contiguous().view(
            -1, out_c)[rcnn_cls_targets[0]].unsqueeze(0)
        return rcnn_bbox_preds

    def loss(self, prediction_dict, feed_dict):
        #  import ipdb
        #  ipdb.set_trace()
        loss_dict = {}

        anchors = self.generate_anchors(feed_dict['im_info'][0][:2])

        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']
        local_angle = feed_dict['local_angle']
        gt_boxes_3d = feed_dict['gt_boxes_3d']

        gt_boxes_3d = torch.cat([gt_boxes_3d[:, :, :3], local_angle], dim=-1)

        rcnn_cls_targets, rcnn_reg_targets,\
            rcnn_cls_weights, rcnn_reg_weights,\
            rcnn_reg_targets_3d, rcnn_reg_weights_3d = self.target_assigner.assign(
            anchors.unsqueeze(0), gt_boxes, gt_boxes_3d, gt_labels)

        pos_indicator = rcnn_reg_weights > 0
        indicator = rcnn_cls_weights > 0

        # rpn_cls_probs = prediction_dict['rpn_cls_probs'][:, :, 1]
        cls_criterion = None

        batch_sampled_mask = self.sampler.subsample_batch(
            self.rcnn_batch_size,
            pos_indicator,
            criterion=cls_criterion,
            indicator=indicator)
        batch_sampled_mask = batch_sampled_mask.type_as(rcnn_cls_weights)
        rcnn_reg_weights_3d = rcnn_reg_weights_3d * batch_sampled_mask
        num_reg_coeff = (rcnn_reg_weights_3d > 0).sum(dim=1)

        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones([]).type_as(num_reg_coeff)

        rcnn_reg_weights_3d = rcnn_reg_weights_3d / num_reg_coeff.float()

        # dims loss
        dims_pred = self.squeeze_bbox_preds(prediction_dict['dims_3d_out'],
                                            rcnn_cls_targets, 3)
        dims_loss = self.dims_loss(dims_pred, rcnn_reg_targets_3d[:, :, :3])
        dims_loss = dims_loss * rcnn_reg_weights_3d.unsqueeze(-1)
        dims_loss = dims_loss.sum(dim=-1).sum(dim=-1)

        # multibin loss
        orient_loss, angle_tp_mask = self.multibin_loss(
            prediction_dict['orients_out'], rcnn_reg_targets_3d[:, :, -1:])

        orient_loss = orient_loss * rcnn_reg_weights_3d
        orient_loss = orient_loss.sum(dim=-1)

        loss_dict['dims_loss'] = dims_loss
        loss_dict['orient_loss'] = orient_loss
        prediction_dict['rcnn_reg_weights'] = rcnn_reg_weights_3d[
            batch_sampled_mask > 0]

        # angles stats
        angle_tp_mask = angle_tp_mask[rcnn_reg_weights_3d > 0]
        angles_tp_num = angle_tp_mask.int().sum().item()
        angles_all_num = angle_tp_mask.numel()

        self.target_assigner.stat.update({
            'cls_orient_2s_all_num': angles_all_num,
            'cls_orient_2s_tp_num': angles_tp_num
        })

        return loss_dict
