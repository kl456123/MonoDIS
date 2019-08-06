# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from utils import box_ops
from lib.model.nms.nms_wrapper import nms
from torch.autograd import Function


class Proposal(Function):
    def init_param(self, model_config):
        pass

    def forward(self, rpn_cls_probs, anchors, rpn_bbox_preds, im_info):
        # TODO create a new Function
        """
        Args:
        rpn_cls_probs: FloatTensor,shape(N,2*num_anchors,H,W)
        rpn_bbox_preds: FloatTensor,shape(N,num_anchors*4,H,W)
        anchors: FloatTensor,shape(N,4,H,W)

        Returns:
        proposals_batch: FloatTensor, shape(N,post_nms_topN,4)
        fg_probs_batch: FloatTensor, shape(N,post_nms_topN)
        """
        # assert len(
        # rpn_bbox_preds) == 1, 'just one feature maps is supported now'
        # rpn_bbox_preds = rpn_bbox_preds[0]
        anchors = anchors[0]
        # do not backward
        anchors = anchors
        rpn_cls_probs = rpn_cls_probs.detach()
        rpn_bbox_preds = rpn_bbox_preds.detach()

        batch_size = rpn_bbox_preds.shape[0]
        rpn_bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous()
        # shape(N,H*W*num_anchors,4)
        rpn_bbox_preds = rpn_bbox_preds.view(batch_size, -1, 4)
        # apply deltas to anchors to decode
        # loop here due to many features maps
        # proposals = []
        # for rpn_bbox_preds_single_map, anchors_single_map in zip(
        # rpn_bbox_preds, anchors):
        # proposals.append(
        # self.bbox_coder.decode(rpn_bbox_preds_single_map,
        # anchors_single_map))
        # proposals = torch.cat(proposals, dim=1)

        proposals = self.bbox_coder.decode_batch(rpn_bbox_preds, anchors)

        # filer and clip
        proposals = box_ops.clip_boxes(proposals, im_info)

        # fg prob
        fg_probs = rpn_cls_probs[:, self.num_anchors:, :, :]
        fg_probs = fg_probs.permute(0, 2, 3, 1).contiguous().view(batch_size,
                                                                  -1)

        # sort fg
        _, fg_probs_order = torch.sort(fg_probs, dim=1, descending=True)

        # fg_probs_batch = torch.zeros(batch_size,
        # self.post_nms_topN).type_as(rpn_cls_probs)
        proposals_batch = torch.zeros(batch_size, self.post_nms_topN,
                                      5).type_as(rpn_bbox_preds)
        proposals_order = torch.zeros(
            batch_size, self.post_nms_topN).fill_(-1).type_as(fg_probs_order)

        for i in range(batch_size):
            proposals_single = proposals[i]
            fg_probs_single = fg_probs[i]
            fg_order_single = fg_probs_order[i]
            # pre nms
            if self.pre_nms_topN > 0:
                fg_order_single = fg_order_single[:self.pre_nms_topN]
            proposals_single = proposals_single[fg_order_single]
            fg_probs_single = fg_probs_single[fg_order_single]

            # nms
            keep_idx_i = nms(
                torch.cat((proposals_single, fg_probs_single.unsqueeze(1)), 1),
                self.nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            # post nms
            if self.post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:self.post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            fg_probs_single = fg_probs_single[keep_idx_i]
            fg_order_single = fg_order_single[keep_idx_i]

            # padding 0 at the end.
            num_proposal = keep_idx_i.numel()
            proposals_batch[i, :, 0] = i
            proposals_batch[i, :num_proposal, 1:] = proposals_single
            # fg_probs_batch[i, :num_proposal] = fg_probs_single
            proposals_order[i, :num_proposal] = fg_order_single
        return proposals_batch, proposals_order
