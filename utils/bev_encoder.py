# Encode target locations and labels.
import torch
from models.detection import DenseSSDPriorBox
import numpy as np


class DataEncoder(object):

    def __init__(self, cfg):
        super(DataEncoder, self).__init__()
        """Compute default box sizes with scale and aspect transform."""
        self.default_boxes = DenseSSDPriorBox()(cfg)

    @staticmethod
    def iou(box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4]; [[xmin, ymin, xmax, ymax], ...]
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """
        N = box1.size(0)
        M = box2.size(0)

        # max(xmin, ymin)
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
        )
        # min(xmax, ymax)
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2)   # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    @staticmethod
    def anchor_info(default_boxes, bev_map):
        """
        Args:
            default_boxes(np.array): Nx4 (xmin, ymin, xmax, ymax)
            bev_map(np.array): C X W X H
        Returns:
            labels
        """
        bev_map = bev_map.cpu().numpy()
        w, h = bev_map.shape[1:]
        default_boxes = np.clip(default_boxes, 0, 1)
        default_boxes[:, 0] *= w
        default_boxes[:, 2] *= w
        default_boxes[:, 1] *= h
        default_boxes[:, 3] *= h
        default_boxes = default_boxes.astype(np.int)
        xmin = default_boxes[:, 0].tolist()
        ymin = default_boxes[:, 1].tolist()
        xmax = default_boxes[: ,2].tolist()
        ymax = default_boxes[: ,3].tolist()

        anchors = [bev_map[-1, xmin[i]:xmax[i], ymin[i]:ymax[i]] for i in range(len(xmin))]
        labels = np.array([_.any() and np.max(_) > 0 for _ in anchors])

        return labels

    # def encode_offline(self, bev_map, boxes, ry, classes, threshold, if):

    def encode(self, bev_map, boxes, ry, classes, threshold=0.5, if_vis=False):
        '''Transform target bounding boxes and class labels to SSD boxes and classes. Match each object box
        to all the default boxes, pick the ones with the Jaccard-Index > 0.5:
            Jaccard(A,B) = AB / (A+B-AB)
        Args:
          boxes: (tensor) object bounding boxes (xmin,ymin,xmax,ymax) of a image, sized [#obj, 4].
          classes: (tensor) object class labels of a image, sized [#obj,].
          threshold: (float) Jaccard index threshold
        Returns:
          boxes: (tensor) bounding boxes, sized [#obj, 8732, 4].
          classes: (tensor) class labels, sized [8732,]
        '''
        default_boxes = self.default_boxes
        default_boxes_min_max = torch.cat([default_boxes[:, :2] - default_boxes[:, 2:]/2,
                                         default_boxes[:, :2] + default_boxes[:, 2:]/2], 1)
        gt_boxes = boxes

        iou = self.iou(boxes, default_boxes_min_max)

        default_boxes_min_max = default_boxes_min_max.cpu().numpy()
        labels = self.anchor_info(default_boxes_min_max, bev_map)

        iou, max_idx = iou.max(0)  # [1,8732]
        max_idx.squeeze_(0)        # [8732,]
        iou.squeeze_(0)            # [8732,]

        boxes = boxes[max_idx]     # [8732,4]
        ry = ry[max_idx] # [8732, 1]
        variances = [0.1, 0.2]
        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]  # [8732,2]
        cxcy /= variances[0] * default_boxes[:, 2:]
        wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]      # [8732,2]
        wh = torch.log(wh) / variances[1]
        loc = torch.cat([cxcy, wh], 1)  # [8732,4]

        conf = 1 + classes[max_idx]     # [8732,], background class = 0
        conf[iou < threshold] = -1      # background
        conf[iou < threshold-0.1] = 0
        conf = conf.cpu().numpy()
        conf[labels == False] = -1
        pos_boxes = default_boxes_min_max[conf > 0]
        conf = torch.from_numpy(conf).long()

        if if_vis:
            return gt_boxes.cpu().numpy(), pos_boxes

        ry = torch.FloatTensor(ry)
        ry = torch.cat([torch.cos(ry).unsqueeze(1), torch.sin(ry).unsqueeze(1)], 1)

        return loc, ry, conf

    def nms(self, bboxes, scores, threshold=0.5, mode='union'):
        '''Non maximum suppression.

        Args:
          bboxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.

        Returns:
          keep: (tensor) selected indices.

        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        '''
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        return torch.LongTensor(keep)

    def decode(self, loc, ry, conf):
        '''Transform predicted loc/conf back to real bbox locations and class labels.
        Args:
          loc: (tensor) predicted loc, sized [8732,4].
          conf: (tensor) predicted conf, sized [8732,21].
        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].
        '''
        has_obj = False
        variances = [0.1, 0.2]
        self.default_boxes = self.default_boxes.cuda()
        wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:]
        cxcy = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)  # [8732,4]

        max_conf, labels = conf.max(1)  # [8732,1]
        ids = labels.nonzero()
        tmp = ids.cpu().numpy()
        if tmp.__len__() > 0:
            print(('detected %d objs' % tmp.__len__()))
            ids = ids.squeeze(1)  # [#boxes,]
            has_obj = True
        else:
            return 0, 0, 0, 0, has_obj

        ry = torch.atan(ry[:, 0]/ry[:, 1])

        keep = self.nms(boxes[ids], max_conf[ids], threshold=0.5).cuda()
        return boxes[ids][keep], ry[ids], labels[ids][keep], max_conf[ids][keep], has_obj

