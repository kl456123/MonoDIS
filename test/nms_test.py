# -*- coding: utf-8 -*-
import sys
sys.path.append('./lib')
from model.nms.nms_wrapper import nms
import torch

dets = [[0, 0, 10, 10, 0.7], [0, 0, 10, 6, 0.1]]

dets = torch.Tensor(dets)
dets = dets.cuda()

keep = nms(dets, 0.7)

print(keep)
