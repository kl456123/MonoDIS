import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from model.faster_rcnn.faster_rcnn import _fasterRCNN

class MobileDet(_fasterRCNN):
    def __init__(self, classes, class_agnostic):
        self.dout_base_model = 512
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)




