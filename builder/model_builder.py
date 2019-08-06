# -*- coding: utf-8 -*-

# from core.models.iou_faster_rcnn_model import IoUFasterRCNN
# from core.models.faster_rcnn_model import FasterRCNN
# from core.models.two_rpn_model import TwoRPNModel
# from core.models.new_faster_rcnn_model import NewFasterRCNN
# from core.models.distance_faster_rcnn_model import DistanceFasterRCNN
# from core.models.refine_faster_rcnn_model import RefineFasterRCNN
# from core.models.gate_faster_rcnn_model import GateFasterRCNN
# from core.models.rfcn_model import RFCNModel
# from core.models.semantic_faster_rcnn_model import SemanticFasterRCNN
# from core.models.overlaps_faster_rcnn_model import OverlapsFasterRCNN
# from core.models.semantic_both_faster_rcnn_model import SemanticBothFasterRCNN
# from core.models.new_semantic_faster_rcnn_model import NewSemanticFasterRCNN
# from core.models.LED_model import LEDFasterRCNN
# from core.models.loss_faster_rcnn_model import LossFasterRCNN
# from core.models.single_iou_model import SingleIoUFasterRCNN
# from core.models.double_iou_faster_rcnn_model import DoubleIoUFasterRCNN
# from core.models.double_iou_faster_rcnn_model_slow import SlowDoubleIoUFasterRCNN
# from core.models.three_iou_faster_rcnn_model import ThreeIoUFasterRCNN
# from core.models.three_iou_faster_rcnn_model_org import OrgThreeIoUFasterRCNN
# from core.models.three_iou_faster_rcnn_model_org_ohem import OrgOHEMThreeIoUFasterRCNN
# from core.models.cascade_faster_rcnn_model import CascadeFasterRCNN
# from core.models.double_iou_faster_rcnn_model_second_stage import DoubleIoUSecondStageFasterRCNN
# from core.models.three_iou_faster_rcnn_model_org_ohem_second import OrgOHEMThreeIoUSecondStageFasterRCNN
# from core.models.fpn_faster_rcnn_model import FPNFasterRCNN
# from core.models.mono_3d import Mono3DFasterRCNN
from core.models.mono_3d_angle import Mono3DAngleFasterRCNN
# from core.models.mono_3d_angle_simpler import Mono3DAngleSimplerFasterRCNN
# from core.models.mono_3d_simpler import Mono3DSimplerFasterRCNN
# from core.models.mono_3d_angle_new import Mono3DAngleNewFasterRCNN
# from core.models.ssd_model import SSDModel
# from core.models.oft_model import OFTModel
# from core.models.mono_3d_better import Mono3DBetterFasterRCNN
# from core.models.refine_oft_model import RefineOFTModel
# from core.models.oft_4c_model import OFT4CModel
# from core.models.mono_3d_final import Mono3DFinalFasterRCNN
from core.models.mono_3d_final_plus import Mono3DFinalPlusFasterRCNN
from core.models.mono_3d_final_angle import Mono3DFinalAngleFasterRCNN

# class ModelBuilder(object):
# def __init__(self, model_config):
# self.model_config = model_config

# def build(self):
# pass

# choose class or function


def build(model_config, training=True):
    net_arch = model_config['net']
    if net_arch == 'vgg16':
        fasterRCNN = vgg16(model_config)
    elif net_arch == 'resnet50':
        fasterRCNN = resnet(
            model_config,
            training, )
    elif net_arch == 'faster_rcnn':
        fasterRCNN = FasterRCNN(model_config)
    elif net_arch == 'fpn':
        fasterRCNN = FPNFasterRCNN(model_config)
    elif net_arch == 'two_rpn':
        fasterRCNN = TwoRPNModel(model_config)
    elif net_arch == 'new_faster_rcnn':
        fasterRCNN = NewFasterRCNN(model_config)
    elif net_arch == 'distance_faster_rcnn':
        fasterRCNN = DistanceFasterRCNN(model_config)
    elif net_arch == 'refine_faster_rcnn':
        fasterRCNN = RefineFasterRCNN(model_config)
    elif net_arch == 'gate_faster_rcnn':
        fasterRCNN = GateFasterRCNN(model_config)
    elif net_arch == 'iou_faster_rcnn':
        fasterRCNN = IoUFasterRCNN(model_config)
    elif net_arch == 'rfcn':
        fasterRCNN = RFCNModel(model_config)
    elif net_arch == 'semantic':
        fasterRCNN = SemanticFasterRCNN(model_config)
    elif net_arch == 'overlaps':
        fasterRCNN = OverlapsFasterRCNN(model_config)
    elif net_arch == 'semantic_both':
        fasterRCNN = SemanticBothFasterRCNN(model_config)
    elif net_arch == 'new_semantic':
        fasterRCNN = NewSemanticFasterRCNN(model_config)
    elif net_arch == 'LED':
        fasterRCNN = LEDFasterRCNN(model_config)
    elif net_arch == 'loss':
        fasterRCNN = LossFasterRCNN(model_config)
    elif net_arch == 'single_iou':
        fasterRCNN = SingleIoUFasterRCNN(model_config)
    elif net_arch == 'double_iou':
        fasterRCNN = DoubleIoUFasterRCNN(model_config)
    elif net_arch == 'double_iou_slow':
        fasterRCNN = SlowDoubleIoUFasterRCNN(model_config)
    elif net_arch == 'three_iou':
        fasterRCNN = ThreeIoUFasterRCNN(model_config)
    elif net_arch == 'three_iou_org':
        fasterRCNN = OrgThreeIoUFasterRCNN(model_config)
    elif net_arch == 'three_iou_org_ohem':
        fasterRCNN = OrgOHEMThreeIoUFasterRCNN(model_config)
    elif net_arch == 'cascade':
        fasterRCNN = CascadeFasterRCNN(model_config)
    elif net_arch == 'double_iou_second':
        fasterRCNN = DoubleIoUSecondStageFasterRCNN(model_config)
    elif net_arch == 'three_iou_org_ohem_second':
        fasterRCNN = OrgOHEMThreeIoUSecondStageFasterRCNN(model_config)
    elif net_arch == 'mono_3d':
        fasterRCNN = Mono3DFasterRCNN(model_config)
    elif net_arch == 'multibin':
        fasterRCNN = Mono3DAngleFasterRCNN(model_config)
    elif net_arch == 'multibin_simpler':
        fasterRCNN = Mono3DAngleSimplerFasterRCNN(model_config)
    elif net_arch == 'mono_3d_simpler':
        fasterRCNN = Mono3DSimplerFasterRCNN(model_config)
    elif net_arch == 'multibin_new':
        fasterRCNN = Mono3DAngleNewFasterRCNN(model_config)
    elif net_arch == 'ssd':
        fasterRCNN = SSDModel(model_config)
    elif net_arch == 'oft':
        fasterRCNN = OFTModel(model_config)
    elif net_arch == 'mono_3d_better':
        fasterRCNN = Mono3DBetterFasterRCNN(model_config)
    elif net_arch == 'refine_oft':
        fasterRCNN = RefineOFTModel(model_config)
    elif net_arch == 'oft_4c':
        fasterRCNN = OFT4CModel(model_config)
    elif net_arch == 'mono_3d_final':
        fasterRCNN = Mono3DFinalFasterRCNN(model_config)
    elif net_arch == 'mono_3d_final_plus':
        fasterRCNN = Mono3DFinalPlusFasterRCNN(model_config)
    elif net_arch == 'pr_net':
        from core.models.pr_model import PRModel
        fasterRCNN = PRModel(model_config)
    elif net_arch == 'mono_3d_final_angle':
        fasterRCNN = Mono3DFinalAngleFasterRCNN(model_config)
    else:
        raise ValueError('net arch {} is not supported'.format(net_arch))

    if training:
        fasterRCNN.train()
    else:
        fasterRCNN.eval()

    fasterRCNN.pre_forward()
    # depercated
    # fasterRCNN.create_architecture()
    return fasterRCNN
