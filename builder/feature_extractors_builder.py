# -*- coding: utf-8 -*-

from core.models.feature_extractors.vggnet import VGGFeatureExtractor
from core.models.feature_extractors.resnet import ResNetFeatureExtractor


def build(feature_extractor_config):
    net_arch = feature_extractor_config['net_arch']
    if net_arch == 'resnet':
        return ResNetFeatureExtractor(feature_extractor_config)
    elif net_arch == 'vggnet':
        return VGGFeatureExtractor(feature_extractor_config)
