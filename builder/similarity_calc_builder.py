# -*- coding: utf-8 -*-

from core.similarity_calc.scale_similarity_calc import ScaleSimilarityCalc
from core.similarity_calc.center_similarity_calc import CenterSimilarityCalc
from core.similarity_calc.distance_similarity_calc import DistanceSimilarityCalc
from core.similarity_calc.iod_similarity_calc import IoDSimilarityCalc
from core.similarity_calc.iog_similarity_calc import IoGSimilarityCalc


def build(similarity_calc_config):
    similarity_calc_type = similarity_calc_config['type']
    if similarity_calc_type == 'scale':
        return ScaleSimilarityCalc()
    elif similarity_calc_type == 'center':
        return CenterSimilarityCalc()
    elif similarity_calc_type == 'distance':
        return DistanceSimilarityCalc()
    elif similarity_calc_type == 'iod':
        return IoDSimilarityCalc()
    elif similarity_calc_type == 'iog':
        return IoGSimilarityCalc()
    else:
        raise ValueError('unsupported type of similarity_calc!')
