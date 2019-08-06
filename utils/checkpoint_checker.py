# -*- coding: utf-8 -*-
"""
Compare two checkpoints to inspect their difference


Conclusion: running_mean and running_var is different !
You should fix them if you want to finetune
"""

import torch


def check_eq(param1, param2):
    tmp = param1[param1 == param2]
    if tmp.size == param1.size:
        return True
    return False


ckpt1_name = '/data/object/liangxiong/mono_3d_angle_reg_2d/multibin_simpler/kitti/faster_rcnn_1_3257.pth'
ckpt2_name = '/data/object/liangxiong/mono_3d_angle_reg_2d/multibin_simpler/kitti/faster_rcnn_2_3257.pth'

# ckpt1 = torch.load(ckpt1_name)
# ckpt2 = torch.load(ckpt2_name)

# model_key = 'model'

# model1 = ckpt1[model_key]
# model2 = ckpt2[model_key]

# param_name = 'feature_extractor.first_stage_feature.0.weight'

# param1 = model1[param_name].cpu().numpy()
# param2 = model2[param_name].cpu().numpy()

# print('CHECK_EQ: ', check_eq(param1, param2))


def check_eq_module(ckpt1_name, ckpt2_name):
    # load modules
    ckpt1 = torch.load(ckpt1_name)
    ckpt2 = torch.load(ckpt2_name)
    model_key = 'model'
    model1 = ckpt1[model_key]
    model2 = ckpt2[model_key]

    for name, weights1 in model1.items():
        weights2 = model2[name].cpu().numpy()
        weights1 = weights1.cpu().numpy()
        eq = check_eq(weights1, weights2)
        if not eq:
            print('{} is not the same'.format(name))


if __name__ == '__main__':
    check_eq_module(ckpt1_name, ckpt2_name)
