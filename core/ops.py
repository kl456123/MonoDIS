# -*- coding: utf-8 -*-
import torch
import math


def meshgrid(a, b):
    """
    Args:
        a: tensor of shape(N)
        b: tensor of shape(M)
    Returns:
        two tensor of shape(M,N) and shape (M,N)
    """
    a_ = a.repeat(b.numel())
    b_ = b.repeat(a.numel(), 1).t().contiguous().view(-1)
    return a_, b_


def get_angle(y, x):
    """
        Args:
            sin: shape(N,num_bins)
        """
    theta = torch.atan2(y, x)
    return -theta

    #  sin = sin.detach()
    #  cos = cos.detach()
    #  norm = torch.sqrt(sin * sin + cos * cos)
    #  sin /= norm
    #  cos /= norm

    #  # range in [-pi, pi]
    #  theta = torch.asin(sin)
    #  cond_pos = (cos < 0) & (sin > 0)
    #  cond_neg = (cos < 0) & (sin < 0)
    #  theta[cond_pos] = math.pi - theta[cond_pos]
    #  theta[cond_neg] = -math.pi - theta[cond_neg]
    return theta


def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv
