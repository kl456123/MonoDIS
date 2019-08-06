# -*- coding: utf-8 -*-

import torch
from core.utils import format_checker


def multidim_index(tensor, index):
    """
    Args:
        tensor: shape(N, M, K)
        index: shape(S, T)
    Returns:
        indexed_tensor: shape(S,T,K)
    """

    format_checker.check_tensor_shape(tensor, [None, None, None])
    format_checker.check_tensor_shape(index, [None, None])
    tensor = tensor.contiguous()
    index = index.contiguous()

    N, M, K = tensor.shape
    S = index.shape[0]
    device = tensor.device

    offset = torch.arange(0, S, device=device) * M
    index = index + offset.view(S, 1).type_as(index)
    return tensor.view(-1, K)[index.view(-1)].view(S, -1, K)
