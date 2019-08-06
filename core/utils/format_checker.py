# -*- coding: utf-8 -*-
"""
There are many formats in detection project.
just list all following here

* anchor format
    shape: (N, 4) (x1,y1,x2,y2)

* rois format
    shape: (batch_size, N, 5) (img_id, x1,y1,x2,y2)

all boxes can be represented by some format,
(e,g corner_format or xywh format)
* box 2d representation
    shape: (batch_size, N, 4) (x1,y1,x2,y2) or (xywh)

* box 3d representation
    shape: (batch_size, N, 7) (xyzhwlry)

* anchor 3d representation
    shape: (batch_size, N, 7) (xyz,dimx,dimy,dimz)

* image format
    shape: (3, H, W) (image-mean)/var or (image/255)
"""
import numpy as np
import torch
from PIL import Image


def check_anchor_2d_format(anchors):
    if not (anchors.shape[-1] == 4):
        raise TypeError(
            'unknown anchors format due to the num of last dim must be 4 not {}'.
            format(anchors.shape[-1]))
    if len(anchors.shape) == 2:
        pass
    elif len(anchors.shape) == 3:
        # including batch dims
        pass
    else:
        raise TypeError(
            'anchors shape is wrong, length of shape is {}, but expect it is 2 or 3'.
            format(len(anchors.shape)))


def check_if_integer(tensor):
    if isinstance(tensor, np.ndarray):
        tensor_float = tensor.astype(np.float32)
        tensor_int = tensor.astype(np.int).astype(np.float32)
        deltas = tensor_int - tensor_float
        deltas_norm = np.linalg.norm(deltas)
        if deltas_norm > 0:
            return False
        return True
    elif isinstance(tensor, torch.Tensor):
        tensor_int = tensor.int().float()
        tensor_float = tensor.float()
        deltas_norm = tensor.norm(tensor_int - tensor_float)
        if deltas_norm > 0:
            return False
        return True
    else:
        raise TypeError('unknown tensor type {}'.format(type(tensor)))


def check_if_positive(tensor):
    if isinstance(tensor, np.ndarray):
        cond = tensor > 0
        return cond.all()
    elif isinstance(tensor, torch.Tensor):
        # the same as that in np
        cond = tensor > 0
        return cond.all()
    else:
        raise TypeError('unknown tensor type {}'.format(type(tensor)))


def check_if_in_interval(tensor,
                         interval,
                         including_left=True,
                         including_right=True):
    assert interval[0] < interval[1], ValueError('the interval is wrong')
    if including_left:
        cond_left = tensor >= interval[0]
    else:
        cond_left = tensor > interval[0]

    if including_right:
        cond_right = tensor <= interval[1]
    else:
        cond_right = tensor <= interval[1]

    cond = cond_left & cond_right
    return cond.all()


def check_rois_format(rois_batch):
    # last dim checker
    if not (rois_batch.shape[-1] == 5):
        raise TypeError(
            'unknown rois format due to the num of last dim must be 5 not {}'.
            format(rois_batch.shape[-1]))

    # batch_checker
    if not (len(rois_batch.shape) == 3):
        raise TypeError('rois batch should have 3 dims not {}'.format(
            len(rois_batch.shape)))

    # batch dim checker
    # batch dim should be interger
    batch_dim = rois_batch[..., 0]
    assert check_if_integer(
        batch_dim
    ), 'batch dim should be integer due to it refers to image index'


def check_box_xywh(boxes):
    assert boxes.shape[-1] == 4

    # wh should be positive number
    assert check_if_positive(boxes[..., 2:]), 'w and h should be positive'


def check_box_3d(boxes_3d):
    if not (boxes_3d.shape[-1] == 7):
        raise TypeError(
            'unknown boxes_3d format due to the num of last dim must be 7 not {}'.
            format(boxes_3d.shape[-1]))

    # check hwl
    dims = boxes_3d[..., 3:6]
    assert check_if_positive(dims), 'dims of boxes_3d should be positive'

    # check ry
    ry = boxes_3d[..., 6]
    assert check_if_in_interval(ry, [-np.pi, np.pi])


def check_image_format(image, num_channels=3):
    # may be batch format or no batch format
    if not (image.shape[-3] == num_channels):
        raise TypeError('num of image channels should be {} no {}'.format(
            num_channels, image.shape[-3]))


def check_image_div255(image):
    assert check_if_in_interval(image, [
        0, 1
    ]), 'image value is too larger than 1 so it can not be number after divided by 255'


def check_anchor_3d_format(anchors_3d):
    if anchors_3d.shape[-1] != 6:
        raise TypeError(
            'unknown anchor_3d format due to the num of last dim must be 6 not {}'.
            format(anchors_3d.shape[-1]))

    dims = anchors_3d[..., 3:6]
    assert check_if_positive(dims), 'dims of anchors should be positive'


def check_tensor_normalized(tensor):
    assert check_if_in_interval(tensor, [0, 1])


def check_pil_image(image):
    try:
        import accimage
    except ImportError:
        accimage = None

    if accimage is not None:
        assert isinstance(image, (Image.Image, accimage.Image))
    else:
        assert isinstance(image, Image.Image)


def check_tensor_dims(tensor, num_dims):
    assert len(tensor.shape) == num_dims


def check_tensor(tensor):
    assert isinstance(tensor, torch.Tensor)


def check_np(tensor):
    assert isinstance(tensor, np.ndarray)


def check_tensor_shape(tensor, shape):
    """
    Note that None dim is ignored
    """
    check_tensor_dims(tensor, len(shape))
    for dim_ind, dim in enumerate(shape):
        if dim is not None:
            assert tensor.shape[dim_ind] == dim


def check_tensor_type(tensor, tensor_type_name):
    tensor_type_map = {
        'float': torch.float32,
        'long': torch.int64,
        'int': torch.int32,
        'double': torch.double
    }
    tensor_type = tensor_type_map[tensor_type_name]
    assert tensor.dtype is tensor_type
