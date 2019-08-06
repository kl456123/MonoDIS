"""
This module converts data to and from the 'box_3d' format
 [x, y, z, l, w, h, ry]
"""
import numpy as np
import math

import torch
from wavedata.tools.obj_detection import obj_utils


def box_3d_to_object_label(box_3d, obj_type='Car'):
    """Turns a box_3d into an ObjectLabel

    Args:
        box_3d: 3D box in the format [x, y, z, l, w, h, ry]
        obj_type: Optional, the object type

    Returns:
        ObjectLabel with the location, size, and rotation filled out
    """

    obj_label = obj_utils.ObjectLabel()

    obj_label.type = obj_type

    obj_label.t = box_3d.take((0, 1, 2))
    obj_label.l = box_3d[3]
    obj_label.w = box_3d[4]
    obj_label.h = box_3d[5]
    obj_label.ry = box_3d[6]

    return obj_label


def object_label_to_box_3d(obj_label):
    """Turns an ObjectLabel into an box_3d

    Args:
        obj_label: ObjectLabel

    Returns:
        anchor: 3D box in box_3d format [x, y, z, l, w, h, ry]
    """

    box_3d = np.zeros(7)

    box_3d[0:3] = obj_label.t
    box_3d[3] = obj_label.l
    box_3d[4] = obj_label.w
    box_3d[5] = obj_label.h
    box_3d[6] = obj_label.ry

    return box_3d


def box_3d_to_anchor(boxes_3d, ortho_rotate=False):
    """ Converts a box_3d [x, y, z, l, w, h, ry]
    into anchor form [x, y, z, dim_x, dim_y, dim_z]

    Anchors in box_3d format should have an ry of 0 or 90 degrees.
    l and w will be matched to dim_x or dim_z depending on the rotation,
    while h will always correspond to dim_y

    Args:
        boxes_3d: N x 7 ndarray of box_3d
        ortho_rotate: optional, if True the box is rotated to the
            nearest 90 degree angle, or else the box is projected
            onto the x and z axes

    Returns:
        N x 6 ndarray of anchors in 'anchor' form
    """

    boxes_3d = np.asarray(boxes_3d).reshape(-1, 7)

    num_anchors = len(boxes_3d)
    anchors = np.zeros((num_anchors, 6))

    # Set x, y, z
    anchors[:, [0, 1, 2]] = boxes_3d[:, [0, 1, 2]]

    # Dimensions along x, y, z
    box_l = boxes_3d[:, [3]]
    box_w = boxes_3d[:, [4]]
    box_h = boxes_3d[:, [5]]
    box_ry = boxes_3d[:, [6]]

    # Rotate to nearest multiple of 90 degrees
    if ortho_rotate:
        half_pi = np.pi / 2
        box_ry = np.round(box_ry / half_pi) * half_pi

    cos_ry = np.abs(np.cos(box_ry))
    sin_ry = np.abs(np.sin(box_ry))

    # dim_x, dim_y, dim_z
    anchors[:, [3]] = box_l * cos_ry + box_w * sin_ry
    anchors[:, [4]] = box_h
    anchors[:, [5]] = box_w * cos_ry + box_l * sin_ry

    return anchors


def torch_box_3d_to_anchor(boxes_3d):
    """Converts a box_3d tensor to anchor format by ortho rotating it.
    This is similar to 'box_3d_to_anchor' above however it takes
    a tensor as input.

    Args:
        boxes_3d: N x 7 tensor of box_3d in the format [x, y, z, l, w, h, ry]

    Returns:
        anchors: N x 6 tensor of anchors in anchor form ->
            [x, y, z, dim_x, dim_y, dim_z]
    """

    boxes_3d = boxes_3d.view([-1, 7])

    anchors_x = boxes_3d[:, 0]
    anchors_y = boxes_3d[:, 1]
    anchors_z = boxes_3d[:, 2]

    # Dimensions along x, y, z
    box_l = boxes_3d[:, 3]
    box_w = boxes_3d[:, 4]
    box_h = boxes_3d[:, 5]
    box_ry = boxes_3d[:, 6]

    # Ortho rotate
    half_pi = np.pi / 2
    box_ry = torch.round(box_ry / half_pi) * half_pi
    cos_ry = torch.abs(torch.cos(box_ry))
    sin_ry = torch.abs(torch.sin(box_ry))

    anchors_dimx = box_l * cos_ry + box_w * sin_ry
    anchors_dimy = box_h
    anchors_dimz = box_w * cos_ry + box_l * sin_ry

    anchors = torch.stack(
        [
            anchors_x, anchors_y, anchors_z, anchors_dimx, anchors_dimy,
            anchors_dimz
        ],
        dim=1)

    return anchors


def anchors_to_box_3d(anchors, fix_lw=False):
    """Converts an anchor form [x, y, z, dim_x, dim_y, dim_z]
    to 3d box format of [x, y, z, l, w, h, ry]

    Note: In this conversion, if the flag 'fix_lw' is set to true,
    the box_3d 'length' will be the longer of dim_x and dim_z, and 'width'
    will be the shorter dimension. All ry values are set to 0.

    Args:
        anchors: N x 6 ndarray of anchors in 'anchor' form
        fix_lw: A boolean flag to switch width and length in the case
            where width is longer than length.

    Returns:
        N x 7 ndarray of box_3d
    """

    tensor_format = isinstance(anchors, torch.Tensor)

    if tensor_format:
        anchors = torch.tensor(anchors).type_as(anchors)
        num_anchors = anchors.shape[0]
        box_3d = torch.zeros((num_anchors, 7)).type_as(anchors)

        # Set x, y, z
        box_3d[:, 0:3] = anchors[:, 0:3]
        # Set length to dim_x
        box_3d[:, 3] = anchors[:, 3]
        # Set width to dim_z
        box_3d[:, 4] = anchors[:, 5]
        # Set height to dim_y
        box_3d[:, 5] = anchors[:, 4]
        box_3d[:, 6] = 0

        if fix_lw:
            swapped_indices = box_3d[:, 4] > box_3d[:, 3]
            modified_box_3d = box_3d.clone()
            modified_box_3d[swapped_indices, 3] = box_3d[swapped_indices, 4]
            modified_box_3d[swapped_indices, 4] = box_3d[swapped_indices, 3]
            modified_box_3d[swapped_indices, 6] = -math.pi / 2
            return modified_box_3d

    else:

        anchors = np.asarray(anchors)
        box_3d = np.zeros((len(anchors), 7))

        # Set x, y, z
        box_3d[:, 0:3] = anchors[:, 0:3]
        # Set length to dim_x
        box_3d[:, 3] = anchors[:, 3]
        # Set width to dim_z
        box_3d[:, 4] = anchors[:, 5]
        # Set height to dim_y
        box_3d[:, 5] = anchors[:, 4]
        box_3d[:, 6] = 0

        if fix_lw:
            swapped_indices = box_3d[:, 4] > box_3d[:, 3]
            modified_box_3d = np.copy(box_3d)
            modified_box_3d[swapped_indices, 3] = box_3d[swapped_indices, 4]
            modified_box_3d[swapped_indices, 4] = box_3d[swapped_indices, 3]
            modified_box_3d[swapped_indices, 6] = -np.pi / 2
            return modified_box_3d

    return box_3d


def box_3d_to_3d_iou_format(boxes_3d):
    """ Returns a numpy array of 3d box format for iou calculation
    Args:
        boxes_3d: list of 3d boxes
    Returns:
        new_anchor_list: numpy array of 3d box format for iou
    """
    boxes_3d = np.asarray(boxes_3d)

    iou_3d_boxes = np.zeros([len(boxes_3d), 7])
    iou_3d_boxes[:, 4:7] = boxes_3d[:, 0:3]
    iou_3d_boxes[:, 1] = boxes_3d[:, 3]
    iou_3d_boxes[:, 2] = boxes_3d[:, 4]
    iou_3d_boxes[:, 3] = boxes_3d[:, 5]
    iou_3d_boxes[:, 0] = boxes_3d[:, 6]

    return iou_3d_boxes


def torch_box_3d_diagonal_length(boxes_3d):
    """Returns the diagonal length of box_3d

    Args:
        boxes_3d: An tensor of shape (N x 7) of boxes in box_3d format.

    Returns:
        Diagonal of all boxes, a tensor of (N,) shape.
    """

    lengths_sqr = torch.square(boxes_3d[:, 3])
    width_sqr = torch.square(boxes_3d[:, 4])
    height_sqr = torch.square(boxes_3d[:, 5])

    lwh_sqr_sums = lengths_sqr + width_sqr + height_sqr
    diagonals = torch.sqrt(lwh_sqr_sums)

    return diagonals
