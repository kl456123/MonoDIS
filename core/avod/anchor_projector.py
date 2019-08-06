"""
Projects anchors into bird's eye view and image space.
Returns the minimum and maximum box corners, and will only work
for anchors rotated at 0 or 90 degrees
"""

import numpy as np
import torch

from wavedata.tools.core import calib_utils


def project_to_bev(anchors, bev_extents):
    """
    Projects an array of 3D anchors into bird's eye view

    Args:
        anchors: list of anchors in anchor format (N x 6):
            N x [x, y, z, dim_x, dim_y, dim_z],
            can be a numpy array or tensor
        bev_extents: xz extents of the 3d area
            [[min_x, max_x], [min_z, max_z]]

    Returns:
          box_corners_norm: corners as a percentage of the map size, in the
            format N x [x1, y1, x2, y2]. Origin is the top left corner
    """

    tensor_format = isinstance(anchors, torch.Tensor)
    if not tensor_format:
        anchors = np.asarray(anchors)

    x = anchors[:, 0]
    z = anchors[:, 2]
    half_dim_x = anchors[:, 3] / 2.0
    half_dim_z = anchors[:, 5] / 2.0

    # Calculate extent ranges
    bev_x_extents_min = bev_extents[0][0]
    bev_z_extents_min = bev_extents[1][0]
    bev_x_extents_max = bev_extents[0][1]
    bev_z_extents_max = bev_extents[1][1]
    bev_x_extents_range = bev_x_extents_max - bev_x_extents_min
    bev_z_extents_range = bev_z_extents_max - bev_z_extents_min

    # 2D corners (top left, bottom right)
    x1 = x - half_dim_x
    x2 = x + half_dim_x
    # Flip z co-ordinates (origin changes from bottom left to top left)
    z1 = bev_z_extents_max - (z + half_dim_z)
    z2 = bev_z_extents_max - (z - half_dim_z)

    if tensor_format:
        bev_box_corners = torch.stack([x1, z1, x2, z2], dim=1)
    else:
        bev_box_corners = np.stack([x1, z1, x2, z2], axis=1)

    # Convert from original xz into bev xz, origin moves to top left
    bev_extents_min_tiled = [
        bev_x_extents_min, bev_z_extents_min, bev_x_extents_min,
        bev_z_extents_min
    ]
    if tensor_format:
        bev_box_corners = bev_box_corners - torch.tensor(
            bev_extents_min_tiled).type_as(bev_box_corners)
    else:
        bev_box_corners = bev_box_corners - bev_extents_min_tiled

    # Calculate normalized box corners for ROI pooling
    extents_tiled = [
        bev_x_extents_range, bev_z_extents_range, bev_x_extents_range,
        bev_z_extents_range
    ]
    if tensor_format:
        bev_box_corners_norm = bev_box_corners / torch.tensor(
            extents_tiled).type_as(bev_box_corners)
    else:
        bev_box_corners_norm = bev_box_corners / extents_tiled

    return bev_box_corners, bev_box_corners_norm


def project_to_image_space(anchors, stereo_calib_p2, image_shape):
    """
    Projects 3D anchors into image space

    Args:
        anchors: list of anchors in anchor format N x [x, y, z,
            dim_x, dim_y, dim_z]
        stereo_calib_p2: stereo camera calibration p2 matrix
        image_shape: dimensions of the image [h, w]

    Returns:
        box_corners: corners in image space - N x [x1, y1, x2, y2]
        box_corners_norm: corners as a percentage of the image size -
            N x [x1, y1, x2, y2]
    """
    if anchors.shape[1] != 6:
        raise ValueError("Invalid shape for anchors {}, should be "
                         "(N, 6)".format(anchors.shape[1]))

    # Figure out box mins and maxes
    x = (anchors[:, 0])
    y = (anchors[:, 1])
    z = (anchors[:, 2])

    dim_x = (anchors[:, 3])
    dim_y = (anchors[:, 4])
    dim_z = (anchors[:, 5])

    dim_x_half = dim_x / 2.
    dim_z_half = dim_z / 2.

    # Calculate 3D BB corners
    x_corners = np.array([
        x + dim_x_half, x + dim_x_half, x - dim_x_half, x - dim_x_half,
        x + dim_x_half, x + dim_x_half, x - dim_x_half, x - dim_x_half
    ]).T.reshape(1, -1)

    y_corners = np.array(
        [y, y, y, y, y - dim_y, y - dim_y, y - dim_y,
         y - dim_y]).T.reshape(1, -1)

    z_corners = np.array([
        z + dim_z_half, z - dim_z_half, z - dim_z_half, z + dim_z_half,
        z + dim_z_half, z - dim_z_half, z - dim_z_half, z + dim_z_half
    ]).T.reshape(1, -1)

    anchor_corners = np.vstack([x_corners, y_corners, z_corners])

    # Apply the 2D image plane transformation
    pts_2d = calib_utils.project_to_image(anchor_corners, stereo_calib_p2)

    # Get the min and maxes of image coordinates
    i_axis_min_points = np.amin(pts_2d[0, :].reshape(-1, 8), axis=1)
    j_axis_min_points = np.amin(pts_2d[1, :].reshape(-1, 8), axis=1)

    i_axis_max_points = np.amax(pts_2d[0, :].reshape(-1, 8), axis=1)
    j_axis_max_points = np.amax(pts_2d[1, :].reshape(-1, 8), axis=1)

    box_corners = np.vstack([
        i_axis_min_points, j_axis_min_points, i_axis_max_points,
        j_axis_max_points
    ]).T

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shape_tiled = [
        image_shape_w, image_shape_h, image_shape_w, image_shape_h
    ]

    box_corners_norm = box_corners / image_shape_tiled

    return np.array(box_corners, dtype=np.float32), \
           np.array(box_corners_norm, dtype=np.float32)


def torch_project_to_image_space(anchors, stereo_calib_p2, image_shape):
    """
    Projects 3D anchors into image space

    Args:
        anchors: list of anchors in anchor format N x [x, y, z,
            dim_x, dim_y, dim_z]
        stereo_calib_p2: stereo camera calibration p2 matrix
        image_shape: dimensions of the image [h, w]

    Returns:
        box_corners: corners in image space - N x [x1, y1, x2, y2]
        box_corners_norm: corners as a percentage of the image size -
            N x [x1, y1, x2, y2]
    """
    if anchors.shape[1] != 6:
        raise ValueError("Invalid shape for anchors {}, should be "
                         "(N, 6)".format(anchors.shape[1]))

    # Figure out box mins and maxes
    x = (anchors[:, 0])
    y = (anchors[:, 1])
    z = (anchors[:, 2])

    dim_x = (anchors[:, 3])
    dim_y = (anchors[:, 4])
    dim_z = (anchors[:, 5])

    dim_x_half = dim_x / 2.
    dim_z_half = dim_z / 2.

    # Calculate 3D BB corners
    x_corners = torch.stack(
        [
            x + dim_x_half, x + dim_x_half, x - dim_x_half, x - dim_x_half,
            x + dim_x_half, x + dim_x_half, x - dim_x_half, x - dim_x_half
        ],
        dim=-1)

    y_corners = torch.stack(
        [y, y, y, y, y - dim_y, y - dim_y, y - dim_y, y - dim_y], dim=-1)

    z_corners = torch.stack(
        [
            z + dim_z_half, z - dim_z_half, z - dim_z_half, z + dim_z_half,
            z + dim_z_half, z - dim_z_half, z - dim_z_half, z + dim_z_half
        ],
        dim=-1)

    anchor_corners = torch.stack([x_corners, y_corners, z_corners], dim=0)

    # Apply the 2D image plane transformation
    pts_2d = project_to_image_tensor(
        anchor_corners.view(3, -1), stereo_calib_p2)

    pts_2d = pts_2d.view(2, -1, 8)

    # Get the min and maxes of image coordinates
    i_axis_min_points, _ = torch.min(pts_2d[0, :, :], dim=1)
    j_axis_min_points, _ = torch.min(pts_2d[1, :, :], dim=1)

    i_axis_max_points, _ = torch.max(pts_2d[0, :, :], dim=1)
    j_axis_max_points, _ = torch.max(pts_2d[1, :, :], dim=1)

    box_corners = torch.stack(
        [
            i_axis_min_points, j_axis_min_points, i_axis_max_points,
            j_axis_max_points
        ],
        dim=-1)

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shape_tiled = [
        image_shape_w, image_shape_h, image_shape_w, image_shape_h
    ]

    box_corners_norm = box_corners / torch.tensor(image_shape_tiled).type_as(
        box_corners).view(-1, 4)

    return box_corners, box_corners_norm


def project_to_image_tensor(points_3d, cam_p2_matrix):
    """Projects 3D points to 2D points in image space.

    Args:
        points_3d: a list of float32 tensor of shape [3, None]
        cam_p2_matrix: a float32 tensor of shape [3, 4] representing
            the camera matrix.

    Returns:
        points_2d: a list of float32 tensor of shape [2, None]
            This is the projected 3D points into 2D .i.e. corresponding
            3D points in image coordinates.
    """
    ones_column = torch.ones_like(points_3d[-1:, :]).type_as(points_3d)

    # Add extra column of ones
    points_3d_concat = torch.cat([points_3d, ones_column], dim=0)

    # Multiply camera matrix by the 3D points
    points_2d = torch.matmul(cam_p2_matrix, points_3d_concat)

    # 'Tensor' object does not support item assignment
    # so instead get the result of each division and stack
    # the results
    points_2d_c1 = points_2d[0, :] / points_2d[2, :]
    points_2d_c2 = points_2d[1, :] / points_2d[2, :]
    stacked_points_2d = torch.stack([points_2d_c1, points_2d_c2], dim=0)

    return stacked_points_2d
