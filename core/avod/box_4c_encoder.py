import numpy as np
import torch
from wavedata.tools.core import geometry_utils

from core.avod import box_3d_encoder
"""Box4c Encoder
Converts boxes between the box_3d and box_4c formats.
- box_4c format: [x1, x2, x3, x4, z1, z2, z3, z4, h1, h2]
- corners are in the xz plane, numbered clockwise starting at the top right
- h1 is the height above the ground plane to the bottom of the box
- h2 is the height above the ground plane to the top of the box
"""


def np_box_3d_to_box_4c(box_3d, ground_plane):
    """Converts a single box_3d to box_4c

    Args:
        box_3d: box_3d (6,)
        ground_plane: ground plane coefficients (4,)

    Returns:
        box_4c (10,)
    """

    anchor = box_3d_encoder.box_3d_to_anchor(box_3d, ortho_rotate=True)[0]

    centroid_x = anchor[0]
    centroid_y = anchor[1]
    centroid_z = anchor[2]
    dim_x = anchor[3]
    dim_y = anchor[4]
    dim_z = anchor[5]

    # Create temporary box at (0, 0) for rotation
    half_dim_x = dim_x / 2
    half_dim_z = dim_z / 2

    # Box corners
    x_corners = np.asarray([half_dim_x, half_dim_x, -half_dim_x, -half_dim_x])

    z_corners = np.array([half_dim_z, -half_dim_z, -half_dim_z, half_dim_z])

    ry = box_3d[6]

    # Find nearest 90 degree
    half_pi = np.pi / 2
    ortho_ry = np.round(ry / half_pi) * half_pi

    # Find rotation to make the box ortho aligned
    ry_diff = ry - ortho_ry

    # Create transformation matrix, including rotation and translation
    tr_mat = np.array([[np.cos(ry_diff), np.sin(ry_diff), centroid_x],
                       [-np.sin(ry_diff), np.cos(ry_diff), centroid_z],
                       [0, 0, 1]])

    # Create a ones row
    ones_row = np.ones(x_corners.shape)

    # Append the column of ones to be able to multiply
    points_stacked = np.vstack([x_corners, z_corners, ones_row])
    corners = np.matmul(tr_mat, points_stacked)

    # Discard the last row (ones)
    corners = corners[0:2]

    # Calculate height off ground plane
    ground_y = geometry_utils.calculate_plane_point(
        ground_plane, [centroid_x, None, centroid_z])[1]
    h1 = ground_y - centroid_y
    h2 = h1 + dim_y

    # Stack into (10,) ndarray
    box_4c = np.hstack([corners.flatten(), h1, h2])
    return box_4c


def torch_box_3d_to_box_4c(boxes_3d, ground_plane):
    """Vectorized conversion of box_3d to box_4c tensors

    Args:
        boxes_3d: Tensor of boxes_3d (N, 7)
        ground_plane: Tensor ground plane coefficients (4,)

    Returns:
        Tensor of boxes_4c (N, 10)
    """

    anchors = box_3d_encoder.torch_box_3d_to_anchor(boxes_3d)

    centroid_x = anchors[:, 0]
    centroid_y = anchors[:, 1]
    centroid_z = anchors[:, 2]
    dim_x = anchors[:, 3]
    dim_y = anchors[:, 4]
    dim_z = anchors[:, 5]

    # Create temporary box at (0, 0) for rotation
    half_dim_x = dim_x / 2
    half_dim_z = dim_z / 2

    # Box corners
    x_corners = torch.stack(
        [half_dim_x, half_dim_x, -half_dim_x, -half_dim_x], dim=1)

    z_corners = torch.stack(
        [half_dim_z, -half_dim_z, -half_dim_z, half_dim_z], dim=1)

    # Rotations from boxes_3d
    all_rys = boxes_3d[:, 6]

    # Find nearest 90 degree
    half_pi = np.pi / 2
    ortho_rys = torch.round(all_rys / half_pi) * half_pi
    ortho_rys = ortho_rys.type_as(all_rys)

    # Get rys and 0/1 padding
    ry_diffs = all_rys - ortho_rys
    zeros = torch.zeros_like(ry_diffs).type_as(ry_diffs)
    ones = torch.ones_like(ry_diffs).type_as(ry_diffs)

    # Create transformation matrix, including rotation and translation
    tr_mat = torch.stack(
        [
            torch.stack(
                [torch.cos(ry_diffs), torch.sin(ry_diffs), centroid_x],
                dim=1), torch.stack(
                    [-torch.sin(ry_diffs), torch.cos(ry_diffs), centroid_z],
                    dim=1), torch.stack(
                        [zeros, zeros, ones], dim=1)
        ],
        dim=2)

    # Create a ones row
    ones_row = torch.ones_like(x_corners)

    # Append the column of ones to be able to multiply
    points_stacked = torch.stack([x_corners, z_corners, ones_row], dim=1)
    corners = torch.matmul(tr_mat.permute(0, 2, 1), points_stacked)

    # Discard the last row (ones)
    corners = corners[:, 0:2]
    flat_corners = torch.reshape(corners, [-1, 8])

    # Get ground plane coefficients
    a = ground_plane[0]
    b = ground_plane[1]
    c = ground_plane[2]
    d = ground_plane[3]

    # Calculate heights off ground plane
    ground_y = -(a * centroid_x + c * centroid_z + d) / b
    h1 = ground_y - centroid_y
    h2 = h1 + dim_y

    batched_h1 = h1.view([-1, 1])
    batched_h2 = h2.view([-1, 1])

    # Stack into (?, 10)
    box_4c = torch.cat([flat_corners, batched_h1, batched_h2], dim=1)
    return box_4c


def torch_box_4c_to_box_3d(box_4c, ground_plane):
    """Converts a single box_4c to box_3d. The longest midpoint-midpoint
    length is used to calculate orientation. Points are projected onto the
    orientation vector and the orthogonal vector to get the bounding box_3d.
    The centroid is calculated by adding a vector of half the projected length
    along the midpoint-midpoint vector, and a vector of the width
    differences along the normal.

    Args:
        box_4c: box_4c to convert (10,)
        ground_plane: ground plane coefficients (4,)

    Returns:
        box_3d (7,)
    """

    # Extract corners
    corners = box_4c[:, 0:8].reshape(-1, 2, 4)

    p1 = corners[:, :, 0]
    p2 = corners[:, :, 1]
    p3 = corners[:, :, 2]
    p4 = corners[:, :, 3]

    # Check for longest axis
    midpoint_12 = (p1 + p2) / 2.0
    midpoint_23 = (p2 + p3) / 2.0
    midpoint_34 = (p3 + p4) / 2.0
    midpoint_14 = (p1 + p4) / 2.0

    vec_34_12 = midpoint_12 - midpoint_34
    vec_34_12_mag = torch.norm(vec_34_12, dim=-1)

    vec_23_14 = midpoint_14 - midpoint_23
    vec_23_14_mag = torch.norm(vec_23_14, dim=-1)

    # Check which midpoint -> midpoint vector is longer
    # midpoint , vec, vec_norm
    cond = vec_34_12_mag > vec_23_14_mag
    midpoint = torch.zeros_like(midpoint_34).type_as(midpoint_34)
    midpoint[cond] = midpoint_34[cond]
    midpoint[~cond] = midpoint_23[~cond]

    vec = torch.zeros_like(vec_34_12).type_as(vec_34_12)
    vec[cond] = vec_34_12[cond]
    vec[~cond] = vec_23_14[~cond]

    vec_mag = torch.zeros_like(vec_34_12_mag).type_as(vec_34_12_mag)
    vec_mag[cond] = vec_34_12_mag[cond]
    vec_mag[~cond] = vec_23_14_mag[~cond]

    vec_norm = vec / vec_mag.unsqueeze(-1)

    vec_mid_p1 = p1 - midpoint
    vec_mid_p2 = p2 - midpoint
    vec_mid_p3 = p3 - midpoint
    vec_mid_p4 = p4 - midpoint

    l1 = (vec_mid_p1 * vec_norm).sum(dim=1)
    l2 = (vec_mid_p2 * vec_norm).sum(dim=1)
    l3 = (vec_mid_p3 * vec_norm).sum(dim=1)
    l4 = (vec_mid_p4 * vec_norm).sum(dim=1)
    all_lengths = torch.stack([l1, l2, l3, l4], dim=-1)

    min_l, _ = torch.min(all_lengths, dim=-1, keepdim=True)
    max_l, _ = torch.max(all_lengths, dim=-1, keepdim=True)
    length_out = max_l - min_l
    length_out = length_out[:, 0]

    ortho_norm = torch.stack([-vec_norm[:, 1], vec_norm[:, 0]], dim=-1)
    w1 = (vec_mid_p1 * ortho_norm).sum(dim=1)
    w2 = (vec_mid_p2 * ortho_norm).sum(dim=1)
    w3 = (vec_mid_p3 * ortho_norm).sum(dim=1)
    w4 = (vec_mid_p4 * ortho_norm).sum(dim=1)
    all_widths = torch.stack([w1, w2, w3, w4], dim=-1)

    min_w, _ = torch.min(all_widths, dim=-1, keepdim=True)
    max_w, _ = torch.max(all_widths, dim=-1, keepdim=True)
    w_diff = max_w + min_w
    width_out = max_w - min_w
    width_out = width_out[:, 0]

    ry_out = -torch.atan2(vec[:, 1], vec[:, 0])

    # New centroid
    centroid = midpoint + vec_norm * (min_l + max_l) / 2.0 + \
        ortho_norm * w_diff

    # Find new centroid y
    a = ground_plane[0]
    b = ground_plane[1]
    c = ground_plane[2]
    d = ground_plane[3]

    h1 = box_4c[:, 8]
    h2 = box_4c[:, 9]

    centroid_x = centroid[:, 0]
    centroid_z = centroid[:, 1]

    ground_y = -(a * centroid_x + c * centroid_z + d) / b

    # h1 and h2 are along the -y axis
    centroid_y = ground_y - h1
    height_out = h2 - h1

    box_3d_out = torch.stack(
        [
            centroid_x, centroid_y, centroid_z, length_out, width_out,
            height_out, ry_out
        ],
        dim=-1)

    return box_3d_out


def calculate_box_3d_info(vec_dir, vec_dir_mag, p1, p2, p3, p4, midpoint):
    """Calculates the box_3d centroid xz, l, w, and ry from the 4 points of
    a box_4c. To calculate length and width, points are projected onto the
    direction vector, and its normal. The centroid is calculated by adding
    vectors of half the length, and the width difference along the normal to
    the starting midpoint. ry is calculated with atan2 of the direction vector.

    Args:
        vec_dir: vector of longest box_4c midpoint to midpoint
        vec_dir_mag: magnitude of the direction vector
        p1: point 1
        p2: point 2
        p3: point 3
        p4: point 4
        midpoint: starting midpoint

    Returns:
        box_3d info (centroid, length_out, width_out, ry_out)
    """
    # import ipdb
    # ipdb.set_trace()
    vec_dir_norm = vec_dir / vec_dir_mag.view(-1, 1)

    vec_mid_p1 = p1 - midpoint
    vec_mid_p2 = p2 - midpoint
    vec_mid_p3 = p3 - midpoint
    vec_mid_p4 = p4 - midpoint

    l1 = (vec_mid_p1 * vec_dir_norm).sum(dim=1)
    l2 = (vec_mid_p2 * vec_dir_norm).sum(dim=1)
    l3 = (vec_mid_p3 * vec_dir_norm).sum(dim=1)
    l4 = (vec_mid_p4 * vec_dir_norm).sum(dim=1)

    all_lengths = torch.stack([l1, l2, l3, l4], dim=1)

    min_l = all_lengths.sum(dim=1, keepdim=True)
    max_l = all_lengths.sum(dim=1, keepdim=True)
    length_out = max_l - min_l

    vec_dir_ortho_norm = torch.stack(
        [-vec_dir_norm[:, 1], vec_dir_norm[:, 0]], dim=1)
    w1 = torch.sum(vec_mid_p1 * vec_dir_ortho_norm, dim=1)
    w2 = torch.sum(vec_mid_p2 * vec_dir_ortho_norm, dim=1)
    w3 = torch.sum(vec_mid_p3 * vec_dir_ortho_norm, dim=1)
    w4 = torch.sum(vec_mid_p4 * vec_dir_ortho_norm, dim=1)
    all_widths = torch.stack([w1, w2, w3, w4], dim=1)

    min_w, _ = torch.min(all_widths, dim=1)
    max_w, _ = torch.max(all_widths, dim=1)
    w_diff = (max_w + min_w).view(-1, 1)
    width_out = (max_w - min_w).view(-1, 1)

    ry_out = (-torch.atan2(vec_dir[:, 1], vec_dir[:, 0])).view(-1, 1)

    # New centroid
    centroid = midpoint +\
        vec_dir_norm * (min_l + max_l) / 2.0 + \
        vec_dir_ortho_norm * w_diff

    return centroid, length_out, width_out, ry_out


def _torch_box_4c_to_box_3d(boxes_4c, ground_plane):
    """Vectorized box_4c to box_3d conversion

    Args:
        boxes_4c: Tensor of boxes_4c (N, 10)
        ground_plane: Tensor of ground plane coefficients (4,)

    Returns:
        Tensor of boxes_3d (N, 7)
    """
    # Extract corners
    corners = boxes_4c[:, 0:8].view([-1, 2, 4])

    p1 = corners[:, :, 0]
    p2 = corners[:, :, 1]
    p3 = corners[:, :, 2]
    p4 = corners[:, :, 3]

    # Get line midpoints
    midpoint_12 = (p1 + p2) / 2.0
    midpoint_23 = (p2 + p3) / 2.0
    midpoint_34 = (p3 + p4) / 2.0
    midpoint_14 = (p1 + p4) / 2.0

    # Check which direction is longer
    vec_34_12 = midpoint_12 - midpoint_34
    vec_34_12_mag = torch.norm(vec_34_12, dim=1)

    vec_23_14 = midpoint_14 - midpoint_23
    vec_23_14_mag = torch.norm(vec_23_14, dim=1)

    # Calculate both possibilities (vec_34_12_mag or vec_23_14_mag),
    # then mask out the values from the shorter direction

    # vec_34_12_mag longer
    vec_34_12_centroid, vec_34_12_length, vec_34_12_width, vec_34_12_ry = \
        calculate_box_3d_info(vec_34_12, vec_34_12_mag,
                              p1, p2, p3, p4, midpoint=midpoint_34)

    # vec_23_14_mag longer
    vec_23_14_centroid, vec_23_14_length, vec_23_14_width, vec_23_14_ry = \
        calculate_box_3d_info(vec_23_14, vec_23_14_mag,
                              p1, p2, p3, p4, midpoint=midpoint_23)

    vec_34_12_mask = vec_34_12_mag > vec_23_14_mag
    vec_23_14_mask = ~vec_34_12_mask

    vec_34_12_float_mask = vec_34_12_mask.float().view(-1, 1)
    vec_23_14_float_mask = vec_23_14_mask.float().view(-1, 1)

    centroid_xz = vec_34_12_centroid * vec_34_12_float_mask + \
        vec_23_14_centroid * vec_23_14_float_mask
    length_out = vec_34_12_length * vec_34_12_float_mask + \
        vec_23_14_length * vec_23_14_float_mask
    width_out = vec_34_12_width * vec_34_12_float_mask + \
        vec_23_14_width * vec_23_14_float_mask
    ry_out = vec_34_12_ry * vec_34_12_float_mask + \
        vec_23_14_ry * vec_23_14_float_mask

    # Find new centroid y
    a = ground_plane[0]
    b = ground_plane[1]
    c = ground_plane[2]
    d = ground_plane[3]

    h1 = boxes_4c[:, 8]
    h2 = boxes_4c[:, 9]

    centroid_x = centroid_xz[:, 0]
    centroid_z = centroid_xz[:, 1]

    # Squeeze to single dimension for stacking
    length_out = torch.squeeze(length_out)
    width_out = torch.squeeze(width_out)
    ry_out = torch.squeeze(ry_out)

    ground_y = -(a * centroid_x + c * centroid_z + d) / b

    # h1 and h2 are along the -y axis
    centroid_y = ground_y - h1
    height_out = h2 - h1

    box_3d_out = torch.stack(
        [
            centroid_x, centroid_y, centroid_z, length_out, width_out,
            height_out, ry_out
        ],
        dim=1)

    return box_3d_out


def torch_box_4c_to_offsets(boxes_4c, box_4c_gt):
    """Calculates box_4c offsets to regress to ground truth

    Args:
        boxes_4c: boxes_4c to calculate offset for (N, 10)
        box_4c_gt: box_4c ground truth to regress to (10,)

    Returns:
        box_4c offsets (N, 10)
    """
    return box_4c_gt - boxes_4c


def torch_offsets_to_box_4c(boxes_4c, offsets):
    """Applies box_4c offsets to boxes_4c

    Args:
        boxes_4c: boxes_4c to apply offsets to
        offsets: box_4c offsets to apply

    Returns:
        regressed boxes_4c
    """
    return boxes_4c + offsets
