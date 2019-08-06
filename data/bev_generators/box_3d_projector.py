import numpy as np

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

    # Stack into (N x 4)
    bev_box_corners = np.stack([x1, z1, x2, z2], axis=1)

    # Convert from original xz into bev xz, origin moves to top left
    bev_extents_min_tiled = [bev_x_extents_min, bev_z_extents_min,
                             bev_x_extents_min, bev_z_extents_min]
    bev_box_corners = bev_box_corners - bev_extents_min_tiled

    # Calculate normalized box corners for ROI pooling
    extents_tiled = [bev_x_extents_range, bev_z_extents_range,
                     bev_x_extents_range, bev_z_extents_range]
    bev_box_corners_norm = bev_box_corners / extents_tiled

    return bev_box_corners, bev_box_corners_norm


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
