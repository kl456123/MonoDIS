import numpy as np


def get_point_filter(point_cloud, extents, ground_plane=None, offset_dist=2.0):
    """
    Creates a point filter using the 3D extents and ground plane

    :param point_cloud: Point cloud in the form [[x,...],[y,...],[z,...]]
    :param extents: 3D area in the form
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    :param ground_plane: Optional, coefficients of the ground plane
        (a, b, c, d)
    :param offset_dist: If ground_plane is provided, removes points above
        this offset from the ground_plane
    :return: A binary mask for points within the extents and offset plane
    """

    point_cloud = np.asarray(point_cloud)

    # Filter points within certain xyz range
    x_extents = extents[0]
    y_extents = extents[1]
    z_extents = extents[2]

    extents_filter = (point_cloud[0] > x_extents[0]) & \
                     (point_cloud[0] < x_extents[1]) & \
                     (point_cloud[1] > y_extents[0]) & \
                     (point_cloud[1] < y_extents[1]) & \
                     (point_cloud[2] > z_extents[0]) & \
                     (point_cloud[2] < z_extents[1])

    if ground_plane is not None:
        ground_plane = np.array(ground_plane)

        # Calculate filter using ground plane
        ones_col = np.ones(point_cloud.shape[1])
        padded_points = np.vstack([point_cloud, ones_col])

        offset_plane = ground_plane + [0, 0, 0, -offset_dist]

        # Create plane filter
        dot_prod = np.dot(offset_plane, padded_points)
        plane_filter = dot_prod < 0

        # Combine the two filters
        point_filter = np.logical_and(extents_filter, plane_filter)
    else:
        # Only use the extents for filtering
        point_filter = extents_filter

    return point_filter


def create_slice_filter(point_cloud, area_extents,
                        ground_plane, ground_offset_dist, offset_dist):
    """ Creates a slice filter to take a slice of the point cloud between
        ground_offset_dist and offset_dist above the ground plane

    Args:
        point_cloud: Point cloud in the shape (3, N)
        area_extents: 3D area extents
        ground_plane: ground plane coefficients
        offset_dist: max distance above the ground
        ground_offset_dist: min distance above the ground plane

    Returns:
        A boolean mask if shape (N,) where
            True indicates the point should be kept
            False indicates the point should be removed
    """

    # Filter points within certain xyz range and offset from ground plane
    offset_filter = get_point_filter(point_cloud, area_extents,
                                     ground_plane, offset_dist)

    # Filter points within 0.2m of the road plane
    road_filter = get_point_filter(point_cloud, area_extents,
                                   ground_plane,
                                   ground_offset_dist)

    slice_filter = np.logical_xor(offset_filter, road_filter)
    return slice_filter
