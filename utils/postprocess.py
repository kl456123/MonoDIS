import numpy as np
from numpy.linalg import norm
from utils.kitti_util import compute_global_angle, roty
from data.kitti_helper import get_depth_coords
from utils.kitti_util import compute_ray_angle


def mono_3d_postprocess(dets_3d, p2):
    """
    Post process for generating 3d object
    Args:
    dets_3d: list of detection result for 3d reconstruction
    """
    # in single image
    # shape(N,7), (fblx,fbly,fbrx,fbry,rblx,rbly,ftly)
    rcnn_3ds = dets_3d
    box_3d = []
    for i in range(rcnn_3ds.shape[0]):
        points_2d = rcnn_3ds[i, :-1].reshape((3, 2))
        y_2d = rcnn_3ds[i, -1]
        points_3d = point_2dto3d(points_2d, p2)
        # points_3d = points_3d.reshape((3, 3))

        points_3d = rect(points_3d)

        ftl = np.stack([rcnn_3ds[i, 0], y_2d], axis=-1)
        ftl_3d = get_top_side(points_3d, ftl, p2)

        # shape(N,4,3) (fbl,fbr,rbl,ftl)
        points_3d = np.concatenate([points_3d, ftl_3d.reshape(1, 3)], axis=0)

        # 3d bbox reconstruction has already done, get rlhwxyz
        l_v = points_3d[2] - points_3d[0]
        h_v = points_3d[3] - points_3d[0]
        w_v = points_3d[1] - points_3d[0]

        l = norm(l_v, axis=-1, keepdims=True)
        h = norm(h_v, axis=-1, keepdims=True)
        w = norm(w_v, axis=-1, keepdims=True)

        center = (l_v + h_v + w_v) / 2 + points_3d[0]

        # x_corners = np.array([l / 2, l / 2, -l / 2, l / 2])
        # y_corners = np.zeros(4, h.shape(0))
        # y_corners[3, :] = -h
        # # y_corners = np.array([0, 0, 0, -h])
        # z_corners = np.array([w / 2, -w / 2, w / 2, w / 2])

        # import ipdb
        # ipdb.set_trace()
        # box = np.vstack((x_corners, y_corners, z_corners))
        # box_rotated = points_3d - center
        # R = np.dot(box_rotated[:, 1:], np.linalg.inv(box[:, 1:]))
        # ry = np.arccos(R[0])
        # just for simplification
        ry = np.arccos(l_v[0] / l)
        if l_v[2] / l < 0:
            ry = -ry

        # bottom center is the origin of object coords frame
        center -= h_v / 2
        # hwlxyzry
        box_3d.append(np.concatenate([h, w, l, center, ry], axis=-1))
    return np.stack(box_3d, axis=0)


def get_top_side(points_3d, ftl, p2):
    """
    Args:
    points_3d: shape(N,3,3)
    h_2d: shape(N,)
    ftl: shape(N,2) point in 2d
    """
    # homo
    ftl_homo = np.concatenate([ftl, np.ones_like(ftl[:1])], axis=-1)
    normal = points_3d[0] - points_3d[2]
    d = -np.dot(points_3d[0], normal.T)

    p2_inv, c = decompose(p2)

    # get direction
    direction_3d = np.dot(p2_inv, ftl_homo.T)

    # c = get_camera_center(p2)
    coeff = -(np.dot(normal, c) + d) / np.dot(normal, direction_3d)
    ftl_3d = c + coeff * direction_3d

    # transform back from homo
    # ftl_3d /= ftl_3d[:, -1]
    return ftl_3d


def null(A, eps=1e-12):
    import scipy
    from scipy import linalg, matrix
    u, s, vh = scipy.linalg.svd(A)
    padding = max(0, np.shape(A)[1] - np.shape(s)[0])
    null_mask = np.concatenate(
        ((s <= eps), np.ones(
            (padding, ), dtype=bool)), axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)


def rect(points_3d):
    """
    Args:
    points_3d: shape(N,3,3)
    points_3d: shape(N,4,3)
    """
    center = (points_3d[2] + points_3d[1]) / 2
    d1 = points_3d[0] - center
    d2 = points_3d[1] - center
    deltas = (norm(d1) - norm(d2)) / 2
    fbl = center + d1 * (1 - deltas / norm(d1))
    fbr = center + d2 * (1 + deltas / norm(d2))
    rbl = center - d2 * (1 + deltas / norm(d2))
    points_3d_rect = np.zeros_like(points_3d)
    points_3d_rect[0] = fbl
    points_3d_rect[1] = fbr
    points_3d_rect[2] = rbl
    return points_3d_rect


def get_ground_plane():
    return np.asarray([0, 1, 0, -2.3])


def get_camera_center(p2):
    return null(p2).reshape((-1, 1))


def decompose(p2):
    M = p2[:, :-1]
    t = p2[:, -1]
    M_inv = np.linalg.inv(M)
    C = -np.dot(M_inv, t)
    return M_inv, C


def point_2dto3d(points_2d, p2):
    """
    Args:
    points_2d: shape(N,2), points in 2d image
    points_3d: shape(N,3), points in camera coords frame
    """
    # p2_inv = np.linalg.pinv(p2)
    p2_inv, c = decompose(p2)

    points_2d_homo = np.concatenate(
        [points_2d, np.ones_like(points_2d[:, :1])], axis=-1)

    # get direction
    direction_3d = np.dot(p2_inv, points_2d_homo.T)

    ground = get_ground_plane()

    # coeff = -np.dot(ground.T, direction_3d) / np.dot(ground.T, c)
    coeff = -(np.dot(ground[:-1].T, c) + ground[-1]) / (np.dot(ground[:-1].T,
                                                               direction_3d))
    points_3d = c + (coeff * direction_3d).T

    points_3d = points_3d

    # transform back from homo
    # points_3d /= points_3d[:, -1:]
    return points_3d


def mono_3d_postprocess_dims(dets_3d, dets_2d, p2):
    """
    X = inv(K) * x * depth
    Args:
        dets_3d: shape(N,6) (hwl_2d,hwl_3d)
        dets_2d: shape(N,5) (xyxyc)
        p2: shape(4,3)
    """
    # decompose p2
    K = p2[:3, :3]
    T = p2[:, -1]
    focal_length = K[0, 0]
    center_2d_x = (dets_2d[:, 0] + dets_2d[:, 2]) / 2
    center_2d_y = (dets_2d[:, 1] + dets_2d[:, 3]) / 2
    h = (dets_2d[:, 3] - dets_2d[:, 1] + 1) / 2
    center_2d_ones = np.ones_like(center_2d_y)
    center_2d_homo = np.stack(
        [center_2d_x, center_2d_y, center_2d_ones], axis=-1)

    ground = get_ground_plane()

    # similarity
    depth_4 = dets_3d[:, 3] / dets_3d[:, 0] * focal_length
    # the same almostly
    depth_center = depth_4
    #  center_3d = np.dot(np.linalg.inv(K),
    #  center_2d_homo.T).T * depth_center.unsqueeze(-1)

    center_3d = np.dot(
        np.linalg.inv(K),
        (depth_center[..., np.newaxis] * center_2d_homo - T).T).T

    # ry
    ry = np.zeros_like(center_3d[:, -1:])
    rcnn_3d = np.concatenate([dets_3d[:, 3:6], center_3d, ry], axis=-1)
    return rcnn_3d


def generate_side_points(dets_2d, orient, normalize=True):
    """
    Generate side points to calculate orientation
    Args:
        dets_2d: shape(N,4) detected box
        orient: shape(N,3) cls_orient and reg_orient
    Returns:
        side_points: shape(N,4)
    """
    cls_orient_argmax = orient[:, 0].astype(np.int32)
    abnormal_cond = cls_orient_argmax == 2
    cls_orient_argmax[abnormal_cond] = 0

    reg_orient = orient[:, 1:3]
    if normalize:
        # import ipdb
        # ipdb.set_trace()
        wh = dets_2d[:, 2:4] - dets_2d[:, :2] + 1
        reg_orient = reg_orient * wh

    abnormal_case = np.concatenate(
        [dets_2d[:, :2], dets_2d[:, :2] + reg_orient], axis=-1)
    side_points = np.zeros((orient.shape[0], 4))

    # cls_orient_argmax = np.argmax(cls_orient, axis=-1)

    row_inds = np.arange(0, cls_orient_argmax.shape[0]).astype(np.int32)

    # two points
    selected_x = np.stack([dets_2d[:, 2], dets_2d[:, 0]], axis=-1)
    # side point
    side_points[:, 3] = dets_2d[:, 3] - reg_orient[:, 1]

    side_points[:, 2] = selected_x[row_inds, cls_orient_argmax]

    # bottom point
    selected_x = np.stack(
        [dets_2d[:, 2] - reg_orient[:, 0], dets_2d[:, 0] + reg_orient[:, 0]],
        axis=-1)
    side_points[:, 1] = dets_2d[:, 3]
    side_points[:, 0] = selected_x[row_inds, cls_orient_argmax]

    side_points[abnormal_cond] = abnormal_case[abnormal_cond]
    return side_points


def calculate_location(dets_3d, p2):
    """
    Args:
        info_3d: shape(N,3)
    """
    K = p2[:3, :3]
    KT = p2[:, -1]
    T = np.dot(np.linalg.inv(K), KT)
    f = K[0, 0]

    h_3d = dets_3d[:, 0]
    lambda_ = f * h_3d / dets_3d[:, 6]

    C_2d = dets_3d[:, 7:9]
    C_2d_homo = np.concatenate([C_2d, np.ones((C_2d.shape[0], 1))], axis=-1)

    C_3d_cam = lambda_ * np.dot(np.linalg.inv(K), C_2d_homo.T)
    C_3d_world = C_3d_cam.T - T

    return C_3d_world


def direction2angle(x, y):
    ry = np.arccos(x / (np.sqrt(x * x + y * y)))
    cond = y < 0
    ry[cond] = -ry[cond]
    return -ry


def twopoints2direction(lines, p2):
    A = lines[:, 3] - lines[:, 1]
    B = lines[:, 0] - lines[:, 2]
    C = lines[:, 2] * lines[:, 1] - lines[:, 0] * lines[:, 3]
    plane = np.dot(p2.T, np.stack([A, B, C], axis=-1).T).T
    a = plane[:, 0]
    c = plane[:, 2]
    ry = direction2angle(c, -a)
    return ry


def fourpoints2direction(points, p2):
    ones = np.ones_like(points[:, -1:])
    points_2d_homo = np.concatenate([points, ones], axis=-1)
    np.dot(p2.T, points_2d_homo)


def mono_3d_postprocess_bbox(dets_3d, dets_2d, p2):
    """
    May be we can improve performance angle prediction by enumerating
    Args:
        dets_3d: shape(N,4) (hwlry)
        dets_2d: shape(N,5) (xyxyc)
        p2: shape(4,3)
    """
    K = p2[:3, :3]
    K_homo = np.eye(4)
    K_homo[:3, :3] = K

    # K*T
    KT = p2[:, -1]
    T = np.dot(np.linalg.inv(K), KT)

    num = dets_3d.shape[0]

    lines = generate_side_points(dets_2d, dets_3d[:, 3:])
    ry = twopoints2direction(lines, p2)

    # ry = np.zeros_like(dets_3d[:, 0])
    # ry[...] = np.pi/2
    # ry = fourpoints2direction(lines)

    # decode h_2ds and c_2ds
    # h = (dets_2d[:, 3] - dets_2d[:, 1] + 1)
    # w = (dets_2d[:, 2] - dets_2d[:, 0] + 1)

    # dets_3d[:, 6] *= h
    # dets_3d[:, 7] = dets_3d[:, 7] * w + dets_2d[:, 0]
    # dets_3d[:, 8] = dets_3d[:, 8] * h + dets_2d[:, 1]
    # C_3d = calculate_location(dets_3d, p2)

    # ry[...] = 1.57
    #  ry_local = dets_3d[:, -1]
    # # compute global angle
    # # center of 2d
    #  center_2d_x = (dets_2d[:, 0] + dets_2d[:, 2]) / 2
    #  center_2d_y = (dets_2d[:, 1] + dets_2d[:, 3]) / 2
    #  center_2d = np.stack([center_2d_x, center_2d_y], axis=-1)
    #  ry = compute_global_angle(center_2d, p2, ry_local)
    C_3d = None
    #  ry = ry_local

    zeros = np.zeros_like(ry)
    ones = np.ones_like(ry)
    R = np.stack(
        [
            np.cos(ry), zeros, np.sin(ry), zeros, ones, zeros, -np.sin(ry),
            zeros, np.cos(ry)
        ],
        axis=-1).reshape(num, 3, 3)

    l = dets_3d[:, 2]
    h = dets_3d[:, 0]
    w = dets_3d[:, 1]
    zeros = np.zeros_like(w)
    x_corners = np.stack(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=-1)
    y_corners = np.stack([zeros, zeros, zeros, zeros, -h, -h, -h, -h], axis=-1)
    z_corners = np.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=-1)

    corners = np.stack([x_corners, y_corners, z_corners], axis=-1)

    # after rotation
    #  corners = np.dot(R, corners)

    top_corners = corners[:, -4:]
    bottom_corners = corners[:, :4]
    diag_corners = bottom_corners[:, [2, 3, 0, 1]]
    #  left_side_corners = bottom_corners[:, [1, 2, 3, 0]]
    #  right_side_corners = bottom_corners[:, [3, 0, 1, 2]]

    # meshgrid
    # 4x4x4 in all

    num_top = top_corners.shape[1]
    num_bottom = bottom_corners.shape[1]
    top_index, bottom_index, side_index = np.meshgrid(
        np.arange(num_top), np.arange(num_bottom), np.arange(num_bottom))

    # in object frame
    # 3d points may be in top and bottom side
    # all corners' shape: (N,M,3)
    top_side_corners = top_corners[:, top_index.ravel()]
    bottom_side_corners = bottom_corners[:, bottom_index.ravel()]

    # 3d points may be in left and right side
    # both left and right are not difference here
    left_side_corners = bottom_corners[:, side_index.ravel()]
    right_side_corners = diag_corners[:, side_index.ravel()]

    num_cases = top_side_corners.shape[1]
    rcnn_3d = []
    for i in range(num):
        # for each detection result

        dets_2d_per = dets_2d[i]
        results_x = []
        errors = []
        for j in range(num_cases):
            # four equations so that four coeff matries
            left_side_corners_per = left_side_corners[i, j]
            right_side_corners_per = right_side_corners[i, j]
            top_side_corners_per = top_side_corners[i, j]
            bottom_side_corners_per = bottom_side_corners[i, j]
            R_per = R[i]

            # left, xmin
            #  RT = np.eye(4)
            #  RT[:3, -1] = np.dot(R_per, left_side_corners_per)
            #  coeff_left = np.dot(K_homo, RT)[0]
            coeff_left = np.asarray([0, 0, dets_2d_per[0]]) - K[0]
            M = np.dot(np.dot(K, R_per), left_side_corners_per)
            bias_left = M[0] - M[2] * dets_2d_per[0]

            # right, xmax
            #  RT = np.eye(4)
            #  RT[:3, -1] = np.dot(R_per, right_side_corners_per)
            #  coeff_right = np.dot(K_homo, RT)[0]
            coeff_right = np.asarray([0, 0, dets_2d_per[2]]) - K[0]
            M = np.dot(np.dot(K, R_per), right_side_corners_per)
            bias_right = M[0] - M[2] * dets_2d_per[2]

            # top, ymin
            #  RT = np.eye(4)
            #  RT[:3, -1] = np.dot(R_per, top_side_corners_per)
            #  coeff_top = np.dot(K_homo, RT)[1]
            coeff_top = np.asarray([0, 0, dets_2d_per[1]]) - K[1]
            M = np.dot(np.dot(K, R_per), top_side_corners_per)
            bias_top = M[1] - M[2] * dets_2d_per[1]

            # bottom, ymax
            #  RT = np.eye(4)
            #  RT[:3, -1] = np.dot(R_per, bottom_side_corners_per)
            #  coeff_bottom = np.dot(K_homo, RT)[1]
            coeff_bottom = np.asarray([0, 0, dets_2d_per[3]]) - K[1]
            M = np.dot(np.dot(K, R_per), bottom_side_corners_per)
            bias_bottom = M[1] - M[2] * dets_2d_per[3]

            A = np.vstack([coeff_left, coeff_top, coeff_right, coeff_bottom])
            b = np.asarray([bias_left, bias_top, bias_right, bias_bottom])
            #  A = coeff_matrix[:, :-1]
            #  b = dets_2d_per[:-1] - A[:, -1]

            # svd reconstruction error
            res = np.linalg.lstsq(A, b)
            # origin of object frame
            results_x.append(res[0] - T)
            # errors
            if len(res[1]):
                errors.append(res[1])
            else:
                errors.append(np.zeros(1))

            #  U, S, V = np.linalg.svd(A)
            #  np.dot(,np.dot(U.T,b))

        results_x = np.stack(results_x, axis=0)
        errors = np.stack(errors, axis=0)
        # idx = errors.argmin()
        # final results
        # idx = match(dets_2d[i, :-1], corners[i], results_x,
        # R[i][np.newaxis, ...], p2)
        X = final_decision(errors, dets_2d[i, :-1], corners[i], results_x,
                           R[i][np.newaxis, ...], p2)
        # X = results_x[idx]
        rcnn_3d.append(X)
    #  import ipdb
    #  ipdb.set_trace()
    translation = np.vstack(rcnn_3d)
    return np.concatenate(
        [dets_3d[:, :3], translation, ry[..., np.newaxis]], axis=-1), C_3d


def final_decision(errors, box_2d, corner, trans, r, p2):
    """
    Two steps to filter final results:
    1. box_2d iou match
    2. errors from svd
    How to combine two constraintions to refine the result
    Args:
    """

    # some parameters
    iou_thresh = None
    error_top_n = 30
    error_filter = False

    # filtered by real condition
    #  z_filter = trans[:, -1] > 0
    #  trans = trans[z_filter]
    #  errors = errors[z_filter]

    # import ipdb
    # ipdb.set_trace()
    if error_filter:
        keep = np.argsort(errors.flatten())[:error_top_n]

        # filtered by errors
        trans = trans[keep]

    # box_2d match
    idx = match(box_2d, corner, trans, r, p2, thresh=iou_thresh)

    return trans[idx]


def generate_coeff(points_3d, line_2d):
    a = None
    b = None
    return a, b


def match(boxes_2d, corners, trans_3d, r, p, thresh=None):
    """
    Args:
        boxes_2d: shape(4)
        corners: shape(8, 3)
        trans_3d: shape(64,3)
        ry: shape(3, 3)
    """
    import ipdb
    ipdb.set_trace()
    corners_3d = np.dot(r, corners.T)
    trans_3d = np.repeat(np.expand_dims(trans_3d.T, axis=1), 8, axis=1)
    corners_3d = corners_3d.transpose((1, 2, 0)) + trans_3d
    corners_3d = corners_3d.reshape(3, -1)
    corners_3d_homo = np.vstack((corners_3d, np.ones(
        (1, corners_3d.shape[1]))))

    corners_2d = np.dot(p, corners_3d_homo)
    corners_2d_xy = corners_2d[:2, :] / corners_2d[2, :]

    corners_2d_xy = corners_2d_xy.reshape(2, 8, -1)
    xmin = corners_2d_xy[0, :, :].min(axis=0)
    ymin = corners_2d_xy[1, :, :].min(axis=0)
    xmax = corners_2d_xy[0, :, :].max(axis=0)
    ymax = corners_2d_xy[1, :, :].max(axis=0)

    boxes_2d_proj = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    #  import ipdb
    #  ipdb.set_trace()
    bbox_overlaps = py_iou(boxes_2d[np.newaxis, ...], boxes_2d_proj)
    if thresh is None:
        idx = bbox_overlaps.argmax(axis=-1)
        return idx
    else:
        keep = np.where(bbox_overlaps > thresh)[1]
        return keep


def py_area(boxes):
    """
    Args:
        boxes: shape(N,M,4)
    """
    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]
    area = width * height
    return area


def py_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: shape(N,4)
        boxes_b: shape(M,4)
    Returns:
        overlaps: shape(N, M)
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]
    boxes_a = np.repeat(np.expand_dims(boxes_a, 1), M, axis=1)
    boxes_b = np.repeat(np.expand_dims(boxes_b, 0), N, axis=0)

    xmin = np.maximum(boxes_a[:, :, 0], boxes_b[:, :, 0])
    ymin = np.maximum(boxes_a[:, :, 1], boxes_b[:, :, 1])
    xmax = np.minimum(boxes_a[:, :, 2], boxes_b[:, :, 2])
    ymax = np.minimum(boxes_a[:, :, 3], boxes_b[:, :, 3])

    w = xmax - xmin
    h = ymax - ymin
    w[w < 0] = 0
    h[h < 0] = 0

    inner_area = w * h
    boxes_a_area = py_area(boxes_a)
    boxes_b_area = py_area(boxes_b)

    iou = inner_area / (boxes_a_area + boxes_b_area - inner_area)
    return iou


def generate_multi_bins(num_bins):
    """
    get all centers of bins
    """
    interval = 2 * np.pi / num_bins
    bin_centers = np.arange(num_bins) * interval
    cond = bin_centers > np.pi
    bin_centers[cond] = bin_centers[cond] - 2 * np.pi
    return bin_centers


def mono_3d_postprocess_angle(dets_3d, dets_2d, p2):
    """
    May be we can improve performance angle prediction by enumerating
    Args:
        dets_3d: shape(N,4) (hwlry)
        dets_2d: shape(N,5) (xyxyc)
        p2: shape(4,3)
    """
    K = p2[:3, :3]
    K_homo = np.eye(4)
    K_homo[:3, :3] = K

    # K*T
    KT = p2[:, -1]
    T = np.dot(np.linalg.inv(K), KT)

    num = dets_3d.shape[0]
    #  ry_local = dets_3d[:, -1]
    #  ry = ry_local

    #  zeros = np.zeros_like(ry)
    #  ones = np.ones_like(ry)
    #  R = np.stack(
    #  [
    #  np.cos(ry), zeros, np.sin(ry), zeros, ones, zeros, -np.sin(ry),
    #  zeros, np.cos(ry)
    #  ],
    #  axis=-1).reshape(num, 3, 3)

    l = dets_3d[:, 2]
    h = dets_3d[:, 0]
    w = dets_3d[:, 1]
    zeros = np.zeros_like(w)
    x_corners = np.stack(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=-1)
    y_corners = np.stack([zeros, zeros, zeros, zeros, -h, -h, -h, -h], axis=-1)
    z_corners = np.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=-1)

    corners = np.stack([x_corners, y_corners, z_corners], axis=-1)

    # after rotation
    #  corners = np.dot(R, corners)

    top_corners = corners[:, -4:]
    bottom_corners = corners[:, :4]
    diag_corners = bottom_corners[:, [2, 3, 0, 1]]
    #  left_side_corners = bottom_corners[:, [1, 2, 3, 0]]
    #  right_side_corners = bottom_corners[:, [3, 0, 1, 2]]

    # meshgrid
    # 4x4x4 in all

    num_top = top_corners.shape[1]
    num_bottom = bottom_corners.shape[1]
    top_index, bottom_index, side_index = np.meshgrid(
        np.arange(num_top), np.arange(num_bottom), np.arange(num_bottom))

    # in object frame
    # 3d points may be in top and bottom side
    # all corners' shape: (N,M,3)
    top_side_corners = top_corners[:, top_index.ravel()]
    bottom_side_corners = bottom_corners[:, bottom_index.ravel()]

    # 3d points may be in left and right side
    # both left and right are not difference here
    left_side_corners = bottom_corners[:, side_index.ravel()]
    right_side_corners = diag_corners[:, side_index.ravel()]

    num_cases = top_side_corners.shape[1]
    rcnn_3d = []
    rcnn_ry = []

    num_bins = 30

    bin_centers = generate_multi_bins(num_bins)
    #  import ipdb
    #  ipdb.set_trace()
    for i in range(num):
        # for each detection result

        dets_2d_per = dets_2d[i]
        results_x = []
        errors = []
        ry = []
        R = []
        for bin_center in bin_centers:
            # generate Rotation matrix
            R_per = roty(bin_center)
            for j in range(num_cases):
                # four equations so that four coeff matries
                left_side_corners_per = left_side_corners[i, j]
                right_side_corners_per = right_side_corners[i, j]
                top_side_corners_per = top_side_corners[i, j]
                bottom_side_corners_per = bottom_side_corners[i, j]
                #  R_per = R[i]

                # left, xmin
                #  RT = np.eye(4)
                #  RT[:3, -1] = np.dot(R_per, left_side_corners_per)
                #  coeff_left = np.dot(K_homo, RT)[0]
                coeff_left = np.asarray([0, 0, dets_2d_per[0]]) - K[0]
                M = np.dot(np.dot(K, R_per), left_side_corners_per)
                bias_left = M[0] - M[2] * dets_2d_per[0]

                # right, xmax
                #  RT = np.eye(4)
                #  RT[:3, -1] = np.dot(R_per, right_side_corners_per)
                #  coeff_right = np.dot(K_homo, RT)[0]
                coeff_right = np.asarray([0, 0, dets_2d_per[2]]) - K[0]
                M = np.dot(np.dot(K, R_per), right_side_corners_per)
                bias_right = M[0] - M[2] * dets_2d_per[2]

                # top, ymin
                #  RT = np.eye(4)
                #  RT[:3, -1] = np.dot(R_per, top_side_corners_per)
                #  coeff_top = np.dot(K_homo, RT)[1]
                coeff_top = np.asarray([0, 0, dets_2d_per[1]]) - K[1]
                M = np.dot(np.dot(K, R_per), top_side_corners_per)
                bias_top = M[1] - M[2] * dets_2d_per[1]

                # bottom, ymax
                #  RT = np.eye(4)
                #  RT[:3, -1] = np.dot(R_per, bottom_side_corners_per)
                #  coeff_bottom = np.dot(K_homo, RT)[1]
                coeff_bottom = np.asarray([0, 0, dets_2d_per[3]]) - K[1]
                M = np.dot(np.dot(K, R_per), bottom_side_corners_per)
                bias_bottom = M[1] - M[2] * dets_2d_per[3]

                A = np.vstack(
                    [coeff_left, coeff_top, coeff_right, coeff_bottom])
                b = np.asarray([bias_left, bias_top, bias_right, bias_bottom])
                #  A = coeff_matrix[:, :-1]
                #  b = dets_2d_per[:-1] - A[:, -1]

                # svd reconstruction error
                res = np.linalg.lstsq(A, b)
                # origin of object frame
                results_x.append(res[0] - T)
                # errors
                if len(res[1]):
                    errors.append(res[1])
                else:
                    errors.append(np.zeros(1))

                ry.append(bin_center)
                R.append(R_per)

        results_x = np.stack(results_x, axis=0)
        errors = np.stack(errors, axis=0)
        ry = np.stack(ry, axis=0)
        R = np.stack(R, axis=0)

        # two methods
        #  import ipdb
        #  ipdb.set_trace()

        # import ipdb
        # ipdb.set_trace()
        # orient_filter
        center_orients = dets_3d[i, 11].astype(np.int32)
        orient_filter = get_cls_orient(results_x, ry, p2, center_orients)
        z_filter = results_x[:, -1] > 0

        orient_filter = orient_filter & z_filter

        # if orient_filter.sum()==0:
        # import ipdb
        # ipdb.set_trace()

        results_x = results_x[orient_filter]
        R = R[orient_filter]
        errors = errors[orient_filter]
        ry = ry[orient_filter]

        # 2d_filter
        keep = match(
            dets_2d[i, :-1], corners[i], results_x, R, p2, thresh=None)
        # if keep.size:
        # errors = errors[keep]
        # results_x = results_x[keep]
        # ry = ry[keep]
        # R = R[keep]
        # idx = errors.argmin()
        idx = keep
        if errors[idx] > 1e3:
            print('error is too large: {}'.format(errors[idx]))

        # top cases
        #  import ipdb
        #  ipdb.set_trace()
        #  keep = np.argsort(errors.flatten())[:30]
        #  results_x = results_x[keep]
        #  ry = ry[keep]
        #  R = R[keep]
        #  idx = match(dets_2d[i, :-1], corners[i], results_x, R, p2)
        X = results_x[idx]
        rcnn_3d.append(X)
        rcnn_ry.append(ry[idx])
    #  import ipdb
    #  ipdb.set_trace()
    translation = np.vstack(rcnn_3d)
    rcnn_ry = np.vstack(rcnn_ry)
    #  import ipdb
    #  ipdb.set_trace()
    return np.concatenate([dets_3d[:, :3], translation, rcnn_ry], axis=-1)


def get_center_orient():
    pass


def get_cls_orient(trans, ry, p2, cls_orient_pred):
    """
    Args:
        trans: shape(N, 3)
    """
    # ry[...] = 1.57
    trans_homo = np.concatenate([trans, np.ones((trans.shape[0], 1))], axis=-1)
    C_2d_homo = np.dot(p2, trans_homo.T).T
    C_2d_homo /= C_2d_homo[:, -1:]
    C_2d = C_2d_homo[:, :2]

    num = trans.shape[0]
    deltas = np.asarray([10, 0, 0])

    zeros = np.zeros_like(ry)
    ones = np.ones_like(ry)
    R = np.stack(
        [
            np.cos(ry), zeros, np.sin(ry), zeros, ones, zeros, -np.sin(ry),
            zeros, np.cos(ry)
        ],
        axis=-1).reshape(num, 3, 3)
    trans2 = np.dot(R, deltas) + trans
    trans2_homo = np.concatenate(
        [trans2, np.ones((trans2.shape[0], 1))], axis=-1)
    C_2d2_homo = np.dot(p2, trans2_homo.T).T
    C_2d2_homo /= C_2d2_homo[:, -1:]
    C_2d2 = C_2d2_homo[:, :2]
    line = [C_2d, C_2d2]

    direction = line[0] - line[1]
    cond = direction[:, 0] * direction[:, 1]
    cls_orient = np.zeros_like(direction[:, 0]).astype(np.int32)
    cls_orient[cond == 0] = -1
    cls_orient[cond > 0] = 1
    cls_orient[cond < 0] = 0

    orient_filter = (cls_orient == cls_orient_pred.astype(np.int32)) | (
        cls_orient == -1)
    return orient_filter


def get_depth_coords_v2():
    pass


def mono_3d_postprocess_depth(dets_3d, dets_2d, p2):
    """
    May be we can improve performance angle prediction by enumerating
    when given locations of cars
    Args:
        dets_3d: shape(N,3 + 3 + 1 + 2 + 2 + 1) (hwl, ind, deltas, angle, d_y)
        dets_2d: shape(N,5) (xyxyc)
        p2: shape(4,3)
    """

    K = p2[:3, :3]
    K_homo = np.eye(4)
    K_homo[:3, :3] = K

    # K*T
    KT = p2[:, -1]
    T = np.dot(np.linalg.inv(K), KT)

    num = dets_3d.shape[0]

    l = dets_3d[:, 2]
    h = dets_3d[:, 0]
    w = dets_3d[:, 1]
    zeros = np.zeros_like(w)
    x_corners = np.stack(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=-1)
    y_corners = np.stack([zeros, zeros, zeros, zeros, -h, -h, -h, -h], axis=-1)
    z_corners = np.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=-1)

    corners = np.stack([x_corners, y_corners, z_corners], axis=-1)

    rcnn_3d = []
    rcnn_ry = []

    num_bins = 30

    bin_centers = generate_multi_bins(num_bins)
    # import ipdb
    # ipdb.set_trace()
    center_2d_x = (dets_2d[:, 0] + dets_2d[:, 2]) / 2
    center_2d_y = (dets_2d[:, 1] + dets_2d[:, 3]) / 2
    center_2d = np.stack([center_2d_x, center_2d_y], axis=-1)
    ray_angle = compute_ray_angle(center_2d, p2)
    angle = np.stack([np.cos(ray_angle), np.sin(ray_angle)], axis=-1)

    locations = get_depth_coords(dets_3d[:, 6:17], dets_3d[:, 17:19], angle,
                                 dets_3d[:, 19:])
    for i in range(num):
        ry = []
        R = []
        for bin_center in bin_centers:
            # generate Rotation matrix
            R_per = roty(bin_center)
            R.append(R_per)
            ry.append(bin_center)
        ry = np.stack(ry, axis=0)
        R = np.stack(R, axis=0)
        results_x = np.stack([locations[i]] * R.shape[0], axis=0)

        # 2d_filter
        idx = match(dets_2d[i, :-1], corners[i], results_x, R, p2, thresh=None)
        rcnn_3d.append(results_x[idx])
        rcnn_ry.append(ry[idx])
    translation = np.vstack(rcnn_3d)
    rcnn_ry = np.vstack(rcnn_ry)

    return np.concatenate([dets_3d[:, :3], translation, rcnn_ry], axis=-1)
