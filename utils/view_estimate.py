# -*- coding: utf-8 -*-

import numpy as np

P2 = np.asarray([[
    7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00,
    7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00,
    1.000000e+00, 2.745884e-03
]]).reshape((3, 4))

K = P2[:, :3]
KT = P2[:, 3]
K_inv = np.linalg.inv(K)
T = np.dot(K_inv, KT)

point_2ds = [[0, 0, 1], [1280, 384, 1], [0, 384, 1], [1280, 0, 1]]


def point2angle(point_2d, D):
    point_3d = np.dot(K_inv, point_2d) * D
    return point_3d


def view_estimate():
    point_3ds = []
    for point_2d in point_2ds:
        D = 80
        point_3d = point2angle(point_2d, D)
        point_3ds.append(point_3d)
        # print(point_3ds)

    deltas = point_3ds[0] - point_3ds[3]
    print(deltas)


def data_analysis():
    from utils.orient_eval import read_labels, label_dir
    import os
    total_3ds_label = np.empty((0, 7))
    # collect
    for lbl_file in os.listdir(label_dir):
        sample_name = os.path.splitext(lbl_file)[0]
        box_2ds_label, box_3ds_label = read_labels(label_dir, sample_name)
        total_3ds_label = np.append(total_3ds_label, box_3ds_label, axis=0)

    # analysis
    dim = total_3ds_label[:, :3]
    pos = total_3ds_label[:, 3:6]
    ry = total_3ds_label[:, 6:]
    dim_mean = dim.mean(axis=0)
    pos_max = pos.max(axis=0)
    pos_min = pos.min(axis=0)
    print(dim_mean)
    print(pos_max)
    print(pos_min)
    return pos


def draw_3d(pos):
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D

    fig = pyplot.figure()
    ax = Axes3D(fig)

    #  sequence_containing_x_vals = list(range(0, 100))
    #  sequence_containing_y_vals = list(range(0, 100))
    #  sequence_containing_z_vals = list(range(0, 100))

    #  random.shuffle(sequence_containing_x_vals)
    #  random.shuffle(sequence_containing_y_vals)
    #  random.shuffle(sequence_containing_z_vals)

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    pyplot.show()


if __name__ == '__main__':
    #  import ipdb
    #  ipdb.set_trace()
    pos = data_analysis()
    draw_3d(pos)
    from utils.analysis import data_vis
    data_vis(pos[:, 1])
    data_vis(pos[:,0])
    data_vis(pos[:,2])
