# -*- coding: utf-8 -*-

# What's the relationship between 2d features and yaw angle
# The scripts simulates the influence with the angle changes

import numpy as np
import os
from PIL import Image, ImageDraw
from utils.box_vis import draw_boxes

P2 = np.asarray([[
    7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00,
    7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00,
    1.000000e+00, 2.745884e-03
]]).reshape((3, 4))

K = P2[:3, :3]
KT = P2[:, -1]
T = np.dot(np.linalg.inv(K), KT)
C = -T


class Object3d(object):
    def __init__(self, dims, location, ry):
        """
        Args:
            dims: (hwl)
            location: (xyz)
            ry: rotation along y axis
        """
        self.dims = dims
        self.location = location
        self.ry = ry

    def translate(self, trans):
        self.location += trans

    def dict(self):
        ret_dict = {
            'dimension': self.dims,
            'ry': self.ry,
            'location': self.location
        }
        return ret_dict

    def numpy(self, batch_format=False):
        ret_np = []
        ret_np += [self.ry]
        ret_np += self.dims
        ret_np += self.location
        if batch_format:
            ret_np = np.asarray([ret_np])
        return np.asarray(ret_np)

    def rotate(self, ry):
        """
        Note that clockwise angle is positive, otherwise is negative
        """
        self.ry += ry

    def set_rotation(self, ry):
        self.ry = ry

    @staticmethod
    def get_rotation_matrix(rotation_y):
        r = [
            np.cos(rotation_y), 0, np.sin(rotation_y), 0, 1, 0,
            -np.sin(rotation_y), 0, np.cos(rotation_y)
        ]
        return r

    def get_orientation(self):
        location = self.location
        theta = np.arctan2(location[2], location[0])
        return -theta

    def get_points(self):
        """
        8 points representation for 3d box
        Returns:
            corners: shape(8, 3)
        """
        h, w, l = self.dims
        x_corners = np.stack(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            axis=-1)
        y_corners = np.stack([0, 0, 0, 0, -h, -h, -h, -h], axis=-1)
        z_corners = np.stack(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            axis=-1)
        corners = np.stack([x_corners, y_corners, z_corners], axis=-1)

        # rotation and translation
        r = self.get_rotation_matrix(self.ry)
        corners_3d = np.dot(r, corners)
        corners_3d = corners_3d + np.array(self.location).reshape(3, 1)
        return corners


class Projector(object):
    def __init__(self, p2):
        self.p2 = p2

    def proj_2to3(self):
        pass

    def proj_3to2(self):
        pass


def main():
    dims = [1.6, 1.7, 3.8]
    location = [-12, 2, 20]
    init_ry = 0
    object_3d = Object3d(dims=dims, location=location, ry=init_ry)

    num = 4
    ry_step = 2* np.pi/num
    # ry_step = 10 / 180 * np.pi
    # num = 360 / 10
    offset = object_3d.get_orientation()
    ry_options = np.arange(num) * ry_step + offset

    img_path = '/data/object/training/image_2/000002.png'
    fv_dir = './results/test_fv'
    bev_dir = './results/test_bev'
    # img

    # import ipdb
    # ipdb.set_trace()
    for ind, ry in enumerate(ry_options):
        img = Image.open(img_path)
        object_3d.set_rotation(ry)
        #  points_3d = object_3d.get_points()
        object_dict = object_3d.numpy(batch_format=True)

        #  save_path = os.path.join(save_dir, )
        save_name = '{}.png'.format(ind)
        # final draw it
        draw_boxes(
            img,
            object_dict,
            P2,
            save_name,
            title=img_path,
            fv_dir=fv_dir,
            bev_dir=bev_dir,
            display=False)


if __name__ == '__main__':
    main()
