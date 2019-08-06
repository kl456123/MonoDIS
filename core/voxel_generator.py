# -*- coding: utf-8 -*-

import torch
import core.ops as ops
import numpy as np


class VoxelGenerator(object):
    def __init__(self, voxel_generator_config):
        self.voxel_size = torch.tensor(
            voxel_generator_config['voxel_size']).cuda().float()
        # xyz(meters)
        self.grid_dims = torch.tensor(
            voxel_generator_config['grid_dims']).cuda().float()
        # from bottom to top
        self.high_interval = voxel_generator_config['high_interval']
        # self.y0 = voxel_generator_config['ground_plane']
        self.z_offset = voxel_generator_config['z_offset']
        # self.original_offset = torch.tensor(
        # voxel_generator_config['original_offset']).cuda().float()

        self.voxel_centers = None
        self.voxel_proj_2d = None

    def init_voxels(self):
        self.generate_voxels()
        # self.proj_voxels_3dTo2d(p2)

    def generate_voxels(self):
        """
        generate all voxels in ground plane
        """
        lattice_dims = self.grid_dims / self.voxel_size
        self.lattice_dims = lattice_dims
        x_inds = torch.arange(0, lattice_dims[0]).cuda()
        y_inds = torch.arange(0, lattice_dims[1]).cuda()
        z_inds = torch.arange(0, lattice_dims[2]).cuda()
        z_inds, x_inds = ops.meshgrid(z_inds, x_inds)

        y_inds1, z_inds = ops.meshgrid(y_inds, z_inds)
        y_inds2, x_inds = ops.meshgrid(y_inds, x_inds)
        y_inds = y_inds1

        corner_coords = torch.stack([x_inds, y_inds, z_inds], dim=-1).float()
        corner_coords *= self.voxel_size

        center_offset = torch.tensor([0.5 * self.voxel_size] *
                                     3).type_as(corner_coords)
        center_coords = corner_coords + center_offset

        high_interval = self.high_interval
        # bugs here
        # y_offset = (high_interval[0] + high_interval[1]) / 2
        y_offset = high_interval[0]

        original_offset = [-0.5 * self.grid_dims[0], y_offset, self.z_offset]
        # original_offset  = self.original_offset
        original_offset = torch.tensor(original_offset).type_as(center_coords)
        center_coords = center_coords + original_offset

        # tmp_coords = torch.tensor(
        # [[-16.53, 2.39, 58.49]]).type_as(center_coords)

        # center_coords = torch.cat([center_coords, tmp_coords])
        self.voxel_centers = center_coords

        return center_coords

    @staticmethod
    def project_to_image(pts_3d, p2):
        """
        Args:
            pts_3d: shape(N, 3)
            p2: shape(3, 4)
        Returns:
            pts_2d: shape(N, 2)
        """

        ones = torch.ones_like(pts_3d[:, -1:])
        pts_3d_homo = torch.cat([pts_3d, ones], dim=-1).transpose(1, 0)
        pts_2d_homo = p2.matmul(pts_3d_homo).transpose(2, 1)

        # inplace op is prohibited here
        pts_2d_homo = pts_2d_homo / pts_2d_homo[:, :, -1:]
        return pts_2d_homo[:, :, :-1]

    def proj_voxels_to_ground(self):
        """
        Proj voxels to ground plane to get their corners
        """
        voxel_centers = self.voxel_centers

        x = voxel_centers[:, 0]
        z = voxel_centers[:, 2]
        xmin = x - 0.5 * self.voxel_size
        zmin = z - 0.5 * self.voxel_size
        xmax = x + 0.5 * self.voxel_size
        zmax = z + 0.5 * self.voxel_size

        voxels_ground_2d = torch.stack([zmin, xmin, zmax, xmax], dim=-1)

        # Only one slice of voxels remains
        lattice_dims = self.lattice_dims.cpu().numpy()
        D = int(lattice_dims[1])
        voxels_ground_2d = voxels_ground_2d.view(-1, D, 4)
        return voxels_ground_2d[:, 0, :]

    def proj_voxels_3dTo2d(self, p2, img_size):
        """
        Project bbox in 3d to bbox in 2d
        """
        # compute rotational matrix around yaw axis
        # R = self.roty(obj.ry)

        # 3d bounding box dimensions
        l = h = w = self.voxel_size

        # 3d bounding box corners
        x_corners = torch.tensor(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        y_corners = torch.tensor([0, 0, 0, 0, -h, -h, -h, -h])
        z_corners = torch.tensor(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

        corners_3d = torch.stack(
            [x_corners, y_corners, z_corners],
            dim=-1).type_as(self.voxel_centers)
        corners_3d = corners_3d + self.voxel_centers.unsqueeze(1)

        # import ipdb
        # ipdb.set_trace()

        corners_2d = self.project_to_image(corners_3d.view(-1, 3), p2).view(
            -1, 8, 2)

        # find the min bbox
        # corners_2d: shape(N, 8, 2)
        xmin, _ = torch.min(corners_2d[:, :, 0], dim=-1)
        ymin, _ = torch.min(corners_2d[:, :, 1], dim=-1)
        xmax, _ = torch.max(corners_2d[:, :, 0], dim=-1)
        ymax, _ = torch.max(corners_2d[:, :, 1], dim=-1)
        boxes_2d = torch.stack([xmin, ymin, xmax, ymax], dim=-1)

        self.voxel_proj_2d = boxes_2d

        normalized_boxes_2d = torch.stack(
            [
                xmin / img_size[:, 1], ymin / img_size[:, 0],
                xmax / img_size[:, 1], ymax / img_size[:, 0]
            ],
            dim=-1)
        # normalize it
        self.normalized_voxel_proj_2d = normalized_boxes_2d

        return boxes_2d


if __name__ == '__main__':
    voxel_generator_config = {
        'voxel_size': 0.5,
        'grid_dims': [80, 4, 80],
        'high_interval': [-2, 3],
        'ground_plane': 0.5
    }
    p2 = np.asarray([[
        7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00,
        7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00,
        1.000000e+00, 2.745884e-03
    ]]).reshape((3, 4))
    p2 = torch.tensor(p2).cuda().float()
    voxel_generator = VoxelGenerator(voxel_generator_config)
    voxel_generator.init_voxels(p2)
