import os
import numpy as np
from PIL import Image, ImageDraw

# from .data_loader import DetDataLoader
from data.det_dataset import DetDataset
from data.kitti_utils import Calibration
from data.bev_generators.bev_slices import BevSlices
from data.kitti_utils import Object3d
from data.bev_generators.box_3d_projector import *
from utils.bev_encoder import DataEncoder
import torch


class KITTIBEVDataset(DetDataset):
    def __init__(self, dataset_config, transforms=None, training=True):
        super(KITTIBEVDataset, self).__init__(training)
        self.config = dataset_config
        self._data_root = self.config['root_path']
        self._set_up_directories()
        self._area_extents = self.config['bev_config']['area_extents']

        # cache bev map
        self.cache_bev = self.config['cache_bev']
        # self.cache_dir = self.config['cache_dir']

        if self.config['dataset_file'] is None:
            print('Demo mode enabled!')
            self.imgs = [dataset_config['demo_file']]
        else:
            self.imgs = self.make_image_list()
        self.camera_baseline = self.config['camera_baseline']
        # self.data_encoder = data_encoder
        self.transforms = transforms

    def _set_up_directories(self):
        training_dir = self._data_root
        self.image_dir = os.path.join(training_dir, 'image_2')
        self.depth_dir = os.path.join(training_dir, 'depth')
        self.calib_dir = os.path.join(training_dir, 'calib')
        self.label_dir = os.path.join(training_dir, 'label_2')
        self.plane_dir = os.path.join(training_dir, 'planes_org')
        self.training_dir = training_dir

    def get_transform_sample(self, index):
        sample_name = int(self.imgs[index])
        bev_map = self.get_bev(sample_name)
        transform_sample = {
            'img': bev_map,
            'im_scale': 1.0,
            'img_name':
            os.path.join(self.depth_dir, self.imgs[index] + '.png'),
        }
        if self.training:
            bbox, ry, labels = self.get_label(sample_name)

            # Transform the bbox to bbox format(xmin, ymin, xmax, ymax)
            h, w = bev_map.shape[:2]
            xrange = self._area_extents[0][1] - self._area_extents[0][0]
            yrange = self._area_extents[2][1] - self._area_extents[2][0]
            bbox[:, 0] *= (w / xrange)
            bbox[:, 2] *= (w / xrange)
            bbox[:, 1] *= (h / yrange)
            bbox[:, 3] *= (h / yrange)

            transform_sample.update({
                'bbox': bbox,
                'ry': ry,
                'label': labels,
            })
        if self.cache_bev:
            transform_sample.update({'img_orig': bev_map.copy()})

        return transform_sample

    def get_training_sample(self, transform_sample):
        # TODO some tranform ops can be moved to transform.py
        # to make here clear

        bev_map = transform_sample['img']
        h, w = bev_map.shape[1:]
        im_scale = transform_sample['im_scale']
        img_info = torch.FloatTensor([h, w, im_scale])

        if self.training:
            bbox = transform_sample['bbox']
            ry = transform_sample['ry']

            bbox = torch.cat((bbox, torch.ones((bbox.size()[0], 1))), dim=1)

            ry = torch.FloatTensor(ry)
            ry = torch.cat(
                [torch.cos(ry).unsqueeze(1), torch.sin(ry).unsqueeze(1)], 1)
            num = torch.LongTensor([bbox.size()[0]])
        else:
            # fake labels
            bbox = torch.zeros((1, 5))
            num = torch.Tensor(1)
            ry = torch.zeros((1, 2))

        training_sample = {}
        training_sample['bev_map'] = bev_map
        training_sample['im_info'] = img_info
        training_sample['num'] = num
        training_sample['bbox'] = bbox
        training_sample['ry'] = ry
        training_sample['img_name'] = transform_sample['img_name']
        training_sample['img'] = transform_sample['img']
        if self.cache_bev:
            training_sample['img_orig'] = transform_sample['img_orig']

        return training_sample

    def __getitem__(self, index, if_vis=False):

        transform_sample = self.get_transform_sample(index)

        if self.transforms is not None:
            transform_sample = self.transforms(transform_sample)

        return self.get_training_sample(transform_sample)

    # if if_vis:
    # print(('sample_name is:', sample_name))
    # gt_boxes, pos_boxes = self.data_encoder.encode(
    # bev_map, bbox, ry, labels, if_vis=if_vis)
    # return sample_name, gt_boxes, pos_boxes

    # loc_target, ry_target, cls_target = self.data_encoder.encode(bev_map, bbox, ry, labels)
    #####
    # bev_map (6,800,700)
    # bbox (1,4) non-normalized (xmin,ymin,...)
    # ry angle

    # return bev_map, img_info, bbox, , ry, sample_name

    def make_image_list(self):
        val_file_path = os.path.join(self.training_dir,
                                     self.config['dataset_file'])
        with open(val_file_path) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            car_files = [line for line in lines if self._check_type(int(line))]

        return car_files

    def get_rgb_img(self, sample_name):
        img_file_name = sample_name + '.png'
        img_file_path = os.path.join(self.image_dir, img_file_name)
        img = Image.open(img_file_path)
        return img

    def get_pc_from_depth(self, sample_name, is_sparse=True):
        depth_file_name = '{:0>6d}.png'.format(sample_name)
        depth_file_path = os.path.join(self.depth_dir, depth_file_name)

        calib_file_name = '{:0>6d}.txt'.format(sample_name)
        calib_file_path = os.path.join(self.calib_dir, calib_file_name)
        calib = Calibration(calib_file_path)

        # Get the points coordinates in the rect camera coordinates
        img = Image.open(depth_file_path)
        depth = np.array(img).astype(np.uint16) / 256.0

        points = []
        space = 2 if is_sparse else 1
        for v in range(0, depth.shape[0], space):
            for u in range(0, depth.shape[1], space):
                tmp = (calib.f_u * self.camera_baseline) / float(depth[v][u])
                pointtmp = [u, v, tmp]
                points.append(pointtmp)
        points_depth = calib.project_image_to_rect(np.array(points))
        points_depth = np.array(points_depth)

        return points_depth

    def get_bev(self, sample_name):
        ground_plane = self._get_road_plane(sample_name)
        points_depth = self.get_pc_from_depth(sample_name)

        bev_images = BevSlices(self.config['bev_config']).generate_bev(
            points_depth.transpose(), ground_plane)
        return np.transpose(bev_images, (1, 2, 0))  # transpose to HxWxC

    def _check_type(self, sample_name, type='Car'):
        label_file_name = '{:0>6d}.txt'.format(sample_name)
        label_file_path = os.path.join(self.label_dir, label_file_name)
        lines = [line.rstrip() for line in open(label_file_path)]
        objects = [Object3d(line) for line in lines]
        cars_3d = [obj.box3d_avod for obj in objects if obj.type == type]
        if cars_3d:
            return True
        else:
            return False

    def get_label(self, sample_name):
        label_file_name = '{:0>6d}.txt'.format(sample_name)
        label_file_path = os.path.join(self.label_dir, label_file_name)
        lines = [line.rstrip() for line in open(label_file_path)]
        objects = [Object3d(line) for line in lines]
        cars_3d = [obj.box3d_avod for obj in objects if obj.type == 'Car']
        cars_3d = np.array(cars_3d)

        # Normalize the coords
        if cars_3d.any():
            # x, y, z, l, w, h, ry
            gt_anchors = box_3d_to_anchor(cars_3d)
            gt_boxes_for_2d_iou, _ = project_to_bev(
                gt_anchors, [self._area_extents[0], self._area_extents[2]])
        else:
            return None, None, None

        return gt_boxes_for_2d_iou, cars_3d[:, -1], np.zeros(
            (gt_anchors.shape[0], ))

    def _get_road_plane(self, sample_name):
        plane_file_name = '{:0>6d}.txt'.format(sample_name)
        plane_file_path = os.path.join(self.plane_dir, plane_file_name)

        with open(plane_file_path, 'r') as input_file:
            lines = input_file.readlines()
            input_file.close()

        # Plane coefficients stored in 4th row
        lines = lines[3].split()

        # Convert str to float
        lines = [float(i) for i in lines]

        plane = np.asarray(lines)

        # Ensure normal is always facing up.
        # In Kitti's frame of reference, +y is down
        if plane[1] > 0:
            plane = -plane

        # Normalize the plane coefficients
        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm

        return plane


# if __name__ == '__main__':
# MODEL_CONFIG = {
# 'num_classes': 2,
# 'output_stride': [4., 8., 16., 32.],
# # 'default_ratio': [0.0449, 0.0772, 0.115, 0.164, 0.227],
# 'default_ratio': [0.034, 0.034, 0.034, 0.034, 0.034],
# 'aspect_ratio': ((1.54, 2.39), (1.54, 2.39), (1.54, 2.39),
# (1.54, 2.39)),
# 'input_shape': (800, 700),
# }
# data_transfer = trans.Compose([trans.ToTensor(), ])
# data_encoder = DataEncoder(MODEL_CONFIG)
# kitti_bev_loader = KITTIBEVLoader(data_config, data_encoder, data_transfer)

# for test_sample in range(100):
# sample_name, gt_boxes, pos_boxes = kitti_bev_loader.__getitem__(
# test_sample, True)
# bev = kitti_bev_loader.get_bev(sample_name)
# density = bev[:, :, 0] * 255
# density = density.astype(np.uint8)
# density_img = Image.fromarray(density).convert('RGB')
# h, w = density.shape[:2]
# gt_boxes[:, 0], pos_boxes[:, 0] = gt_boxes[:, 0] * w, pos_boxes[:,
# 0] * w
# gt_boxes[:, 2], pos_boxes[:, 2] = gt_boxes[:, 2] * w, pos_boxes[:,
# 2] * w
# gt_boxes[:, 1], pos_boxes[:, 1] = gt_boxes[:, 1] * h, pos_boxes[:,
# 1] * h
# gt_boxes[:, 3], pos_boxes[:, 3] = gt_boxes[:, 3] * h, pos_boxes[:,
# 3] * h

# draw = ImageDraw.Draw(density_img)
# gt_boxes = gt_boxes.astype(np.int).tolist()
# pos_boxes = pos_boxes.astype(np.int).tolist()
# for gt_box in gt_boxes:
# draw.rectangle(gt_box, outline=(0, 0, 255))
# # for pos_box in pos_boxes:
# #     draw.rectangle(pos_box, outline=(255, 0, 0))
# density_img.save('../result/{:0>6d}_density.png'.format(sample_name))
