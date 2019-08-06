import os
import cv2
import numpy
import torch

from PIL import Image
from data.det_dataset import DetDataset
from utils.box_vis import load_projection_matrix
from utils.kitti_util import *
import copy
from utils import geometry_utils

color_map = [(0, 0, 142)]

MEAN_DIMS = {
    'Car': [3.88311640418, 1.62856739989, 1.52563191462],
    'Van': [5.06763659, 1.9007158, 2.20532825],
    'Truck': [10.13586957, 2.58549199, 3.2520595],
    'Pedestrian': [0.84422524, 0.66068622, 1.76255119],
    'Person_sitting': [0.80057803, 0.5983815, 1.27450867],
    'Cyclist': [1.76282397, 0.59706367, 1.73698127],
    'Tram': [16.17150617, 2.53246914, 3.53079012],
    'Misc': [3.64300781, 1.54298177, 1.92320313]
}


class Mono3DKittiDataset(DetDataset):
    def __init__(self, dataset_config, transforms=None, training=True):
        super(Mono3DKittiDataset, self).__init__(training)
        self.root_path = os.path.join(dataset_config['root_path'],
                                      'object/training')
        classes = dataset_config.get('classes')
        if classes is None:
            self.classes = ['Car']
        else:
            self.classes = classes
        # if self.training:
        if dataset_config['dataset_file'] is None:
            print('Demo mode enabled!')
            if dataset_config.get('demo_file') is not None:
                self.imgs = [dataset_config['demo_file']]
                self.calib_file = dataset_config['calib_file']
            else:
                self.imgs = self._read_imgs_from_dir(dataset_config['img_dir'])
                self.calib_file = dataset_config['calib_file']
        else:
            self.labels = self.make_label_list(dataset_config['dataset_file'])
            self.imgs = self.make_image_list()
            self.calibs = self.make_calib_list()

        self.transforms = transforms
        self.max_num_gt_boxes = 40
        self.use_rect_v2 = dataset_config['use_rect_v2']
        self.use_proj_2d = dataset_config['use_proj_2d']

    def _read_imgs_from_dir(self, img_dir):
        imgs = []
        for img_name in sorted(os.listdir(img_dir)):
            imgs.append(os.path.join(img_dir, img_name))
        return imgs

    def get_training_sample(self, transform_sample):
        # bbox and num
        img = transform_sample['img']
        im_scale = transform_sample['im_scale']
        w = img.size()[2]
        h = img.size()[1]
        if type(im_scale) is float:
            img_info = torch.FloatTensor([h, w, im_scale])
        else:
            img_info = torch.FloatTensor([h, w, *im_scale])

        if self.training:
            bbox = transform_sample['bbox']
            # For car, the label is one
            # import ipdb
            # ipdb.set_trace()
            bbox = torch.cat(
                (bbox, transform_sample['label'].unsqueeze(-1).float()), dim=1)
            num = torch.LongTensor([bbox.size()[0]])
            bbox_3d = transform_sample['bbox_3d']
            coords = transform_sample['coords']
            coords_uncoded = transform_sample['coords_uncoded']
            points_3d = transform_sample['points_3d']
            dims_2d = transform_sample['dims_2d']
            oritations = transform_sample['oritation']
            local_angle_oritations = transform_sample['local_angle_oritation']
            boxes_2d_proj = transform_sample['boxes_2d_proj']
            local_angles = transform_sample['local_angle']
            cls_orients = transform_sample['cls_orient']
            reg_orients = transform_sample['reg_orient']
            h_2ds = transform_sample['h_2d']
            c_2ds = transform_sample['c_2d']
            r_2ds = transform_sample['r_2d']
            cls_orient_4s = transform_sample['cls_orient_4']
            center_orients = transform_sample['center_orient']

            # angles_camera = transform_sample['angles_camera']
            # distances = transform_sample['distance']
            d_ys = transform_sample['d_y']
            gt_boxes_2d_ground_rect = transform_sample[
                'gt_boxes_2d_ground_rect']
            gt_boxes_2d_ground_rect_v2 = transform_sample[
                'gt_boxes_2d_ground_rect_v2']
            encoded_side_points = transform_sample['encoded_side_points']
            encoded_bottom_points = transform_sample['encoded_bottom_points']
            keypoint_gt = transform_sample['keypoint_gt']
            keypoint_gt_weights = transform_sample['keypoint_gt_weights']
            used = transform_sample['used']
        else:
            # fake gt
            bbox = torch.zeros((1, 5))
            num = torch.Tensor(1)
            bbox_3d = torch.zeros((1, 7))
            coords = torch.zeros((1, 7))
            coords_uncoded = torch.zeros((1, 7))
            points_3d = torch.zeros((1, 8))
            dims_2d = torch.zeros((1, 3))
            oritations = torch.zeros((1, 2))
            local_angle_oritations = torch.zeros((1, 2))
            boxes_2d_proj = torch.zeros((1, 4))
            local_angles = torch.zeros((1, 1))
            cls_orients = torch.zeros((1, 1))
            reg_orients = torch.zeros((1, 2))
            h_2ds = torch.zeros((1, 1))
            c_2ds = torch.zeros((1, 1))
            r_2ds = torch.zeros((1, 1))
            cls_orient_4s = torch.zeros((1, 1))
            center_orients = torch.zeros((1, 1))
            # distances = torch.zeros((1, 3))
            # angles_camera = torch.zeros((1, 2))
            d_ys = torch.zeros((1, 1))
            gt_boxes_2d_ground_rect = torch.zeros((1, 4))
            gt_boxes_2d_ground_rect_v2 = torch.zeros((1, 4))
            encoded_side_points = torch.zeros((1, 4))
            encoded_bottom_points = torch.zeros((1, 8))
            keypoint_gt = torch.zeros((1, 2))
            keypoint_gt_weights = torch.ones((1, ))
            used = torch.ones((1, ))

        h, w = transform_sample['img'].shape[-2:]
        training_sample = {}
        training_sample['img'] = transform_sample['img']
        training_sample['im_info'] = img_info
        training_sample['input_size'] = torch.FloatTensor([h, w])
        training_sample['img_name'] = transform_sample['img_name']
        training_sample['im_scale'] = im_scale
        training_sample['gt_labels'] = bbox[:, -1].long()
        training_sample['num'] = num
        training_sample['coords_uncoded'] = coords_uncoded
        training_sample['img_orig'] = transform_sample['img_orig']
        training_sample['points_3d'] = points_3d
        training_sample['dims_2d'] = dims_2d
        training_sample['oritation'] = oritations
        training_sample['local_angle_oritation'] = local_angle_oritations
        training_sample['local_angle'] = local_angles
        training_sample['reg_orient'] = reg_orients
        training_sample['cls_orient'] = cls_orients
        training_sample['h_2d'] = h_2ds
        training_sample['c_2d'] = c_2ds
        training_sample['r_2d'] = r_2ds
        training_sample['cls_orient_4'] = cls_orient_4s
        training_sample['center_orient'] = center_orients
        training_sample['encoded_side_points'] = encoded_side_points
        training_sample['encoded_bottom_points'] = encoded_bottom_points
        training_sample['keypoint_gt'] = keypoint_gt
        training_sample['keypoint_gt_weights'] = keypoint_gt_weights
        training_sample['used'] = used

        training_sample['mean_dims'] = transform_sample['mean_dims']

        # use proj instead of original box
        # training_sample['boxes_2d_proj'] = boxes_2d_proj
        training_sample['gt_boxes_proj'] = boxes_2d_proj
        if self.use_proj_2d:
            training_sample['gt_boxes'] = boxes_2d_proj
        else:
            training_sample['gt_boxes'] = bbox[:, :4]

        # note here it is not truely 3d,just their some projected points in 2d
        training_sample['gt_boxes_3d'] = bbox_3d
        training_sample['coords'] = coords
        training_sample['p2'] = transform_sample['p2']
        training_sample['orig_p2'] = transform_sample['orig_p2']

        # training_sample['angles_camera'] = angles_camera
        # training_sample['distance'] = distances
        training_sample['d_y'] = d_ys

        if self.use_rect_v2:
            training_sample[
                'gt_boxes_ground_2d_rect'] = gt_boxes_2d_ground_rect_v2
        else:
            training_sample[
                'gt_boxes_ground_2d_rect'] = gt_boxes_2d_ground_rect

        return training_sample

    def _decompose_project_matrix(self, p2):
        K = p2[:3, :3]
        KT = p2[:, 3]
        T = numpy.dot(numpy.linalg.inv(K), KT)
        return K, T

    def get_transform_sample(self, index):
        img_file = self.imgs[index]
        img = Image.open(img_file)

        if self.training or not hasattr(self, 'calib_file'):
            calib_file = self.calibs[index]
        else:
            calib_file = self.calib_file

        p2 = self.read_calibration(calib_file).astype(numpy.float32)
        # decompose p2
        K, T = self._decompose_project_matrix(p2)

        if self.training:
            lbl_file = self.labels[index]
            bbox, bbox_3d, lbl, used = self.read_annotation(lbl_file)
            ry = bbox_3d[:, -1]
            height = bbox_3d[:, 4]
            ry[height < 0] = geometry_utils.reverse_angle(ry[height < 0])

            # make sample
            transform_sample = {
                'img': img,
                'bbox': bbox,
                'label': lbl,
                'im_scale': 1.0,
                'img_name': img_file,
                'bbox_3d': bbox_3d,
                'p2': p2,
                'orig_p2': copy.deepcopy(p2),
                'K': K,
                'T': T,
                'used': used
            }
        else:
            # make sample
            transform_sample = {
                'img': img,
                'im_scale': 1.0,
                'img_name': img_file,
                'p2': p2,
                'orig_p2': copy.deepcopy(p2),
                'K': K,
                'T': T
            }
        transform_sample.update({'img_orig': numpy.asarray(img).copy()})

        # get mean dims for encode and decode
        mean_dims = self._get_mean_dims()
        transform_sample['mean_dims'] = mean_dims

        return transform_sample

    def _get_mean_dims(self):
        cls_mean_dims = []
        for cls in self.classes:
            cls_mean_dims.append(MEAN_DIMS[cls][::-1])
        return numpy.asarray(cls_mean_dims)

    def __getitem__(self, index):

        transform_sample = self.get_transform_sample(index)

        if self.transforms is not None:
            transform_sample = self.transforms(transform_sample)

        return self.get_training_sample(transform_sample)

    def read_calibration(self, calib_path):
        return load_projection_matrix(calib_path)

    def check_if_used(self, truncated, occlude):
        truncated = float(truncated)
        occluded = float(occlude)
        if truncated > 0.3 or occluded > 1:
            return 0
        return 1

    def read_annotation(self, file_name):
        """
        read annotation from file
        :param file_name:
        :return:boxes, labels
        boxes: [[xmin, ymin, xmax, ymax], ...]
        """
        boxes = []
        boxes_3d = []
        labels = []
        annos = self.load_annotation(file_name)
        used = []

        for obj in annos:
            obj = obj.split(' ')
            obj[-1] = obj[-1][:-1]
            obj_name = obj[0]

            if not self.is_annotation(obj_name):
                # print obj_name
                continue

            # occluded = int(float(obj[2]))
            # if occluded > 2:
            #     continue
            #
            # truncated = float(obj[1])
            # if truncated > 0.8:
            #     continue

            obj_id = self.encode_obj_name(obj_name)
            xmin = int(float(obj[4]))
            ymin = int(float(obj[5]))
            xmax = int(float(obj[6]))
            ymax = int(float(obj[7]))

            boxes_3d.append(obj[8:])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj_id)
            used.append(self.check_if_used(obj[1], obj[2]))

        boxes = numpy.array(boxes, dtype=float)
        labels = numpy.array(labels, dtype=int)
        boxes_3d = numpy.array(boxes_3d, dtype=float)
        used = numpy.array(used, dtype=numpy.float32)

        return boxes, boxes_3d, labels, used

    @staticmethod
    def load_annotation(file_name):
        with open(file_name) as f:
            bbox = f.readlines()

        return bbox

    def encode_obj_name(self, name):
        OBJ_CLASSES = self.classes
        _id = -1
        for i in range(OBJ_CLASSES.__len__()):
            if name == OBJ_CLASSES[i]:
                # 0 refers to bg
                _id = i + 1
                break
        if _id == -1:
            print("wrong label !")
        return _id

    @staticmethod
    def is_label_file(filename):
        return filename.endswith(".txt")

    def is_annotation(self, _name):
        return any(category == _name for category in self.classes)

    def make_label_list(self, dataset_file):
        train_list_path = os.path.join(dataset_file)
        # train_list_path = './train.txt'
        # train_list_path = './demo.txt'
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            labels = [line.strip() for line in lines]
            labels = [
                os.path.join(self.root_path, 'label_2/{}.txt'.format(label))
                for label in labels
            ]
            if self.training:
                # when testing,do not filter out labels for increase fp
                labels = [
                    label for label in labels if self.__check_has_car(label)
                ]
        return labels

    def make_image_list(self):
        images = []
        for lab in self.labels:
            lab = lab.split('/')[-1]
            lab = lab[:-4]
            img_name = lab + '.png'

            read_path = os.path.join(self.root_path,
                                     'image_2/{}'.format(img_name))
            images.append(read_path)
        return images

    def make_calib_list(self):
        calibs = []
        for lab in self.labels:
            lab = lab.split('/')[-1]
            lab = lab[:-4]
            calib_name = lab + '.txt'

            read_path = os.path.join(self.root_path,
                                     'calib/{}'.format(calib_name))
            calibs.append(read_path)
        return calibs

    def __check_has_car(self, file_path):
        lines = [line.rstrip() for line in open(file_path)]
        objs = [Object3d(line) for line in lines]
        for obj in objs:
            if obj.type in self.classes:
                # if self.check_if_used(obj.truncation, obj.occlusion):
                return True
        return False

    def __test_load_annotation(self, annos):
        is_trusted = True
        obj_count = 0
        for obj in annos:
            obj = obj.split(' ')
            obj_name = obj[0]

            # occluded = int(float(obj[2]))
            # if occluded > 2:
            #     continue
            #
            # truncated = float(obj[1])
            # if truncated > 0.8:
            #     continue

            if self.is_annotation(obj_name):
                obj_count += 1

        if obj_count < 1:
            is_trusted = False

        return is_trusted

    @staticmethod
    def visualize_bbox(img, bbox, lbl):
        img = numpy.array(img, dtype=float)
        img = numpy.around(img)
        img = numpy.clip(img, a_min=0, a_max=255)
        img = img.astype(numpy.uint8)
        for i, box in enumerate(bbox):
            img = cv2.rectangle(
                img, (int(box[0] * 1024), int(box[1] * 512)),
                (int(box[2] * 1024), int(box[3] * 512)),
                color=color_map[lbl[i]],
                thickness=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", img)
        cv2.waitKey(0)
