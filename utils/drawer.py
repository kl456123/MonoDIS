# -*- coding: utf-8 -*-
import os
import logging
import copy
import cv2
import sys
import time
import shutil
from utils import geometry_utils
from wavedata.tools.obj_detection.obj_utils import ObjectLabel
from core.utils.logger import setup_logger

from utils.box_vis import draw_line
import numpy as np
from PIL import Image, ImageDraw
import random

# image_path = '/data/object/training/image_2/000052.png'
# a = np.asarray([[[740.2719, 243.6114], [711.2286, 230.7807]], [
# [625.7795, 239.5613], [631.7291, 227.8167]
# ], [[321.1204, 217.5381],
# [406.7209, 217.0684]], [[69.1882, 209.3627], [74.0953, 212.0611]],
# [[310.9117, 248.0898], [378.7721, 260.0074]]])
# draw_line(image_path, a)


class Drawer(object):
    pass


class ImageVisualizer(object):
    def __init__(self,
                 image_dir,
                 result_dir,
                 label_dir=None,
                 calib_dir=None,
                 calib_file=None,
                 online=False,
                 save_dir=None):

        # path config
        self.image_dir = image_dir
        self.result_dir = result_dir
        self.label_dir = label_dir

        # used for visualize 3d bbox, if not, it can be None
        # all image use the same calib or each one is different
        self.calib_dir = calib_dir
        self.calib_file = calib_file

        # unknown now, add class to the list later
        # and each class is assigned with a unique color generated randomly
        self.classes = []
        self.colors = []

        self.side_plane_colors = []
        # for i in range(6):
        # self.side_plane_colors.append(self.get_random_color())
        self.front_side_color = (255, 0, 255)

        # display online or just save it
        self.online = online

        if not online:
            assert save_dir is not None, \
                'save_dir should be specified when not in online mode'
        self.save_dir = save_dir

        self.logger = setup_logger()

        # image config
        self.max_size = 1280
        self.min_size = 384

        # stats
        self.start_ind = 0

    @staticmethod
    def get_num_file_in_dir(result_dir):
        num = 0
        for file in os.listdir(result_dir):
            num += 1
        return num

    def get_image_suffix(self):
        for image in os.listdir(self.image_dir):
            return image[-4:]

    def get_image_path(self, sample_name):
        suffix = self.get_image_suffix()
        return os.path.join(self.image_dir, '{}{}'.format(sample_name, suffix))

    def get_results_path(self, sample_name):
        if self.result_dir is None:
            return None
        return os.path.join(self.result_dir, '{}.txt'.format(sample_name))

    def get_label_path(self, sample_name):
        if self.label_dir is None:
            return None
        return os.path.join(self.label_dir, '{}.txt'.format(sample_name))

    def get_calib_path(self, sample_name):
        if self.calib_file is not None:
            return self.calib_file
        elif self.calib_dir is not None:
            return os.path.join(self.calib_dir, '{}.txt'.format(sample_name))
        else:
            # self.logger.warn(
            # 'calib file or calib dir should not be None at the same time, disable 3d display'
            # )
            return None

    def visualize_images(self, total_result_files):
        # total_num = Image2DVisualizer.get_num_file_in_dir(self.result_dir)
        total_num = len(total_result_files)
        self.logger.info('{} results in total !'.format(total_num))
        for ind, sample_name in enumerate(total_result_files, self.start_ind):
            image_path = self.get_image_path(sample_name)
            results_path = self.get_results_path(sample_name)
            label_path = self.get_label_path(sample_name)
            calib_path = self.get_calib_path(sample_name)

            start_time = time.time()
            self.visualize_single_image(image_path, results_path, label_path,
                                        calib_path)
            dura_time = time.time() - start_time
            sys.stdout.write('\r{}/{}/{}'.format(ind + 1, total_num,
                                                 dura_time))
            sys.stdout.flush()

    def get_sample_name_from_path(self, path):
        return os.path.basename(path)[:-4]

    def get_saved_path(self, sample_name):
        # use jpg to save storage
        return os.path.join(self.save_dir, '{}.jpg'.format(sample_name))

    def read_labels(self, label_dir, sample_name, results=False):
        """Reads in label data file from Kitti Dataset.

        Returns:
        obj_list -- List of instances of class ObjectLabel.

        Keyword arguments:
        label_dir -- directory of the label files
        img_idx -- index of the image
        """

        # Define the object list
        obj_list = []

        # Extract the list
        if os.stat(label_dir + "/{}.txt".format(sample_name)).st_size == 0:
            return

        if results:
            p = np.loadtxt(
                label_dir + "/{}.txt".format(sample_name),
                delimiter=' ',
                dtype=str,
                usecols=np.arange(start=0, step=1, stop=16))
        else:
            p = np.loadtxt(
                label_dir + "/{}.txt".format(sample_name),
                delimiter=' ',
                dtype=str,
                usecols=np.arange(start=0, step=1, stop=15))

        # Check if the output is single dimensional or multi dimensional
        if len(p.shape) > 1:
            label_num = p.shape[0]
        else:
            label_num = 1

        for idx in np.arange(label_num):
            obj = ObjectLabel()

            if label_num > 1:
                # Fill in the object list
                obj.type = p[idx, 0]
                obj.truncation = float(p[idx, 1])
                obj.occlusion = float(p[idx, 2])
                obj.alpha = float(p[idx, 3])
                obj.x1 = float(p[idx, 4])
                obj.y1 = float(p[idx, 5])
                obj.x2 = float(p[idx, 6])
                obj.y2 = float(p[idx, 7])
                obj.h = float(p[idx, 8])
                obj.w = float(p[idx, 9])
                obj.l = float(p[idx, 10])
                obj.t = (float(p[idx, 11]), float(p[idx, 12]), float(
                    p[idx, 13]))
                obj.ry = float(p[idx, 14])
                if results:
                    obj.score = float(p[idx, 15])
                else:
                    obj.score = 0.0
            else:
                # Fill in the object list
                obj.type = p[0]
                obj.truncation = float(p[1])
                obj.occlusion = float(p[2])
                obj.alpha = float(p[3])
                obj.x1 = float(p[4])
                obj.y1 = float(p[5])
                obj.x2 = float(p[6])
                obj.y2 = float(p[7])
                obj.h = float(p[8])
                obj.w = float(p[9])
                obj.l = float(p[10])
                obj.t = (float(p[11]), float(p[12]), float(p[13]))
                obj.ry = float(p[14])
                if results:
                    obj.score = float(p[15])
                else:
                    obj.score = 0.0

            obj_list.append(obj)

        return obj_list

    def _obj_label_to_box_3d(self, obj_label):
        """
          box_3d format: (location, dims , ry, score)
        """
        box_3d = np.zeros(8)
        box_3d[3:6] = [obj_label.h, obj_label.w, obj_label.l]
        box_3d[:3] = obj_label.t
        box_3d[6] = obj_label.ry
        box_3d[7] = obj_label.score
        return box_3d

    def _obj_label_to_box_2d(self, obj_label):
        """
            2d format: [x1y1x2y1, score]
        """
        box_2d = np.zeros(5)
        box_2d = [
            obj_label.x1, obj_label.y1, obj_label.x2, obj_label.y2,
            obj_label.score
        ]
        return box_2d

    def _class_str_to_index(self, class_type):
        if class_type not in self.classes:
            self.classes.append(class_type)
            self.colors.append(self.get_random_color())

        return self.classes.index(class_type)

    def get_random_color(self):
        color_code = []
        for _ in range(3):
            color_code.append(random.randint(0, 255))
        return color_code

    def load_projection_matrix(self, calib_file):
        """Load the camera project matrix."""
        assert os.path.isfile(calib_file)
        with open(calib_file) as f:
            lines = f.readlines()
            line = lines[2]
            line = line.split()
            assert line[0] == 'P2:'
            p = [float(x) for x in line[1:]]
            p = np.array(p).reshape(3, 4)
        return p

    def check_if_use_3d(self, obj_labels):
        """
            Note check the first one is enough
        """
        if obj_labels is None:
            # no any valid results
            return False
        return obj_labels[0].t[0] != -1000

    def check_2d(self, label_boxes_2d):
        """
            boxes_2d should be in the image
        """
        pass

    def check_3d(self, label_boxes_3d):
        dim = label_boxes_3d[:, 3:6]
        if not (dim > 0).all():
            self.logger.warning('dim should be positive!')

    def parse_kitti_format(self,
                           results_dir,
                           results_path,
                           calib_path=None,
                           results=False):
        """
            Note that calib_dir is shared between gt and det results, directly use self.calib_dir
        """
        sample_name = self.get_sample_name_from_path(results_path)
        obj_labels = self.read_labels(results_dir, sample_name, results)
        if obj_labels is None:
            return None, None, None, None
            # return np.zeros((0, 8)), np.zeros((0, 5)), np.zeros(
        # (0, 1)), np.zeros((3, 4))

        label_boxes_2d = np.asarray(
            [self._obj_label_to_box_2d(obj_label) for obj_label in obj_labels])
        self.check_2d(label_boxes_2d)

        label_classes = [
            self._class_str_to_index(obj_label.type)
            for obj_label in obj_labels
        ]

        if not self.check_if_use_3d(obj_labels):
            # dont use 3d results
            label_boxes_3d = None
            stereo_calib_p2 = None
        else:
            # import ipdb
            # ipdb.set_trace()
            stereo_calib_p2 = self.load_projection_matrix(calib_path)
            label_boxes_3d = np.asarray([
                self._obj_label_to_box_3d(obj_label)
                for obj_label in obj_labels
            ])
            # used for checking 3d is valid (dim should be positive)
            self.check_3d(label_boxes_3d)

        return label_boxes_3d, label_boxes_2d, label_classes, stereo_calib_p2

    def parse_image(self, image_path, format='cv2'):
        if format == 'cv2':
            return cv2.imread(image_path)
        elif format == 'pil':
            return Image.open(image_path)
        else:
            raise ValueError('unknown image parsed method !')

    def render_image_2d(self, image, boxes_2d, label_classes):
        for i, box in enumerate(boxes_2d):
            class_name = self.classes[label_classes[i]]
            color = self.colors[label_classes[i]]
            image = cv2.rectangle(
                image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                color=color,
                thickness=2)

            text = '{} {:.3f}'.format(class_name, box[4])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            rectangle_bgr = (0, 0, 0)
            (text_width, text_height) = cv2.getTextSize(
                text, font, fontScale=font_scale, thickness=1)[0]
            box_coords = ((int(box[0]),
                           int(box[1])), (int(box[0]) + text_width - 2,
                                          int(box[1]) - text_height - 2))
            cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr,
                          cv2.FILLED)
            cv2.putText(
                image,
                text, (int(box[0]), int(box[1])),
                fontFace=font,
                fontScale=font_scale,
                color=(255, 255, 255))

        return image

    def clear(self):
        # for file in os.listdir(self.save_dir):
        self.logger.info('remove dir: {}'.format(self.save_dir))
        shutil.rmtree(self.save_dir)
        self.logger.info('make new dir: {}'.format(self.save_dir))
        os.makedirs(self.save_dir)

    def resume(self):
        if self.result_dir is None:
            result_dir = self.label_dir
        else:
            result_dir = self.result_dir
        # remove suffix
        total_result_files = sorted(os.listdir(result_dir))
        saved_result_files = sorted(os.listdir(self.save_dir))
        total_result_files = [file[:-4] for file in total_result_files]
        saved_result_files = [file[:-4] for file in saved_result_files]
        used_result_files = []

        for file in total_result_files:
            if file not in saved_result_files:
                used_result_files.append(file)
        self.start_ind = len(saved_result_files)
        self.visualize_images(used_result_files)

    def restart(self):
        self.clear()
        self.resume()

    def render_image_3d(self, image, boxes_3d, label_classes, p2):
        connected_points = [[2, 4, 5], [1, 3, 6], [2, 4, 7], [1, 3, 8],
                            [1, 6, 8], [2, 5, 7], [3, 6, 8], [4, 5, 7]]
        connected_points = np.array(connected_points) - 1
        connected_points = connected_points.tolist()
        # connected_points_2d = [[1, 3], [0, 2], [1, 3], [0, 2]]

        # parse calib first
        corners_3d = geometry_utils.boxes_3d_to_corners_3d(boxes_3d[:, :-1])
        corners_2d = geometry_utils.points_3d_to_points_2d(
            corners_3d.reshape((-1, 3)), p2).reshape(-1, 8, 2)

        # bev image
        voxel_size = 0.05
        width = 80
        height = 75
        bev_width = int(height / voxel_size)
        bev_height = int(width / voxel_size)
        connected_points_2d = [[1, 3], [0, 2], [1, 3], [0, 2]]
        bev_image = np.zeros((bev_height, bev_width, 3), np.uint8)
        bev_image[...] = 255

        corners_2d = corners_2d.astype(np.int32).tolist()
        for i in range(boxes_3d.shape[0]):
            color = self.colors[label_classes[i]]
            corners_image = corners_2d[i]

            corners_bird = corners_3d[i][:4, [0, 2]]
            corners_bird = corners_bird[:, ::-1]
            corners_bird[:, 1] = corners_bird[:, 1] + 1 / 2 * width
            corners_bird = (corners_bird / voxel_size).astype(np.int)

            # render box in bev view
            for i in range(4):
                for j in range(4):
                    if j in connected_points_2d[i]:
                        start_point = (corners_bird[i][0], corners_bird[i][1])
                        end_point = (corners_bird[j][0], corners_bird[j][1])
                        cv2.line(bev_image, start_point, end_point, color, 2)

            # render box in image
            for i in range(8):
                for j in range(8):
                    if j in connected_points[i]:
                        start_point = (corners_image[i][0],
                                       corners_image[i][1])
                        end_point = (corners_image[j][0], corners_image[j][1])
                        cv2.line(image, start_point, end_point, color, 2)

        # rotate bev_image
        bev_image = cv2.rotate(bev_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image, bev_image

    def get_image_final_size(self, image):
        origin_shape = image.shape[:-1]
        max_size = max(*origin_shape)
        scale = self.max_size / max_size
        return scale

    def get_bev_final_size(self, final_image, bev_image):
        final_image_height = final_image.shape[0]
        bev_height, bev_width = bev_image.shape[:2]
        scale = float(final_image_height) / float(bev_height)
        final_bev_shape = [final_image_height, 800]
        return final_bev_shape

    def render_image(self, image, results):
        boxes_3d, boxes_2d, label_classes, p2 = results
        if boxes_3d is not None:
            image_3d = copy.deepcopy(image)
            image_3d, bev_image = self.render_image_3d(image_3d, boxes_3d,
                                                       label_classes, p2)

        if boxes_2d is not None:
            image_2d = copy.deepcopy(image)
            image_2d = self.render_image_2d(image_2d, boxes_2d, label_classes)

        # vertical stack two images
        if boxes_3d is not None:
            image = np.concatenate((image_2d, image_3d), axis=0)

            # resize bev_image
            bev_shape = self.get_bev_final_size(image, bev_image)
            bev_image = cv2.resize(
                bev_image,
                tuple(bev_shape[::-1]),
                interpolation=cv2.INTER_CUBIC)

            # stack with bev image
            image = np.concatenate([image, bev_image], axis=1)
        elif boxes_2d is not None:
            image = image_2d

        scale = self.get_image_final_size(image)
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return image

    def visualize_single_image(self,
                               image_path,
                               results_path=None,
                               label_path=None,
                               calib_path=None):

        image = self.parse_image(image_path)

        # image preprocess
        image = np.around(np.array(image, dtype=float))
        image = np.clip(image, a_min=0, a_max=255).astype(np.uint8)

        # draw results bboxes
        if results_path is not None:
            results = self.parse_kitti_format(self.result_dir, results_path,
                                              calib_path, True)
            image = self.render_image(image, results)

        # draw label bboxes
        if label_path is not None:
            labels = self.parse_kitti_format(self.label_dir, label_path,
                                             calib_path, False)
            image = self.render_image(image, labels)

        if self.online:
            # image postprocess
            cv2.imshow("test", image)
            cv2.waitKey(0)
        else:
            sample_name = self.get_sample_name_from_path(image_path)
            saved_path = self.get_saved_path(sample_name)
            cv2.imwrite(saved_path, image)

    def render_image_corners_2d(self,
                                image_path,
                                image=None,
                                corners_2d=None,
                                boxes_2d=None,
                                corners_3d=None,
                                p2=None):
        color = (255, 255, 0)

        if corners_2d is None:
            assert corners_3d is not None
            corners_2d = geometry_utils.points_3d_to_points_2d(
                corners_3d.reshape(-1, 3), p2).reshape(-1, 8, 2)
        num_boxes = corners_2d.shape[0]
        corners_2d = corners_2d.astype(np.int32).tolist()

        if image is None:
            image = self.parse_image(image_path)
        connected_points = [[2, 4, 5], [1, 3, 6], [2, 4, 7], [1, 3, 8],
                            [1, 6, 8], [2, 5, 7], [3, 6, 8], [4, 5, 7]]
        connected_plane = [[0, 1, 5, 4], [0, 4, 7, 3], [3, 7, 6, 2],
                           [2, 6, 5, 1], [4, 5, 6, 7], [0, 1, 2, 3]]
        connected_points = np.array(connected_points) - 1
        front_side_line = [0, 1, 4, 5]
        connected_points = connected_points.tolist()
        # connected_points_2d = [[1, 3], [0, 2], [1, 3], [0, 2]]

        # parse calib first
        # corners_3d = geometry_utils.boxes_3d_to_corners_3d(boxes_3d[:, :-1])
        # corners_2d = geometry_utils.points_3d_to_points_2d(
        # corners_3d.reshape((-1, 3)), p2).reshape(-1, 8, 2)

        # bev image
        if corners_3d is not None:
            # bev image
            voxel_size = 0.05
            width = 80
            height = 75
            bev_width = int(height / voxel_size)
            bev_height = int(width / voxel_size)
            connected_points_2d = [[1, 3], [0, 2], [1, 3], [0, 2]]
            bev_image = np.zeros((bev_height, bev_width, 3), np.uint8)
            bev_image[...] = 255

            for i in range(corners_3d.shape[0]):
                corners_bird = corners_3d[i][:4, [0, 2]]
                corners_bird = corners_bird[:, ::-1]
                corners_bird[:, 1] = corners_bird[:, 1] + 1 / 2 * width
                corners_bird = (corners_bird / voxel_size).astype(np.int)

                # render box in bev view
                for i in range(4):
                    for j in range(4):
                        if j in connected_points_2d[i]:
                            start_point = (corners_bird[i][0],
                                           corners_bird[i][1])
                            end_point = (corners_bird[j][0],
                                         corners_bird[j][1])
                            cv2.line(bev_image, start_point, end_point, color,
                                     2)
            bev_image = cv2.rotate(bev_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        image_corners = copy.deepcopy(image)

        for i in range(num_boxes):
            corners_image = corners_2d[i]

            # render box in image
            for i in range(8):
                for j in range(8):
                    if j in connected_points[i]:
                        if i in front_side_line and j in front_side_line:
                            color = self.front_side_color
                        else:
                            color = (255, 255, 0)
                        start_point = (corners_image[i][0],
                                       corners_image[i][1])
                        end_point = (corners_image[j][0], corners_image[j][1])
                        cv2.line(image_corners, start_point, end_point, color,
                                 2)

            # for i in range(6):
            # corners_image = np.asarray(corners_image)
            # side_plane = np.asarray(corners_image[connected_plane[i]])
            # image_corners = ImageVisualizer.fill(image_corners, side_plane,
            # self.side_plane_colors[i])
        # rotate bev_image
        # bev_image = cv2.rotate(bev_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if boxes_2d is not None:
            image_2d = copy.deepcopy(image)
            for i, box in enumerate(boxes_2d):
                # class_name = self.classes[label_classes[i]]
                # color = self.colors[label_classes[i]]
                image = cv2.rectangle(
                    image_2d, (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color=color,
                    thickness=2)

            image = np.concatenate((image_2d, image_corners), axis=0)
        else:
            image = image_corners

        if corners_3d is not None:
            # resize bev_image
            bev_shape = self.get_bev_final_size(image, bev_image)
            bev_image = cv2.resize(
                bev_image,
                tuple(bev_shape[::-1]),
                interpolation=cv2.INTER_CUBIC)
            # stack with bev image
            image = np.concatenate([image, bev_image], axis=1)

        scale = self.get_image_final_size(image)
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if self.online:
            # image postprocess
            cv2.imshow("test", image)
            cv2.waitKey(0)
        else:
            sample_name = self.get_sample_name_from_path(image_path)
            saved_path = self.get_saved_path(sample_name)
            cv2.imwrite(saved_path, image)

    def render_image_and_depthmap(self, image_path, boxes_2d, depth_map, p2):
        pass

    @staticmethod
    def fill(img, points, color):
        filler = cv2.convexHull(points)
        cv2.fillConvexPoly(img, filler, color)

        return img


if __name__ == '__main__':
    # image_dir = '/data/object/training/image_2'
    # label_dir = '/data/object/training/label_2'
    # calib_dir = '/data/object/training/calib'
    # result_dir = None

    # NUSCENES
    #  image_dir = '/data/nuscenes/samples/CAM_FRONT'
    #  result_dir = './results/data'
    #  save_dir = 'results/images'
    #  calib_dir = '/data/nuscenes/calibs'
    #  calib_file = None
    #  label_dir = None

    # KITTI
    image_dir = '/data/object/training/image_2'
    result_dir = './results/data'
    save_dir = 'results/images'
    calib_dir = '/data/object/training/calib'
    label_dir = None
    #  label_dir = '/data/object/training/label_2'
    calib_file = None

    # MONOGRNET
    # image_dir = '/data/object/training/image_2'
    # result_dir = './detections'
    # save_dir = 'results/images'
    # calib_dir = '/data/object/training/calib'
    # label_dir = None
    # calib_file = None

    # BDD
    # image_dir = '/data/bdd/bdd100k/images/100k/val'
    # result_dir = 'results/data'
    # save_dir = 'results/images'
    # calib_dir = None
    # label_dir = None
    # calib_file = None

    # DM
    # image_dir = '/data/dm202_3w/left_img'
    # calib_file = './000004.txt'
    # calib_dir = None
    # result_dir = './results/data'
    # save_dir = 'results/images'
    # label_dir = None

    visualizer = ImageVisualizer(
        image_dir,
        result_dir,
        label_dir=label_dir,
        calib_dir=calib_dir,
        calib_file=calib_file,
        online=False,
        save_dir=save_dir)
    # visualizer.visualize_all_images()
    # visualizer.restart()
    visualizer.resume()
