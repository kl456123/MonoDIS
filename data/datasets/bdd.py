# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np

from PIL import Image
from data.det_dataset import DetDataset


class BDDDataset(DetDataset):
    def __init__(self, dataset_config, transforms=None, training=True):
        super().__init__(training)
        # import ipdb
        # ipdb.set_trace()
        self.transforms = transforms
        self.root_path = dataset_config['root_path']
        self.data_path = os.path.join(self.root_path,
                                      dataset_config['data_path'])
        self.label_path = os.path.join(self.root_path,
                                       dataset_config['label_path'])

        self.classes = ['bg'] + dataset_config['classes']
        self.labels = self.make_label_list(
            os.path.join(self.label_path, dataset_config['dataset_file']))
        self.imgs = self.make_image_list()

    def _check_class(self, label):
        return label in self.classes

    def _check_anno(self, anno):
        labels = anno['labels']
        use = False
        for label in labels:
            if self._check_class(label['category']):
                use = True
        return use

    def make_label_list(self, dataset_file):
        annotations = self.load_annotation(dataset_file)
        new_annotations = []
        for anno in annotations:
            if self._check_anno(anno):
                new_annotations.append(anno)

        return new_annotations

    def make_image_list(self):
        imgs = []
        for anno in self.labels:
            imgs.append(os.path.join(self.data_path, anno['name']))
        return imgs

    def __getitem__(self, index):

        transform_sample = self.get_transform_sample(index)

        if self.transforms is not None:
            transform_sample = self.transforms(transform_sample)

        return self.get_training_sample(transform_sample)

    @staticmethod
    def load_annotation(file_name):
        with open(file_name) as f:
            anno = json.load(f)
        return anno

    def encode_obj_name(self, obj_name):
        return self.classes.index(obj_name)

    def read_annotation(self, anno):
        bboxes = []
        labels = []

        for label in anno['labels']:
            if label == 1:
                continue
            category = label['category']
            if self._check_class(category):
                labels.append(self.encode_obj_name(category))
                box2d = label['box2d']
                keys = ['x1', 'y1', 'x2', 'y2']
                bboxes.append([box2d[key] for key in keys])

        labels = np.asarray(labels, dtype=np.int)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        return bboxes, labels

    def get_transform_sample(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        transform_sample = {}
        bbox, label = self.read_annotation(self.labels[index])
        transform_sample = {
            'img': img,
            'bbox': bbox,
            'label': label,
            'im_scale': 1.0,
            'img_name': img_path
        }
        transform_sample.update({'img_orig': np.asarray(img).copy()})
        return transform_sample

    def get_training_sample(self, transform_sample):
        img = transform_sample['img']
        img = np.asarray(img)
        im_scale = transform_sample['im_scale']
        w = img.shape[2]
        h = img.shape[1]
        if type(im_scale) is float:
            img_info = np.asarray([h, w, im_scale], dtype=np.float32)
        else:
            img_info = np.asarray([h, w, *im_scale], dtype=np.float32)

        training_sample = {}
        training_sample['img'] = img
        training_sample['im_info'] = img_info
        training_sample['input_size'] = [h, w]
        training_sample['img_name'] = transform_sample['img_name']
        training_sample['im_scale'] = im_scale
        training_sample['gt_labels'] = transform_sample['label'].long()

        training_sample['gt_boxes'] = transform_sample['bbox']
        training_sample['img_orig'] = transform_sample['img_orig']

        return training_sample

    @staticmethod
    def visualize_bbox(img, bbox, lbl):
        img = np.array(img, dtype=float)
        img = np.around(img)
        img = np.clip(img, a_min=0, a_max=255)
        img = img.astype(np.uint8)
        for i, box in enumerate(bbox):
            img = cv2.rectangle(
                img, (int(box[0] ), int(box[1])),
                (int(box[2]), int(box[3] )),
                color=(55, 255, 155),
                thickness=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    dataset_config = {
        'root_path': '/data/bdd/bdd100k/',
        'dataset_file': 'bdd100k_labels_images_val.json',
        'data_path': 'images/100k/val',
        'label_path': 'labels',
        'classes': ['car', 'person', 'bus']
    }
    dataset = BDDDataset(dataset_config, training=False)
    for sample in dataset:
        img = sample['img']
        bbox = sample['gt_boxes']
        dataset.visualize_bbox(img, bbox, None)
