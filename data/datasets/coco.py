# -*- coding: utf-8 -*-

import numpy as np
import os
from PIL import Image, ImageOps

from data.det_dataset import DetDataset
try:
    from pycocotools.coco import COCO
except:
    pass
import torch

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
           'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
           'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
           'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')


class CocoDataset(DetDataset):
    def __init__(self, dataset_config, transforms=None, training=True):
        super().__init__(training)
        self.root_path = dataset_config['root_path']
        self.data_path = os.path.join(self.root_path,
                                      dataset_config['data_path'])
        self.coco = COCO(
            os.path.join(self.root_path, dataset_config['label_path']))

        self.ids = list(self.coco.imgs.keys())

        self.transforms = transforms

        self.obj_names = dataset_config['classes']
        # self.id_list = dataset_config['id_list']
        self.num_classes = len(self.obj_names)
        self.id_list = self.coco.getCatIds(catNms=self.obj_names)

        self.filter_ids()

    def __is_annotation(self, _id):
        return any(category == _id for category in self.id_list)

    def get_training_sample(self, transform_sample):
        bbox = transform_sample['bbox']
        img = transform_sample['img']

        training_sample = {}
        training_sample['gt_boxes'] = bbox[:, :4]
        training_sample['img'] = img
        training_sample['gt_labels'] = transform_sample['label']

        im_scale = transform_sample['im_scale']
        w = img.size()[2]
        h = img.size()[1]
        img_info = torch.FloatTensor([h, w, im_scale])
        training_sample['im_info'] = img_info

        training_sample['img_name'] = transform_sample['img_name']

        if not self.training:
            training_sample['img_orig'] = transform_sample['img_orig']

        return training_sample

    def __getitem__(self, index):
        img, bbox, lbl, img_name = self.get_transform_sample(index)
        sample = {
            'img': img,
            'bbox': bbox,
            'label': lbl,
            'im_scale': 1.0,
            'img_name': img_name,
            'img_orig': np.asarray(img).copy()
        }
        if self.transforms is not None:
            sample = self.transforms(sample)

        sample = self.get_training_sample(sample)

        return sample

    def __len__(self):
        return len(self.ids)

    def get_transform_sample(self, idx):
        return self.test_and_load(idx)

    def filter_ids(self):
        new_ids = []
        coco = self.coco
        for idx in range(len(self.ids)):
            img_id = self.ids[idx]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            bbox, lbl = self.read_annotation(target)
            x = bbox.shape
            if x.__len__() >= 2:
                new_ids.append(img_id)

        self.ids = new_ids
        self.ids = new_ids[:1]
        return new_ids

    def test_and_load(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        bbox, lbl = self.read_annotation(target)

        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.data_path, path)
        img = Image.open(img_path).convert('RGB')

        return img, bbox, lbl, img_path

    def encode_obj_name(self, name):
        _id = -1
        for i in range(self.num_classes):
            if name == self.id_list[i]:
                _id = i + 1
                break
        if _id == -1:
            print("wrong label !")
        return _id

    def read_annotation(self, targets):
        """
        read annotation from file
        :param targets:
        :return:boxes, labels
        boxes: [[xmin, ymin, xmax, ymax], ...]
        """
        boxes = []
        labels = []
        for obj in targets:
            obj_id = obj['category_id']
            if not self.__is_annotation(obj_id):
                continue
            obj_id = self.encode_obj_name(obj_id)
            box = obj['bbox']
            boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            labels.append(obj_id)

        boxes = np.array(boxes, dtype=float)
        labels = np.array(labels, dtype=int)

        return boxes, labels


def priorbox_cluster():
    import cv2
    import math
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    # img_path = '/home/mark/Dataset/COCO/train2017'
    # lbl_path = '/home/mark/Dataset/COCO/annotations/instances_train2017.json'
    dataset_config = {
        'root_path': '/data/liangxiong/COCO2017',
        'data_path': 'train2017',
        'label_path': 'annotations/instances_train2017.json',
        'classes': ['car']
    }
    loader = CocoDataset(dataset_config)

    obj_area = []
    bbox = []
    # loader = CocoDataset(img_path, lbl_path, data_encoder=None)
    for idx in tqdm(loader.ids):
        ann_ids = loader.coco.getAnnIds(imgIds=idx)
        target = loader.coco.loadAnns(ann_ids)
        boxes, labels = loader.read_annotation(target)
        path = loader.coco.loadImgs(idx)[0]['file_name']
        img = cv2.imread(os.path.join(loader.data_path, path))
        shape = img.shape
        for b in boxes:
            area = math.sqrt(
                (b[2] - b[0]) * (b[3] - b[1]) / shape[0] / shape[1])
            obj_area.append(area)
            box = [
                1.,
                max((b[2] - b[0]) / (b[3] - b[1] + 1),
                    (b[3] - b[1]) / (b[2] - b[0] + 1))
            ]
            bbox.append(box)

    x = np.array(obj_area).reshape((-1, 1))
    x = np.clip(x, 0, 1.)

    bbox = np.array(bbox)
    bbox = np.clip(bbox, 0, 5.)
    kmeans1 = KMeans(n_clusters=6).fit(x)
    kmeans2 = KMeans(n_clusters=2).fit(bbox)
    print("scales:")
    print(kmeans1.cluster_centers_)
    print("ratios:")
    print(kmeans2.cluster_centers_)


if __name__ == '__main__':
    # dataset_config = {
    # 'data_path': '/data/liangxiong/COCO2017/train2017',
    # 'label_path':
    # '/data/liangxiong/COCO2017/annotations/instances_train2017.json',
    # 'classes':
    # ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']
    # }
    # dataset = CocoDataset(dataset_config)
    # sample = dataset[0]

    priorbox_cluster()
