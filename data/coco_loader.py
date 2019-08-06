import os
import cv2
import numpy

from PIL import Image, ImageOps
from data.data_loader import DetDataLoader


class CocoLoader(DetDataLoader):

    def __init__(self, data_path, label_path, data_encoder, transforms=None, is_padding=True):
        super(CocoLoader, self).__init__()
        from data.pycocotools.coco import COCO
        from cfgs.data_cfgs.coco81_cfg import DataCFG
        self.root = data_path
        self.coco = COCO(label_path)
        self.ids = list(self.coco.imgs.keys())

        self.filter_ids()

        self.transforms = transforms
        self.data_encoder = data_encoder
        self.is_pad = is_padding

        self.obj_names = DataCFG['obj_names']
        self.colormap = DataCFG['colormap']
        self.id_list = DataCFG['id_list']
        self.num_classes = self.id_list.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img, bbox, lbl = self.test_and_load(index)
        if self.is_pad:
            img, bbox = self.pad_img(img, bbox)

        if self.transforms is not None:
            img, bbox, lbl = self.transforms(img, bbox, lbl)
        loc_target, cls_target, iou_target = self.data_encoder.encode(bbox, lbl)

        return img, loc_target, cls_target, iou_target

    def __len__(self):
        return len(self.ids)

    def __is_annotation(self, _id):
        return any(category == _id for category in self.id_list)

    def encode_obj_name(self, name):
        _id = -1
        for i in range(self.num_classes):
            if name == self.id_list[i]:
                _id = i
                break
        if _id == -1:
            print("wrong label !")
        return _id

    def filter_ids(self):
        new_ids = []
        coco = self.coco
        for idx in len(self.ids):
            img_id = self.ids[idx]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            bbox, lbl = self.read_annotation(target)
            x = bbox.shape
            if x.__len__() >= 2:
                new_ids.append(idx)

        self.ids = new_ids
        return new_ids


    def test_and_load(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        bbox, lbl = self.read_annotation(target)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        return img, bbox, lbl

    def __test_load(self):
        coco = self.coco
        img_id = self.ids[35093]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        bbox, lbl = self.read_annotation(target)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        self.draw_bbox(img, bbox, lbl)

    @staticmethod
    def pad_img(img, bbox):
        w, h = img.size
        a = max(w, h)

        if w < a:
            w_pl = int((a - w) / 2)
            w_pr = (a - w) - w_pl
            img = ImageOps.expand(img, border=(w_pl, 0, w_pr, 0), fill=(124, 117, 104))
            bbox[:, 0] += w_pl
            bbox[:, 2] += w_pl

        elif h < a:
            h_pu = int((a - h) / 2)
            h_pd = (a - h) - h_pu
            img = ImageOps.expand(img, border=(0, h_pu, 0, h_pd), fill=(124, 117, 104))
            bbox[:, 1] += h_pu
            bbox[:, 3] += h_pu

        return img, bbox

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

        boxes = numpy.array(boxes, dtype=float)
        labels = numpy.array(labels, dtype=int)

        return boxes, labels

    def draw_bbox(self, img, bbox, lbls):
        img = numpy.array(img, dtype=float)
        img = numpy.around(img)
        img = numpy.clip(img, a_min=0, a_max=255).astype(numpy.uint8)
        for box, lbl in zip(bbox, lbls):
            xmin = int(box[0])
            xmax = int(box[2])
            ymin = int(box[1])
            ymax = int(box[3])

            c = self.colormap[lbl]
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=c, thickness=2)

            class_name = self.obj_names[lbl]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, class_name, (xmin + 5, ymax - 5), font, fontScale=0.5, color=c, thickness=2)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return img


def visualize_data():
    img_path = '/home/mark/Dataset/COCO/train2017'
    lbl_path = '/home/mark/Dataset/COCO/annotations/instances_train2017.json'

    loader = CocoLoader(img_path, lbl_path, data_encoder=None)
    for idx in loader.ids:
        ann_ids = loader.coco.getAnnIds(imgIds=idx)
        target = loader.coco.loadAnns(ann_ids)
        bbox, lbl = loader.read_annotation(target)

        path = loader.coco.loadImgs(idx)[0]['file_name']

        img = Image.open(os.path.join(loader.root, path)).convert('RGB')
        img, bbox = loader.pad_img(img, bbox)
        img = loader.draw_bbox(img, bbox, lbl)
        cv2_img = numpy.asarray(img)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', cv2_img)
        cv2.waitKey(0)


def priorbox_cluster():
    import cv2
    import math
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    img_path = '/home/mark/Dataset/COCO/train2017'
    lbl_path = '/home/mark/Dataset/COCO/annotations/instances_train2017.json'

    obj_area = []
    bbox = []
    loader = CocoLoader(img_path, lbl_path, data_encoder=None)
    for idx in tqdm(loader.ids):
        ann_ids = loader.coco.getAnnIds(imgIds=idx)
        target = loader.coco.loadAnns(ann_ids)
        boxes, labels = loader.read_annotation(target)
        path = loader.coco.loadImgs(idx)[0]['file_name']
        img = cv2.imread(os.path.join(loader.root, path))
        shape = img.shape
        for b in boxes:
            area = math.sqrt((b[2] - b[0]) * (b[3] - b[1]) / shape[0] / shape[1])
            obj_area.append(area)
            box = [1., max((b[2] - b[0]) / (b[3] - b[1] + 1), (b[3] - b[1]) / (b[2] - b[0] + 1))]
            bbox.append(box)

    x = numpy.array(obj_area).reshape((-1, 1))
    x = numpy.clip(x, 0, 1.)

    bbox = numpy.array(bbox)
    bbox = numpy.clip(bbox, 0, 5.)
    kmeans1 = KMeans(n_clusters=6).fit(x)
    kmeans2 = KMeans(n_clusters=2).fit(bbox)
    print("scales:")
    print(kmeans1.cluster_centers_)
    print("ratios:")
    print(kmeans2.cluster_centers_)


if __name__ == '__main__':
    # visualize_data()
    priorbox_cluster()
