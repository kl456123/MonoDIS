# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


class DetDataset(Dataset, metaclass=ABCMeta):
    """
    The important thing is that data and label should be separated
    so that it can adapt to both training and testing mode
    """

    def __init__(self, training):
        self.imgs = None
        self.scale = None
        self.is_gray = None
        self.data_path = None
        self.transforms = None
        self.num_classes = None
        self.training = training

    def __len__(self):
        return len(self.imgs)

    @abstractmethod
    def __getitem__(self, item_idx):
        """
        return data and label directly
        """
        pass

    @staticmethod
    def is_image_file(filename):
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    @abstractmethod
    def get_training_sample(self, transform_sample):
        pass

    @abstractmethod
    def get_transform_sample(self, idx):
        pass
