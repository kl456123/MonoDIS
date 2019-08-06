# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from utils.analysis import read_boxes_from_label

kitti_label_dir = '/data/object/training/label_2'
all_boxes = read_boxes_from_label(kitti_label_dir, use_3d=True)
# img_w = all_boxes[:, 2] - all_boxes[:, 0]
# img_h = all_boxes[:, 3] - all_boxes[:, 1]
# r = img_h / img_w
# X = np.vstack([r, img_h / 10]).transpose()
h = all_boxes[:, 4]
w = all_boxes[:, 5]
l = all_boxes[:, 6]
X = np.vstack([h, w]).transpose()
y_pred = KMeans(n_clusters=9, random_state=9).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
