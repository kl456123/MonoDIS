# -*- coding: utf-8 -*-

import os
import numpy as np
from utils.orient_eval import read_labels

dets_dir = './results/data'


def main():
    for dets_file in os.listdir(dets_dir):
        sample_name = os.path.splitext(dets_file)[0]
        box_2ds_det, box_3ds_det = read_labels(dets_dir, sample_name)
        dims = box_3ds_det[:, :3]
        neg = dims[dims <= 0]
        if neg.size:
            print(sample_name)
            #  import ipdb
            #  ipdb.set_trace()


if __name__ == '__main__':
    main()
