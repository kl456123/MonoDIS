# -*- coding: utf-8 -*-

import os
from shutil import copyfile

results_dir = './results/data'
label_dir = '/data/object/training/label_2'

for fn in os.listdir(results_dir):
    src = os.path.join(label_dir, fn)
    dst = os.path.join(results_dir, fn)
    copyfile(src, dst)


def add_scores():
    for fn in os.listdir(results_dir):
        fpath = os.path.join(results_dir, fn)
        with open(fpath, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() + ' 1.0' for line in lines]
            res = '\n'.join(lines)

        with open(fpath, 'w') as f:
            f.write(res)


add_scores()
