# -*- coding: utf-8 -*-
import os

hw_dir = '/data/hw'
hw_img_dir = os.path.join(hw_dir, 'image_2')

datafile = './hw.txt'

sample_names = []
for file in sorted(os.listdir(hw_img_dir)):
    sample_name = os.path.splitext(file)[0]
    sample_names.append(sample_name)

with open(datafile, 'w') as f:
    f.write('\n'.join(sample_names))
