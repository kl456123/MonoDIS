# -*- coding: utf-8 -*-

from utils.visualize import vis_featmap
import numpy as np

np.random.seed(0)
uniform_data = np.random.rand(10, 12, 3)
vis_featmap(uniform_data)
