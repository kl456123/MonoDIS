# -*- coding: utf-8 -*-

# import numpy as np
import torch
# import sys
# sys.path.append('.')
from core.similarity_calc.center_similarity_calc import CenterSimilarityCalc

center_similarity_calc = CenterSimilarityCalc()

bboxes = torch.tensor([539.83, 169.37, 571.59, 198.25]).view(1, -1)
gt_boxes = torch.tensor([540.224, 171.856, 577.278, 203.155]).view(1, -1)
overlaps = center_similarity_calc.compare(bboxes, gt_boxes)
print(overlaps)

