# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

pr_file = './car_detection.txt'
pr = np.ndfromtxt(pr_file)

# easy moderate hard
legends = ['easy', 'moderate', 'hard']
for i in range(3):
    plt.plot(pr[:, 0], pr[:, i + 1])

plt.show()
