# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_table('./pr_0.7.txt', delim_whitespace=True)

tps = data['tp']
fps = data['fp']
fns = data['fn']


def calculate_ap(tps, fps, display=False):
    ap = 0
    count = 0
    for tp, fp in zip(tps, fps):
        if (count % 4 == 0):
            ap += 1.0 * tp / (tp + fp)
            if display:
                print("count: {} ap: {}".format(count, ap))
        count += 1
    print(ap / 11 * 100)


calculate_ap(tps, fps, True)
