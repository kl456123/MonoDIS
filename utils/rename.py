# -*- coding: utf-8 -*-

import os

env = './env'
for dirpath, dirnames, filenames in os.walk(env):
    for file in filenames:
        if os.path.splitext(file)[-1] == '.bak':
            abspath = os.path.join(dirpath, file)
            os.rename(abspath, abspath[:-4])
