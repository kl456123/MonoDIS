# -*- coding: utf-8 -*-

import torch


class Profiler(object):
    def __init__(self):
        self._start_time = {}
        self._end_time = {}
        self.duration = {}
        self.loop_nums = {}
        self._enable = False

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def start(self, name):
        if self._enable:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self._start_time[name] = start
            self._end_time[name] = end
            start.record()

    def end(self, name):
        if self._enable:
            end = self._end_time[name]
            end.record()
            torch.cuda.synchronize()

            start = self._start_time[name]
            duration = start.elapsed_time(end)

            if self.duration.get(name) is None:
                self.duration[name] = duration
            else:
                self.duration[name] += duration
        else:
            self.duration[name] = 0

        if self.loop_nums.get(name) is None:
            self.loop_nums[name] = 1
        else:
            self.loop_nums[name] += 1

    def clear(self):
        self.duration = {}
        self._start_time = {}
        self._end_time = {}
        self.loop_nums = {}
