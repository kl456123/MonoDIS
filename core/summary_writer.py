# -*- coding: utf-8 -*-

from tensorboardX import SummaryWriter as SW


class SummaryWriter(SW):
    def add_scalar_dict(self, loss_dict, step):
        for key, val in loss_dict.items():
            self.add_scalar(key, val.mean().item(), step)
