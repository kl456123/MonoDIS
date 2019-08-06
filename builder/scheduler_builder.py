# -*- coding: utf-8 -*-

import torch


class SchedulerBuilder(object):
    def __init__(self, optimizer, scheduler_config):
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config

    def build(self):
        config = self.scheduler_config

        scheduler_type = config['type']
        # schedule
        if scheduler_type == 'step':
            lr_decay_step = config['lr_decay_step']
            lr_decay_gamma = config['lr_decay_gamma']
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                lr_decay_step,
                lr_decay_gamma,
                last_epoch=config['last_step'])
        else:
            raise ValueError('this type of scheduler is unknown')
        return scheduler
