# -*- coding: utf-8 -*-

import os
import torch
import shutil
import logging
from core.utils.model_serialization import load_state_dict


class Saver():
    def __init__(self, checkpoint_dir, logger=None):
        self.checkpoint_dir = checkpoint_dir
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def get_checkpoint_path(self, checkpoint_name):
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        return checkpoint_path

    def load_pretrained_model(self, model, checkpoint_name):
        # import ipdb
        # ipdb.set_trace()
        checkpoint_path = self.get_checkpoint_path(checkpoint_name)
        self.logger.info(("loading checkpoint %s" % (checkpoint_path)))
        state_dict = torch.load(checkpoint_path)
        load_state_dict(model, state_dict)
        self.logger.info(("loaded checkpoint %s" % (checkpoint_name)))

    def load(self, params_dict, checkpoint_name):
        checkpoint_path = self.get_checkpoint_path(checkpoint_name)
        self.logger.info(("loading checkpoint %s" % (checkpoint_path)))

        checkpoint = torch.load(checkpoint_path)
        for name, module in list(params_dict.items()):
            if name in checkpoint:
                if isinstance(module, torch.nn.parallel.DataParallel):
                    # module.module.load_state_dict(checkpoint[name])
                    module = module.module
                if hasattr(module, 'load_state_dict'):
                    if name == 'model':
                        # load model
                        module_dict = module.state_dict()

                        checkpoint_dict = {}
                        for k, v in checkpoint[name].items():
                            if k in module_dict:
                                if module_dict[k].shape != v.shape:
                                    self.logger.warning(
                                        'size mismatch for {} shape({}/{}), ignore it by default!'.
                                        format(k, module_dict[k].shape,
                                               v.shape))
                                else:
                                    checkpoint_dict[k] = v

                        module_dict.update(checkpoint_dict)
                        module.load_state_dict(module_dict)
                    else:
                        # load optimizer or scheduler
                        module.load_state_dict(checkpoint[name])
                else:
                    params_dict[name] = checkpoint[name]
            else:
                self.logger.warning(
                    ('module:{} can not be loaded'.format(name)))

        self.logger.info(("loaded checkpoint %s" % (checkpoint_name)))

    def save(self, params_dict, checkpoint_name, is_best=False):
        checkpoint_path = self.get_checkpoint_path(checkpoint_name)
        state = {}
        for name, module in list(params_dict.items()):
            if isinstance(module, torch.nn.parallel.DataParallel):
                state[name] = module.module.state_dict()
            elif hasattr(module, 'state_dict'):
                state[name] = module.state_dict()
            else:
                state[name] = module
        self.save_checkpoint(state, is_best, checkpoint_path)
        self.logger.info(('save model: {}'.format(checkpoint_name)))

    @staticmethod
    def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')
