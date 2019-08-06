# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import sys


class FeatVisualizer(object):
    def __init__(self, model=None):
        # model for visualization
        self.model = model

    def visualize_maps(self, featmaps):
        """
        visualize all maps tensor in order
        Args:
            featmaps: dict of featname-featmap pair
        """
        imgs = []
        titles = []
        for title in featmaps:
            imgs.extend(self.convert_tensor_to_img(featmaps[title]))
            titles.append(title)

        # loop among mutilple images
        for idx, img in enumerate(imgs):
            self.visualize_map(img, title=titles[idx])

    def visualize_model(self, inputs, model, feat_levels):
        """
        visualize all generated maps during model forward
        """
        if model is None:
            model = self.model
        if model is None:
            raise ValueError("model should be specified")
        x = inputs
        count = 0
        featmaps = []

        for m in self.travel_model(model, False):
            x = m(x)
            if count in feat_levels:
                featmaps.append(x)
                count += 1

        self.visualize_maps(featmaps)

        sys.exit(0)

    def travel_model(self, network, unsequeeze=True):
        for layer in network.children():
            if unsequeeze and type(layer) == nn.Sequential:
                for layer in self.travel_model(layer, unsequeeze):
                    yield layer
            else:
                yield layer

    def convert_tensor_to_img(self, featmap):
        featmap = featmap.detach().cpu().numpy()
        assert len(featmap.shape) == 4
        batch_size = featmap.shape[0]
        imgs = []
        for i in range(batch_size):
            imgs.append(np.transpose(featmap[i], (1, 2, 0)))
        return imgs

    def visualize_map(self, featmap, title='', save=True):
        """
        Args:
        featmap:Tensor(H,W,C)
        """

        if len(featmap.shape) == 3:
            featmap = featmap.sum(axis=-1)

        assert len(featmap.shape) == 2
        plt.imshow(featmap, cmap='gray', aspect='auto')
        plt.title(title)
        plt.axis('off')
        plt.savefig(title + '.png')
        plt.show()
