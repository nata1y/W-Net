import numpy as np
import torch
import torch.nn as nn


class Module(nn.Module):

    def __init__(self, in_channels, out_channels, n_features, id):
        super(Module, self).__init__()

        self.id = id
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.layer = []

        for i in range(2, 6):
            self.modules[i] = []

            # separable convolution 3*3
            for conv in range(2):
                kernels_per_layer = 0
                depthwise_conv = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=3, padding=1, groups=in_channels)
                pointwise_conv = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)
                self.layers.append([depthwise_conv, pointwise_conv, nn.ReLU()])

