import numpy as np
import torch
import torch.nn as nn
from module import Module


class WNet(nn.Module):

    def __init__(self, n_features):
        super(WNet, self).__init__()

        self.n_features = n_features
        self.modules = {1: Module(id=1, in_channels=3, out_channels=64, n_features=n_features)}
        self.up_convs = {}

        in_channels = 32

        # down sampling
        for i in range(2, 6):
            in_channels *= 2
            n_features /= 2
            self.modules[i] = Module(id=i, in_channels=in_channels, out_channels=(in_channels * 2),
                                     n_features=n_features)

        # up sampling
        for i in range(6, 10):
            self.modules[i] = Module(id=i, in_channels=(in_channels * 2), out_channels=in_channels,
                                     n_features=(n_features * 2))

            # means TO module i
            self.up_convs[i] = nn.ConvTranspose2d(in_channels=(in_channels*2), out_channels=(in_channels*2),
                                                  kernel_size=2)
            in_channels /= 2
            n_features *= 2

        # stride, padding?
        self.module_down_conn = nn.MaxPool2d(2)

        # placeholder until know amount of classes k?
        self.soft_max = None # nn.SoftMax()
        self.conv1x1 = nn.Conv2d(n_features, n_features, 1)

    def u_enc(self):
        pass

    def u_dec(self):
        pass
