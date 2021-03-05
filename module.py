import numpy as np
import torch
import torch.nn as nn


class Module(nn.Module):

    def __init__(self, in_channels, out_channels, separable=True):
        super(Module, self).__init__()

        self.separable = separable
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = None
        
        if self.separable:
            # separable convolution 3*3
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                #self.layers.extend([depthwise_conv, nn.ReLU(), pointwise_conv, nn.ReLU()])
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
                #self.layers.extend([depthwise_conv2, nn.ReLU(), pointwise_conv2, nn.ReLU()])
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
                
    def forward(self, x):
        return self.block.forward(x)
