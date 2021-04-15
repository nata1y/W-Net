import math

import numpy as np
import torch
import torch.nn as nn
from module import Module
import torch.nn.functional as F
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


class EncoderDecoder(nn.Module):

    def __init__(self, in_channels, n_features):
        super(EncoderDecoder, self).__init__()

        self.n_features = n_features
        self.up_convs = {}

        self.add_module("enc0", Module(in_channels=in_channels, out_channels=64, separable=False))
        self.add_module("max_pool_2x2_0", nn.MaxPool2d(2))

        in_channels = 64
        self.depth = 4

        # down sampling
        for i in range(1, self.depth):
            self.add_module(f"enc{i}", Module(in_channels=in_channels, out_channels=(in_channels * 2)))
            self.add_module(f"max_pool_2x2_{i}", nn.MaxPool2d(2))
            in_channels *= 2

        self.add_module("middle", Module(in_channels=in_channels, out_channels=(in_channels * 2)))

        # up sampling
        for i in range(self.depth)[::-1]:
            self.add_module(f"up_conv2x2_{i}",
                            nn.ConvTranspose2d(in_channels=(in_channels * 2), out_channels=in_channels, kernel_size=2,
                                               stride=2))

            self.add_module(f"dec{i}",
                            Module(in_channels=(in_channels * 2), out_channels=in_channels, separable=(i != 0)))
            in_channels //= 2

        # placeholder until know amount of classes k?
        self.add_module("conv1x1", nn.Conv2d(in_channels * 2, n_features, 1))

    def forward(self, x):
        stored_values = []
        for i in range(self.depth):
            x = getattr(self, f"enc{i}").forward(x)
            stored_values.append(x)
            x = getattr(self, f"max_pool_2x2_{i}").forward(x)

        x = self.middle.forward(x)

        for i in range(self.depth)[::-1]:
            x = getattr(self, f"up_conv2x2_{i}").forward(x)
            x = torch.cat((stored_values[i], x), 1)
            x = getattr(self, f"dec{i}").forward(x)

        x = self.conv1x1.forward(x)
        return x


def createKernel(sigma, r):
    kernel = torch.zeros((1, 1, 2 * r + 1, 2 * r + 1), requires_grad=False)
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            kernel[0, 0, i, j] = math.exp(-1 * ((r - i) ** 2 + (r - j) ** 2) / (sigma ** 2))
            if np.sqrt((r - i) ** 2 + (r - j) ** 2) > r:
                kernel[0, 0, i, j] = 0

    return kernel / torch.sum(kernel)


kernel1d = createKernel(4, 5)
kernel3d = torch.cat([kernel1d, kernel1d, kernel1d], 1)
kernel3d = torch.cat([kernel3d, kernel3d, kernel3d], 0)
kernel3d.requires_grad = False


def crf(softmax_outputs, inputs):
    result = torch.zeros(softmax_outputs.shape)
    idx = 0
    for input in inputs:
        # unary
        u = unary_from_softmax(softmax_outputs[idx].cpu().detach().numpy()).reshape(softmax_outputs.shape[1], -1)

        # pairwise
        p = create_pairwise_bilateral(sdims=(25, 25), schan=(0.05, 0.05), img=input.cpu().detach().numpy(), chdim=0)

        crf = dcrf.DenseCRF2D(inputs.shape[3], inputs.shape[2], softmax_outputs.shape[1])
        # unary potential
        crf.setUnaryEnergy(u)
        # + pairwise potential
        crf.addPairwiseEnergy(p, compat=100)
        Q = crf.inference(10)
        print(Q)
        result[idx] = torch.tensor(np.array(Q).reshape((-1, inputs.shape[2], inputs.shape[3])))
        idx += 1

    return result


def globalPb():
    pass


def contour2ucm():
    pass


def NCutLoss2D(labels, inputs, num_features=224, sigma_x=4, sigma_i=10, r=5):
    num_classes = labels.shape[1]
    loss = 0

    weights = torch.nn.functional.conv2d(inputs, weight=kernel3d, padding=r)
    weights.requires_grad = False

    # Calculate the actual weight matrix by calculating the pixel distances
    weights2 = torch.exp(torch.norm(inputs - weights, p=2, dim=1).pow(2).mul(-1 / (sigma_i ** 2))).unsqueeze(1)
    weights2.requires_grad = False

    for k in range(num_classes):
        class_probs = labels[:, k]
        numerator = torch.sum(class_probs * torch.nn.functional.conv2d(
            class_probs * weights2, weight=kernel1d, padding=r))
        # print("NUMERATOR ->", numerator)
        denominator = torch.sum(class_probs * torch.nn.functional.conv2d(
            weights2, weight=kernel1d, padding=r))
        # print("DENOMINATOR ->", denominator)
        loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))
        # print("LOSS ->", loss)

    return num_classes - loss


class WNet(nn.Module):

    def __init__(self, in_channels, n_features):
        super(WNet, self).__init__()

        self.n_features = n_features
        self.U_enc = EncoderDecoder(in_channels, n_features)
        self.soft_max = nn.Softmax2d()
        self.U_dec = EncoderDecoder(n_features, 3)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.U_enc.forward(x)
        output_enc = self.soft_max.forward(x)
        output_dec = self.U_dec.forward(output_enc)
        return output_dec

    def test(self, image):
        x = self.U_enc.forward(image)
        output_enc = self.soft_max.forward(x)
        ncutloss = NCutLoss2D(output_enc, image)
        output_dec = self.forward(image)
        loss = self.criterion(output_dec, image)
        return ncutloss, loss

    def train(self, image, optimizer, optimizer2, doCRF):
        x = self.U_enc.forward(image)
        output_enc = self.soft_max.forward(x)
        # TODO: change type of loss here!
        ncutloss = NCutLoss2D(output_enc, image)
        output_crf = None

        if doCRF:
            output_crf = crf(output_enc, image)
        output_dec = self.forward(image)

        loss = self.criterion(output_dec, image)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return output_enc, output_crf, output_dec, loss
