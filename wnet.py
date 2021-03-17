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
        # self.enc1 = Module(in_channels=in_channels, out_channels=64, separable=False)
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
            # self.modules[i] = Module(in_channels=(in_channels * 2), out_channels=in_channels, separable=(i!=9))
            self.add_module(f"up_conv2x2_{i}",
                            nn.ConvTranspose2d(in_channels=(in_channels * 2), out_channels=in_channels, kernel_size=2,
                                               stride=2))

            self.add_module(f"dec{i}",
                            Module(in_channels=(in_channels * 2), out_channels=in_channels, separable=(i != 0)))
            # self.up_convs[i] = nn.ConvTranspose2d(in_channels=(in_channels*2), out_channels=in_channels,
            #                                      kernel_size=2, stride=2)
            in_channels //= 2

        # stride, padding?
        # self.module_down_conn = nn.MaxPool2d(2)

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
# kernel3d = torch.cat([kernel1d, kernel1d, kernel1d], 0)
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


def NCutLoss2D_Good(labels, inputs, num_features=224, sigma_x=4, sigma_i=10, r=5):
    r"""Computes the continuous N-Cut loss, given a set of class probabilities (labels) and raw images (inputs).
    Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights
    relative to the class-wide average, rather than for every individual pixel.
    :param labels: Predicted class probabilities
    :param inputs: Raw images
    :return: Continuous N-Cut loss
    """
    num_classes = labels.shape[1]
    #kernel = gaussian_kernel(radius=self.radius, sigma=self.sigma_1, device=labels.device.type)
    loss = 0

    for k in range(num_classes):
        # Compute the average pixel value for this class, and the difference from each pixel
        class_probs = labels[:, k].unsqueeze(1)
        class_mean = torch.mean(inputs * class_probs, dim=(2, 3), keepdim=True) / \
            torch.add(torch.mean(class_probs, dim=(2, 3), keepdim=True), 1e-5)
        diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

        # Weight the loss by the difference from the class average.
        weights = torch.exp(diff.pow(2).mul(-1 / sigma_i ** 2))

        # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
        numerator = torch.sum(class_probs * F.conv2d(class_probs * weights, kernel1d, padding=r))
        denominator = torch.sum(class_probs * F.conv2d(weights, kernel1d, padding=r))
        loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))

    return num_classes - loss


def NCutLoss2D(labels, inputs, num_features=224, sigma_x=4, sigma_i=10, r=5):
    num_classes = labels.shape[1]
    loss = 0
    # print(inputs.shape)
    # print(labels.shape)

    # this should give us an image with 3 channels
    # where each channel is added according to a gaussian blur
    # blocks = torch.nn.functional.unfold(inputs, (2*r+1, 2*r+1), padding=r)
    # blocks = blocks.reshape((2*r+1)*(2*r+1), 3, 224, 224)

    weights = torch.nn.functional.conv2d(inputs, weight=kernel3d, padding=r)
    weights.requires_grad = False

    # Calculate the actual weight matrix by calculating the pixel distances
    weights2 = torch.exp(torch.norm(inputs - weights, p=2, dim=1).pow(2).mul(-1 / (sigma_i ** 2))).unsqueeze(1)
    weights2.requires_grad = False

    for k in range(num_classes):
        class_probs = labels[:, k]

        # diff_pixel_values = torch.nn.functional.pairwise_distance(class_probs.reshape(class_probs.shape[1],
        #                                                                     class_probs.shape[2], 1),
        #                                                 class_probs.reshape(class_probs.shape[1],
        #                                                                     class_probs.shape[2], 1))
        diff_pixel_values = torch.cdist(class_probs.reshape(class_probs.shape[1],
                                                                            class_probs.shape[2], 1),
                                                        class_probs.reshape(class_probs.shape[1],
                                                                            class_probs.shape[2], 1))

        weights2 = torch.exp(diff_pixel_values.pow(2).mul(-1 / sigma_i ** 2))
        result = torch.zeros(class_probs.shape)
        result2 = torch.zeros(class_probs.shape)
        print(result.shape)
        for i in range(112):
            for j in range(112):
                result[0, i, j] = class_probs[0, i, j] * weights2[i, i, j]
                result2[0, i, j] = weights2[i, i, j]
        print(result.shape)
        print(class_probs.shape)

        result = result.unsqueeze(0)
        result2 = result2.unsqueeze(0)
        print(torch.nn.functional.conv2d(result, weight=kernel1d, padding=r).shape)
        # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
        numerator = torch.sum(class_probs * torch.nn.functional.conv2d(result, weight=kernel1d, padding=r))
        # print("NUMERATOR ->", numerator)
        denominator = torch.sum(class_probs * torch.nn.functional.conv2d(result2, weight=kernel1d, padding=r))
        # print("DENOMINATOR ->", denominator)
        loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))
        print("LOSS ->", loss)

    print("LOSS:", num_classes - loss)
    return num_classes - loss


class WNet(nn.Module):

    def __init__(self, in_channels, n_features):
        super(WNet, self).__init__()

        self.n_features = n_features
        self.U_enc = EncoderDecoder(in_channels, n_features)
        self.soft_max = nn.Softmax2d()
        self.U_dec = EncoderDecoder(n_features, 3)
        self.criterion = nn.MSELoss()

    def soft_norm_cut_loss(self, output, image, k):
        j = k - torch.sum()

    def forward(self, x):
        x = self.U_enc.forward(x)
        output_enc = self.soft_max.forward(x)
        output_dec = self.U_dec.forward(output_enc)
        # output_dec = self.sigmoid.forward(output_dec)
        return output_dec

    def train(self, image, optimizer, optimizer2, doCRF):
        # output_enc = self.soft_max(self.U_enc.forward(image))
        # n_cut_loss = gradient_regularization(output_enc)*0.5
        # n_cut_loss.backward()
        # ptimizer.step()
        # ptimizer.zero_grad()
        # print(image)
        x = self.U_enc.forward(image)
        output_enc = self.soft_max.forward(x)
        # output_enc, output_dec = self.forward(image)
        ncutloss = NCutLoss2D(output_enc, image)
        ncutloss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        output_crf = None
        if doCRF:
            output_crf = crf(output_enc, image)
        output_dec = self.forward(image)
        # print(image)
        # print(NCutLoss2D(output_enc, image).item())
        # print(image)
        # print(output_dec)
        # print(output_dec)

        # loss = torch.mean(torch.pow(torch.pow(image, 2) + torch.pow(output_dec, 2), 0.5))*(1-0.5)
        # loss = torch.pow(torch.norm(image - output_dec, p=2), 2)
        # print(loss)

        # print('Our loss: ', loss)
        # print('Their loss: ', rec_loss)
        # l1 = nn.L1Loss(reduction='sum')
        loss = self.criterion(output_dec, image)
        # loss = torch.pow(loss, 2)
        # print('And well known: ', loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return output_enc, output_crf, output_dec, loss
