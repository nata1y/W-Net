from torch import optim

from wnet import WNet
import torch
from torch.utils import data
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path

img_path = Path('./BSR/BSDS500/data/images/trainset/')
test_path = Path('./BSR/BSDS500/data/images/testset/')


def categorical_image(tensor, image, red_lines):
    newImage = np.zeros_like(image.numpy()).squeeze()
    newImage = np.moveaxis(newImage, 0, -1)
    categories = torch.argmax(torch.squeeze(tensor), 0).numpy()
    for x in range(np.max(categories) + 1):
        matches = categories == x
        if matches.any():
            image_colors = np.stack((image[0, 0][matches], image[0, 1][matches], image[0, 2][matches]))
            color = np.mean(image_colors, axis=1)
            newImage[matches] = color
    if red_lines:
        img_sobel = np.roll(categories, 1, axis=0) != categories
        newImage[img_sobel] = [1, 0, 0]
        img_sobel = np.roll(categories, 1, axis=1) != categories
        newImage[img_sobel] = [1, 0, 0]
    return newImage


# Use this for training
# NOTE! this code is for running on cpu, need some modifications for gpu mode
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    print(img_path)

    dataset = datasets.ImageFolder(img_path, transform=transform)
    testdataset = datasets.ImageFolder(img_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    testdataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # the second parameter is k, can be modified freely
    wnet = WNet(3, 5)
    wnet.to(device)
    print(wnet)
    max_epoch = 100
    optimizer = optim.Adam(wnet.parameters())
    optimizer2 = optim.Adam(wnet.parameters(), lr=0.001)

    for epoch in range(max_epoch):
        for batch_idx, (image, labels) in enumerate(dataloader):
            image = image.to(device)
            out1, outcrf, out2, loss = wnet.train(image, optimizer, optimizer2, batch_idx % 500 == 0)
        lozz = 0
        ncutlozz = 0
        for batch_idx, (image, labels) in enumerate(testdataloader):
            image = image.to(device)
            ncutloss, loss = wnet.test(image)
            lozz += loss.detach()
            ncutlozz += ncutloss.detach()
        print(f"NCutLoss: {ncutlozz}, Loss: {lozz}")
        torch.save(wnet.state_dict(), f"model_{epoch}_both_k=5.pth")