from torch import optim, nn

from wnet import WNet
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import os
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import PIL
import numpy as np
import sys
from pathlib import Path


img_path = Path('./BSR/BSDS500/data/images/testset/')


def categorical_image(tensor, image, red_lines):
    newImage = np.zeros_like(image.numpy()).squeeze()
    newImage = np.moveaxis(newImage, 0, -1)
    categories = torch.argmax(torch.squeeze(tensor), 0).numpy()
    for x in range(np.max(categories) + 1):
        matches = categories == x
        if matches.any():
            image_colors = np.stack((image[0, 0][matches],image[0, 1][matches],image[0, 2][matches]))
            color = np.mean(image_colors, axis=1)
            newImage[matches] = color
    if red_lines:
        img_sobel = np.roll(categories,1, axis=0)!=categories
        newImage[img_sobel] = [1, 0, 0]
        img_sobel = np.roll(categories,1, axis=1)!=categories
        newImage[img_sobel] = [1, 0, 0]
    return newImage


# Use this for testing / evaluation on an image
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(img_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    wnet = WNet(3, 5)
    wnet.to(device)
    max_epoch = 5
    optimizer = optim.SGD(wnet.parameters(), lr=0.01, momentum=0.9)
    optimizer2 = optim.SGD(wnet.parameters(), lr=0.01, momentum=0.9)
    
    wnet.load_state_dict(torch.load(sys.argv[1]))
    
    image, label = next(iter(dataloader))
    image = image.to(device)
    out1, outcrf, out2, loss = wnet.train(image, optimizer, optimizer2, True)
    print(f"Loss: {loss}")
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(transforms.ToPILImage()(torch.squeeze(image.cpu())))
    axarr[1].imshow(categorical_image(out1.cpu(), image.cpu(), True))
    axarr[2].imshow(categorical_image(outcrf.cpu(), image.cpu(), True))
    axarr[3].imshow(transforms.ToPILImage()(torch.squeeze(out2.cpu())))
    plt.show()
