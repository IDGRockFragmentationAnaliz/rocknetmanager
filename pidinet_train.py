import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from rockedgesdetectors.pidinet.utils import cross_entropy_loss_RCF
from rockedgesdetectors import ModelPiDiNet
from rockedgesdetectors.dataset import Dataset

import matplotlib.pyplot as plt

from rockedgesdetectors.train import ModelTrain


def main():
    path_lst = Path('D:/1.ToSaver/profileimages/NYUD/image-train.lst')
    data = Dataset(path_lst=path_lst)

    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    checkpoint_path_7 = Path("models/pidinetmodels/table7_pidinet.pth")
    checkpoint_path_5 = Path("models/pidinetmodels/table5_pidinet.pth")
    image_path = Path("data/test.png")

    #model = ModelPiDiNet(checkpoint_path_7)
    model = ModelPiDiNet()

    conv_weights, bn_weights, relu_weights = model.get_weights()
    wd = 1e-4
    lr = 0.005

    param_groups = [{
        'params': conv_weights,
        'weight_decay': wd,
        'lr': lr}, {
        'params': bn_weights,
        'weight_decay': 0.1 * wd,
        'lr': lr}, {
        'params': relu_weights,
        'weight_decay': 0.0,
        'lr': lr
    }]

    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.99))
    # optimizer = torch.optim.SGD(param_groups, momentum=0.9)

    train_loader = DataLoader(data, batch_size=1, num_workers=4, shuffle=True)

    model.model.eval()
    result_1 = model(cv2.imread(str(image_path)))

    fig = plt.figure(figsize=(7, 9))
    axs = [fig.add_subplot(1, 1, 1)]
    axs[0].imshow(result_1)
    plt.show()

    ModelTrain.train(train_loader, model, optimizer)

    result_2 = model(cv2.imread(str(image_path)))
    model.model.eval()
    fig = plt.figure(figsize=(7, 9))
    axs = [fig.add_subplot(1, 2, 1),
           fig.add_subplot(1, 2, 2)]
    axs[0].imshow(result_1)
    axs[1].imshow(result_2)
    plt.show()


if __name__ == '__main__':
    main()


