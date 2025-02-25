from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import cv2
import matplotlib.pyplot as plt
class Train_Dataset(dataset):
    def __init__(self, image_path):
        self.path = image_path
        self.filename_list = os.listdir(os.path.join(self.path,"flair"))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.transforms_label = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ])

    def __getitem__(self, index):
        label = cv2.imread(os.path.join(self.path, "label", self.filename_list[index]),-1)
        shape_1 = cv2.imread(os.path.join(self.path, "shape", self.filename_list[index]),-1)
        label = self.transforms_label((label//62).astype(np.uint8))
        shape_1 = self.transforms_label(shape_1.astype(np.uint8))
        # sample = torch.randn(4, 240, 240)
        # sample = torch.cat((sample,label),dim=0)
        # print(sample.shape)
        label = torch.cat((label,shape_1),dim=0)
        return label

    def __len__(self):
        return len(os.path.join(self.path,"flair"))


if __name__ == "__main__":
    # sys.path.append('/ssd/lzq/3DUNet')
    # from config import args
    train_ds = Train_Dataset(r"E:\HLX\PythonProject\Diffusion_2D_BraTS\BraTS2020_2D")

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, image in enumerate(train_dl):
        print(i,image.size())