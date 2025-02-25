from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import cv2
class Train_Dataset(dataset):
    def __init__(self, image_path):
        self.path = image_path
        self.filename_list = os.listdir(os.path.join(self.path,"H"))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((96, 96)),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __getitem__(self, index):
        image_1 = cv2.imread(os.path.join(self.path, "H", self.filename_list[index]), -1)
        image_2 = cv2.imread(os.path.join(self.path, "Xe", self.filename_list[index]), -1)

        image_1 = self.transforms(image_1)
        image_2 = self.transforms(image_2)
        print(image_1.dtype)
        image = torch.cat((image_1,image_2),dim=0)
        return image

    def __len__(self):
        # print(len(os.path.join(self.path,"H")))
        return len(os.listdir(os.path.join(self.path,"H")))


if __name__ == "__main__":
    # sys.path.append('/ssd/lzq/3DUNet')
    # from config import args
    train_ds = Train_Dataset(r"H:\Diffusion_2D\Lung")

    # 定义数据加载
    train_dl = DataLoader(train_ds, 1, False)

    for i, image in enumerate(train_dl):
        print(i,image.size())
        break