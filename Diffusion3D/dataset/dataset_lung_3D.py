from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
# from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import cv2
import SimpleITK as sitk
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize, RandomResize,RandomCrop3D
import torchvision.transforms as trans
import matplotlib.pyplot as plt

class Train_Dataset(dataset):
    def __init__(self, image_path):
        self.path = image_path
        self.image_filename_list = os.listdir(os.path.join(self.path,"data"))
        self.label_filename_list = os.listdir(os.path.join(self.path, "label"))
        self.transforms = Compose([
            RandomFlip_LR(prob=0.5),
            RandomFlip_UD(prob=0.5),
        ])
    def __getitem__(self, index):
        image = sitk.ReadImage(os.path.join(self.path, "data", self.image_filename_list[index]))
        image = sitk.GetArrayFromImage(image).transpose([3,0,1,2])
        image = torch.FloatTensor(image)
        label = sitk.ReadImage(os.path.join(self.path, "label", self.label_filename_list[index]))
        label = sitk.GetArrayFromImage(label)
        label = torch.FloatTensor(label).unsqueeze(0)
        # image, label = self.transforms(image,label)
        return image, label.squeeze(0)

    def __len__(self):
        # print(len(os.listdir(os.path.join(self.path,"data"))))
        return len(os.listdir(os.path.join(self.path,"data")))


if __name__ == "__main__":
    train_ds = Train_Dataset(r"H:\Diffusion3D\lung")
    from torchvision.utils import make_grid
    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)
    for i, image in enumerate(train_dl):
        print(i,image[0].size(),image[1].size())
        img = image[0].cpu().detach().numpy()
        # grid = make_grid(img[0][0][8])
        # grid_img = trans.ToPILImage(grid)
        print(img[0][0][8].shape)
        plt.imshow(img[1][1][8])
        plt.show()
        break