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
# from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize, RandomResize,RandomCrop3D
import torchvision.transforms as trans
import matplotlib.pyplot as plt

class Train_Dataset(dataset):
    def __init__(self, image_path):
        self.path = image_path
        self.image_filename_list = os.listdir(self.path)

    def __getitem__(self, index):
        image = sitk.ReadImage(os.path.join(self.path, self.image_filename_list[index]))
        image = sitk.GetArrayFromImage(image).transpose([3,0,1,2])
        image = torch.FloatTensor(image)
        return image

    def __len__(self):
        return len(os.listdir(self.path))


if __name__ == "__main__":
    train_ds = Train_Dataset(r"H:\Diffusion_Complex\recon")
    from torchvision.utils import make_grid
    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)
    for i, image in enumerate(train_dl):
        print(i,image[0].size())
        img = image.cpu().detach().numpy()
        # grid = make_grid(img[0][0][8])
        # grid_img = trans.ToPILImage(grid)
        print(img.shape)
        plt.imshow(img[0][0][8])
        plt.show()
        break