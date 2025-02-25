from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import cv2
import SimpleITK as sitk
from torchvision.transforms import Compose,ToTensor,Normalize,ToPILImage
class Train_Dataset(dataset):
    def __init__(self, image_path):
        self.path = image_path
        self.filename_list = os.listdir(self.path)
        self.transforms = Compose([
            ToTensor(),
            # Normalize(0.5,0.5)
        ])

    def __getitem__(self, index):
        item_name = os.listdir(os.path.join(self.path,self.filename_list[index]))
        label = sitk.ReadImage(os.path.join(self.path,  self.filename_list[index], item_name[1]))
        label = sitk.GetArrayFromImage(label).astype(np.uint8)

        label = torch.FloatTensor(label).unsqueeze(0)
        return label

    def __len__(self):
        return len(self.filename_list)

if __name__ == "__main__":
    # sys.path.append('/ssd/lzq/3DUNet')
    # from config import args
    train_ds = Train_Dataset(r"F:\BraTS2020\diffusion")

    # 定义数据加载
    train_dl = DataLoader(train_ds, 4, False, num_workers=1)

    for i, image in enumerate(train_dl):
        print(i,image.size())
        break