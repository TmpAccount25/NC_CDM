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
        image_1 = sitk.ReadImage(os.path.join(self.path,  self.filename_list[index], item_name[0]))
        image_1 = sitk.GetArrayFromImage(image_1).transpose([1,2,0]).astype(np.float32)
        image_1 = image_1/np.max(image_1) *2 -1
        image_2 = sitk.ReadImage(os.path.join(self.path,  self.filename_list[index], item_name[2]))
        image_2 = sitk.GetArrayFromImage(image_2).transpose([1,2,0]).astype(np.float32)
        image_2 = image_2 / np.max(image_2) *2 -1
        image_3 = sitk.ReadImage(os.path.join(self.path,  self.filename_list[index], item_name[3]))
        image_3 = sitk.GetArrayFromImage(image_3).transpose([1,2,0]).astype(np.float32)
        image_3 = image_3 / np.max(image_3) *2 -1
        image_4 = sitk.ReadImage(os.path.join(self.path,  self.filename_list[index], item_name[4]))
        image_4 = sitk.GetArrayFromImage(image_4).transpose([1,2,0]).astype(np.float32)
        image_4 = image_4 / np.max(image_4) *2 -1
        label = sitk.ReadImage(os.path.join(self.path,  self.filename_list[index], item_name[1]))
        label = sitk.GetArrayFromImage(label).astype(np.uint8)

        image_1 = self.transforms(image_1).unsqueeze(0)
        image_2 = self.transforms(image_2).unsqueeze(0)
        image_3 = self.transforms(image_3).unsqueeze(0)
        image_4 = self.transforms(image_4).unsqueeze(0)

        label = torch.FloatTensor(label).unsqueeze(0)
        image = torch.cat((image_1,image_2,image_3,image_4),dim=0)
        return image,label

    def __len__(self):
        return len(self.filename_list)


if __name__ == "__main__":
    # sys.path.append('/ssd/lzq/3DUNet')
    # from config import args
    train_ds = Train_Dataset(r"F:\BraTS2020\diffusion")

    # 定义数据加载
    train_dl = DataLoader(train_ds, 4, False, num_workers=1)

    for i, image in enumerate(train_dl):
        print(i,image[0].size(),image[1].size())
        break