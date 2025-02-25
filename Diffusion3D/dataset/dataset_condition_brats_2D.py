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
        self.filename_list = os.listdir(os.path.join(self.path,"flair"))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((240, 240)),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __getitem__(self, index):
        image_1 = cv2.imread(os.path.join(self.path, "flair", self.filename_list[index]), -1)
        image_2 = cv2.imread(os.path.join(self.path, "t1", self.filename_list[index]), -1)
        image_3 = cv2.imread(os.path.join(self.path, "t1ce", self.filename_list[index]), -1)
        image_4 = cv2.imread(os.path.join(self.path, "t2", self.filename_list[index]), -1)
        label = cv2.imread(os.path.join(self.path, "label", self.filename_list[index]), -1)
        # image = np.zeros([240,240,4])
        # print(image[0].shape)
        # image[:,:,0] = image_1
        # image[:,:,1] = image_2
        image_1 = self.transforms(image_1)
        image_2 = self.transforms(image_2)
        image_3 = self.transforms(image_3)
        image_4 = self.transforms(image_4)
        label = self.transforms(label)
        # print(image.shape)
        # image = self.transforms(image)
        image = torch.cat((image_1,image_2,image_3,image_4),dim=0)
        return image,label

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