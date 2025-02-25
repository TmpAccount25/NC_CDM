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
        self.filename_list = os.listdir(os.path.join(self.path,"real"))
        # self.transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Resize((96, 96)),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.Normalize([0.5], [0.5])
        # ])

    def __getitem__(self, index):
        # print(os.path.join(self.path, "real", self.filename_list[index]))
        # print(os.path.join(self.path, "imag", self.filename_list[index]))
        image_1 = cv2.imread(os.path.join(self.path, "real", self.filename_list[index]), -1)
        image_2 = cv2.imread(os.path.join(self.path, "imag", self.filename_list[index]), -1)

        image_1 = torch.FloatTensor(image_1/255).unsqueeze(0)
        image_2 = torch.FloatTensor(image_2/255).unsqueeze(0)
        # print(image_1.dtype)

        image = torch.cat((image_1,image_2),dim=0)
        return image

    def __len__(self):
        # print(len(os.path.join(self.path,"H")))
        return len(os.listdir(os.path.join(self.path,"real")))


if __name__ == "__main__":
    # sys.path.append('/ssd/lzq/3DUNet')
    # from config import args
    train_ds = Train_Dataset(r"E:\Diffusion_Complex_2D\recon_2d")
    import matplotlib.pyplot as plt
    # 定义数据加载
    train_dl = DataLoader(train_ds, 16, False)

    for i, image in enumerate(train_dl):
        print(i, image.size())
        img = image.cpu().detach().numpy()
        # grid = make_grid(img[0][0][8])
        # grid_img = trans.ToPILImage(grid)
        print(img.shape)
        plt.imshow(img[2][0],cmap="Greys")
        plt.show()
        plt.imshow(img[2][1],cmap="Greys")
        plt.show()
        break