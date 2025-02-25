import time

import SimpleITK
import torchvision.utils
from diffusers import DDPMPipeline,DDPMScheduler
import cv2
import torch
import torchvision
from PIL import Image
import model
import numpy as np
import time
from model import model_Condition_3D
from dataset import dataset_condition_lung_3D_test
from torch.utils.data import DataLoader
def show_images(x):
    x = x * 0.5 + 0.5
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1,2,0).clip(0,1) *255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im
def show_volumns(x):
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu()
    image = SimpleITK.GetImageFromArray(grid_im)
    return image

if __name__ == "__main__":
    device = torch.device('cuda')
    dataset = dataset_condition_lung_3D_test.Train_Dataset(r"E:\Diffusion3D\lung")
    train_dl = DataLoader(dataset, 1, False, num_workers=1)
    # sample = torch.randn(1, 4, 136,184,144).to(device)
    model = model_Condition_3D().to(device)
    ckpt = torch.load(r"8_best.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['net'].items() if k.startswith('module.')})

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )
    flag = 0
    for step, batch in enumerate(train_dl):
        # start = time.time()
        clean_images = batch.to(device)
        sample = torch.randn(1, 2, 16, 96, 96).to(device)

        for i,t in enumerate(noise_scheduler.timesteps):
            print(t)
            with torch.no_grad():
                residual = model(sample,t,clean_images)
            sample = noise_scheduler.step(residual, t, sample).prev_sample
        time_flag = time.time()
        print(sample.shape)

        image = show_volumns(sample[0][0])
        SimpleITK.WriteImage(image, str(time_flag) +'_0'+'.nii.gz')
        image = show_volumns(sample[0][1])
        SimpleITK.WriteImage(image, str(time_flag) +'_1'+ '.nii.gz')
        # image = show_images(sample[0][2])
        # image.save(str(time_flag) + '_2' + '.png')
        # image = show_images(sample[0][3])
        # image.save(str(time_flag) + '_3' + '.png')

